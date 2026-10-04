"""CPU-only backend lifecycle integration with explicit fake XPU loader boundaries."""
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, modeling_utils
from runtime.waystone.backend import XPUBackend
from runtime.waystone.norm_gain import _install
from verification.test_norm_gain import Norm, model as toy_model


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.stack = __import__('contextlib').ExitStack()
        self.addCleanup(self.stack.close)
        self.events = []
        self.m = toy_model()
        # Explicit loader seam: real CPU tensors exercise gain arithmetic; fake
        # parameter/device reports isolate backend orchestration from XPU.
        self.m.parameters = lambda: iter([SimpleNamespace(device=SimpleNamespace(type='xpu'))])
        self.m.config = SimpleNamespace(vision_config=SimpleNamespace(_attn_implementation='sdpa'),
                                        text_config=SimpleNamespace(_attn_implementation='eager'))
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side='right'),
                                         image_processor=SimpleNamespace(size=None))
        self.loader = self.stack.enter_context(patch.object(AutoModelForImageTextToText, 'from_pretrained', return_value=self.m))
        self.stack.enter_context(patch.object(AutoProcessor, 'from_pretrained', return_value=self.processor))
        self.stack.enter_context(patch('runtime.waystone.model_identity.verify', return_value={'verified': True}))
        self.stack.enter_context(patch('runtime.waystone.bootstrap.prepare_runtime'))
        properties = SimpleNamespace(device_id=0x56A0, is_integrated_gpu=False, platform_name='Level-Zero', name='fake', uuid='fake')
        for name, value in [('is_available', True), ('device_count', 1), ('get_device_properties', properties),
                            ('is_initialized', True), ('memory_allocated', 0), ('memory_reserved', 0), ('max_memory_allocated', 0)]:
            self.stack.enter_context(patch.object(torch.xpu, name, return_value=value))
        self.stack.enter_context(patch.object(torch.xpu, 'synchronize', side_effect=lambda: self.events.append('synchronize')))
        self.stack.enter_context(patch.object(torch.xpu, 'empty_cache', side_effect=lambda: self.events.append('empty_cache')))
        def install(m):
            return _install(m, Norm, ('first', 'second'), device_type='cpu')
        self.installer = self.stack.enter_context(patch('runtime.waystone.norm_gain.install_gain_cache', side_effect=install))

    def test_original_default_skips_cache(self):
        backend = XPUBackend('fake', attention_backend='hybrid')
        backend.load()
        self.installer.assert_not_called()
        self.assertEqual(backend.health()['normalization'], 'original')
        self.assertFalse(backend.health()['gain_cache']['active'])
        backend.unload()

    def test_load_publish_and_unload_restore(self):
        backend = XPUBackend('fake', attention_backend='hybrid', normalization='gain-only')
        warmup = modeling_utils.caching_allocator_warmup
        backend.load()
        backend.load()
        self.loader.assert_called_once()
        self.installer.assert_called_once()
        self.assertIs(modeling_utils.caching_allocator_warmup, warmup)
        owner = backend._gain_cache
        self.assertEqual(backend.health()['gain_cache'], {'active': True, 'modules': 2, 'bytes': 32})
        forward = self.m['first'].forward.__func__
        cells = dict(zip(forward.__code__.co_freevars, forward.__closure__))
        gain_ref = weakref.ref(cells['gain'].cell_contents)
        del cells, forward
        backend.unload()
        gc.collect()
        self.assertTrue(owner.closed)
        self.assertIsNone(gain_ref())
        self.assertIsNone(backend.model)
        self.assertIsNone(backend.processor)
        self.assertIsNone(backend._gain_cache)
        self.assertNotIn('forward', self.m['first'].__dict__)
        self.assertEqual(self.events, ['synchronize', 'empty_cache'])
        self.assertFalse(backend.health()['gain_cache']['active'])
        backend.unload()

    def test_install_failure_leaves_cold_backend_and_restores_warmup(self):
        backend = XPUBackend('fake', attention_backend='hybrid', normalization='gain-only')
        warmup = modeling_utils.caching_allocator_warmup
        self.installer.side_effect = RuntimeError('injected cache installation failure')
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            backend.load()
        self.assertIs(modeling_utils.caching_allocator_warmup, warmup)
        self.assertIsNone(backend.model)
        self.assertIsNone(backend.processor)
        self.assertIsNone(backend._gain_cache)
        self.assertEqual(self.events, ['synchronize', 'empty_cache'])

    def test_model_load_failure_restores_warmup(self):
        backend = XPUBackend('fake', normalization='gain-only')
        warmup = modeling_utils.caching_allocator_warmup
        self.loader.side_effect = RuntimeError('injected model load failure')
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            backend.load()
        self.assertIs(modeling_utils.caching_allocator_warmup, warmup)
        self.installer.assert_not_called()
        self.assertIsNone(backend.model)


if __name__ == '__main__':
    unittest.main()
