"""Model-free controls and the bundled, authenticated four-case prepared-output regression."""
import argparse
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from PIL import Image

from . import prepare as P


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "recorded-cli-four-case"
EXPECTED_SHA256 = "87eb20de6d273f2901c6b7e51afe96350b7220ce33a68cd4ba0048eb29f9e7aa"
CASES = ("native-0", "native-1", "scanned-0", "scanned-1")
CORPUS_RESULTS = []


def encoded_image(image, fmt="PNG", **options):
    target = io.BytesIO()
    image.save(target, format=fmt, **options)
    raw = target.getvalue()
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(raw).decode("ascii"), raw


def payload(url):
    return {"model": "chandra", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Transcribe this page."}]}]}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_bound(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Corpus fixture exceeds 4 MiB")
    return raw


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.url, self.png = encoded_image(Image.new("RGB", (64, 64), (17, 29, 41)))

    def test_geometry_profile(self):
        for original, expected in [((1624, 2100), (1536, 2016)), ((64, 96), (64, 96)), ((1, 1), (64, 64)), ((1, 200), (32, 800))]:
            with self.subTest(original=original):
                self.assertEqual(P.smart_resize(*original), expected)
        for original in [(0, 64), (-1, 64), (True, 64), (64.0, 64), (1, 201)]:
            with self.subTest(original=original), self.assertRaises(P.PreparationError):
                P.smart_resize(*original)

    def test_same_size_rgb_is_exact(self):
        pixels = (np.arange(96 * 64 * 3, dtype=np.uint32) % 251).astype(np.uint8).reshape(96, 64, 3)
        url, raw = encoded_image(Image.fromarray(pixels))
        image = P.prepare_image(url)
        with Image.open(io.BytesIO(image.png)) as prepared:
            self.assertEqual(prepared.size, (64, 96))
            self.assertEqual(prepared.tobytes(), pixels.tobytes())
        self.assertFalse(image.metadata["resized"])
        self.assertEqual(image.metadata["original_encoded_sha256"], sha(raw))
        self.assertEqual(image.metadata["prepared_rgb_sha256"], sha(pixels.tobytes()))

    def test_small_resize_preserves_constant_rgb(self):
        url, _ = encoded_image(Image.new("RGB", (1, 1), (17, 29, 41)))
        image = P.prepare_image(url)
        with Image.open(io.BytesIO(image.png)) as prepared:
            self.assertEqual(prepared.size, (64, 64))
            self.assertEqual(prepared.tobytes(), bytes((17, 29, 41)) * 4096)
        self.assertTrue(image.metadata["resized"])

    def test_supported_formats_preserve_decoded_rgb(self):
        for fmt in ("PNG", "JPEG", "WEBP"):
            with self.subTest(fmt=fmt):
                url, raw = encoded_image(Image.new("RGB", (64, 64), (17, 29, 41)), fmt)
                with Image.open(io.BytesIO(raw)) as original:
                    expected = original.convert("RGB").tobytes()
                result = P.prepare_image(url)
                with Image.open(io.BytesIO(result.png)) as actual:
                    self.assertEqual(actual.tobytes(), expected)
                self.assertEqual(result.metadata["original_format"], fmt)

    def test_existing_rgb_conversion_semantics(self):
        for mode, color, expected in [("L", 107, (107, 107, 107)), ("RGBA", (17, 29, 41, 0), (17, 29, 41))]:
            with self.subTest(mode=mode):
                url, _ = encoded_image(Image.new(mode, (64, 64), color))
                result = P.prepare_image(url)
                with Image.open(io.BytesIO(result.png)) as actual:
                    self.assertEqual(actual.mode, "RGB")
                    self.assertEqual(actual.tobytes(), bytes(expected) * 4096)

    def test_request_preservation_and_allowance(self):
        for tokens in (None, 1, 12384):
            with self.subTest(tokens=tokens):
                request = payload(self.url)
                request.update(stream=True, stream_options={"include_usage": True}, top_p=0.37, temperature=0, caller_metadata={"kept": [1, 2]})
                request["messages"][0]["content"].reverse()
                if tokens is not None:
                    request["max_tokens"] = tokens
                before = copy.deepcopy(request)
                transformed, image = P.prepare_request(request)
                self.assertEqual(request, before)
                restored = copy.deepcopy(transformed)
                restored["messages"][0]["content"][1]["image_url"]["url"] = self.url
                self.assertEqual(restored, before)
                self.assertEqual(transformed["messages"][0]["content"][1]["image_url"]["url"], image.data_url)

    def test_bad_data_urls_and_media(self):
        for url in (None, "https://example.invalid/image.png", "data:image/gif;base64,AAAA", "data:image/png;base64,", "data:image/png;base64,!?", "data:image/png;base64,AAAA", self.url.replace("image/png", "image/jpeg"), "data:image/png;base64," + base64.b64encode(self.png[:40]).decode("ascii")):
            with self.subTest(url=str(url)[:50]), self.assertRaises(P.PreparationError):
                P.prepare_image(url)

    def test_multiframe_refused(self):
        url, _ = encoded_image(Image.new("RGB", (64, 64), "red"), save_all=True, append_images=[Image.new("RGB", (64, 64), "blue")], duration=100)
        with self.assertRaisesRegex(P.PreparationError, "still image"):
            P.prepare_image(url)

    def test_admission_precedes_resize(self):
        with patch.object(P, "resize_uint8_rgb") as resize, self.assertRaisesRegex(P.PreparationError, "Original image"):
            P.prepare_image(self.url, max_image_pixels=4095)
        resize.assert_not_called()
        for limit in (0, True, 6_291_457):
            with self.subTest(limit=limit), self.assertRaises(P.PreparationError):
                P.prepare_image(self.url, max_image_pixels=limit)

    def test_encoded_limit_precedes_decode(self):
        with patch.object(P, "MAX_BYTES", 32), patch.object(P.Image, "open") as decode:
            with self.assertRaisesRegex(P.PreparationError, "128 MiB"):
                P.prepare_request(payload(self.url))
        decode.assert_not_called()

    def test_pillow_decompression_limits(self):
        for limit in (100, 3000):
            with self.subTest(limit=limit), patch.object(Image, "MAX_IMAGE_PIXELS", limit), warnings.catch_warnings(record=True):
                warnings.simplefilter("always", Image.DecompressionBombWarning)
                with self.assertRaises(P.PreparationError):
                    P.prepare_image(self.url)
                self.assertEqual(Image.MAX_IMAGE_PIXELS, limit)

    def test_bad_request_shapes_and_allowance(self):
        cases = [None, {}, {"messages": []}, payload(self.url), payload(self.url), payload(self.url)]
        cases[3]["max_tokens"] = 12385
        cases[4]["max_tokens"] = True
        cases[5]["model"] = "other-model"
        for request in cases:
            with self.subTest(request=request), self.assertRaises(P.PreparationError):
                P.prepare_request(request)

    def test_no_torch_or_transformers_import(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("fastapi", sys.modules)
        self.assertNotIn("chandra_service", sys.modules)


class CorpusTests(unittest.TestCase):
    def test_four_prepared_outputs(self):
        expected_raw = read_bound(FIXTURES / "expected.json")
        self.assertEqual(sha(expected_raw), EXPECTED_SHA256)
        manifest = json.loads(expected_raw)
        self.assertEqual(manifest["schema"], "chandra.cpu-preparation-golden.v1")
        self.assertEqual(manifest["context_tokens"], 16384)
        self.assertEqual(manifest["recorded_prompt_tokens"], 3617)
        self.assertEqual(manifest["max_tokens"], 12384)
        self.assertEqual([row["case"] for row in manifest["cases"]], list(CASES))
        for row in manifest["cases"]:
            case = row["case"]
            with self.subTest(case=case):
                raw = read_bound(FIXTURES / f"{case}.request.json")
                self.assertEqual(len(raw), row["original_request_bytes"])
                self.assertEqual(sha(raw), row["original_request_sha256"])
                request = json.loads(raw)
                before = copy.deepcopy(request)
                transformed, image = P.prepare_request(request)
                serialized = json.dumps(transformed, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
                self.assertEqual(sha(serialized), row["prepared_request_sha256"])
                expected_metadata = manifest["common_metadata"] | row["metadata"]
                self.assertEqual({key: value for key, value in image.metadata.items() if key not in ("resize_ms", "total_ms")}, expected_metadata)
                self.assertEqual(len(image.png), expected_metadata["prepared_png_bytes"])
                self.assertEqual(sha(image.png), expected_metadata["prepared_png_sha256"])
                with Image.open(io.BytesIO(image.png)) as prepared:
                    self.assertEqual(prepared.mode, "RGB")
                    self.assertEqual(list(prepared.size), expected_metadata["prepared_dimensions"])
                    self.assertEqual(sha(prepared.tobytes()), expected_metadata["prepared_rgb_sha256"])
                self.assertEqual(transformed["max_tokens"], 12384)
                self.assertEqual(request, before)
                restored = copy.deepcopy(transformed)
                original_part = next(part for part in before["messages"][0]["content"] if part["type"] == "image_url")
                prepared_part = next(part for part in restored["messages"][0]["content"] if part["type"] == "image_url")
                self.assertEqual(prepared_part["image_url"]["url"], image.data_url)
                prepared_part["image_url"]["url"] = original_part["image_url"]["url"]
                self.assertEqual(restored, before)
                CORPUS_RESULTS.append({"case": case, "request_sha256": sha(serialized), "metadata": image.metadata})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("Report already exists")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if args.report:
        package = Path(__file__).resolve().parent
        report = {"successful": result.wasSuccessful(), "tests_run": result.testsRun, "skipped": len(result.skipped), "failures": len(result.failures), "errors": len(result.errors), "corpus": CORPUS_RESULTS, "fixture_manifest_sha256": sha(read_bound(FIXTURES / "expected.json")), "source_sha256": {path.name: sha(path.read_bytes()) for path in sorted(package.glob("*.py"))}}
        with args.report.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
