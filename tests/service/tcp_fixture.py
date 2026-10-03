"""Task-owned subprocess fixture: deterministic CPU backend and real TCP router."""
import argparse
import json
import threading
import time

import uvicorn

from chandra_service.router import create_router
from chandra_service.service import create_app


class FakeBackend:
    def __init__(self, events):
        self.events = events
        self.lock = threading.Lock()
        self.loaded = False

    def record(self, event):
        with self.lock, open(self.events, 'a', encoding='utf-8') as output:
            output.write(json.dumps({'event': event, 'time': time.monotonic()}) + '\n')

    def health(self):
        return {'state': 'ready' if self.loaded else 'cold'}

    def unload(self):
        if self.loaded:
            self.loaded = False
            self.record('unload')

    def generate(self, request, cancel):
        self.loaded = True
        self.record('start')
        yield 'first'
        # Deliberately do not honor cancellation: ownership must survive until stop.
        time.sleep(.8)
        self.record('finish')
        yield 'last'
        yield {'usage': {'prompt_tokens': 2, 'completion_tokens': 2, 'total_tokens': 4}, 'finish_reason': 'stop'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['backend', 'router'])
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--events')
    parser.add_argument('--upstream')
    args = parser.parse_args()
    if args.mode == 'backend':
        app = create_app(FakeBackend(args.events), queue_limit=0, idle_seconds=.2, inference_seconds=5)
    else:
        app = create_router({'backends': {'waystone': {'url': args.upstream}}, 'inference_seconds': 5, 'cooldown_seconds': .1})
    uvicorn.run(app, fd=args.fd, log_level='warning', access_log=False, timeout_graceful_shutdown=1)


if __name__ == '__main__':
    main()
