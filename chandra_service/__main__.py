"""Explicit service/router launcher; importing this module never loads model weights."""
import argparse
import asyncio
import socket
import ipaddress
import json
from pathlib import Path


def validate_host(host):
    address = ipaddress.ip_address(host)
    if not (address.is_private or address.is_loopback) or address.is_unspecified or address.is_multicast:
        raise ValueError('Use a concrete private or loopback address')
    return address


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['waystone', 'router'])
    parser.add_argument('--host', action='append', help='Repeat to bind loopback and explicit private LAN addresses in one process')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--model-path')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--queue-limit', type=int, default=2)
    parser.add_argument('--inference-seconds', type=float, default=1800)
    parser.add_argument('--idle-seconds', type=float, default=300)
    parser.add_argument('--max-pixels', type=int, default=4_000_000)
    parser.add_argument('--max-input-tokens', type=int, default=16384)
    parser.add_argument('--max-output-tokens', type=int, default=12384)
    parser.add_argument('--attention', choices=['hybrid', 'eager'], default='hybrid', help='Hybrid uses qualified vision SDPA and text eager attention; full SDPA failed cached decoding on the tested XPU')
    args = parser.parse_args()
    if args.max_pixels <= 0 or args.max_input_tokens <= 0 or not 1 <= args.max_output_tokens <= 12384:
        parser.error('Pixel/context limits must be positive and output limit must be 1..12384')
    hosts = args.host or ['127.0.0.1']
    if len(set(hosts)) != len(hosts):
        parser.error('--host addresses must be unique')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be 1..65535')
    try:
        for host in hosts: validate_host(host)
    except ValueError:
        parser.error('--host must be a concrete private or loopback IP address')
    if args.mode == 'waystone':
        if not args.model_path:
            parser.error('waystone requires --model-path to a locally provisioned model')
        from runtime.waystone import XPUBackend
        from .service import create_app
        backend = XPUBackend(args.model_path, max_pixels=args.max_pixels, max_input_tokens=args.max_input_tokens, max_output_tokens=args.max_output_tokens, attention_backend=args.attention)
        app = create_app(backend, queue_limit=args.queue_limit, inference_seconds=args.inference_seconds, idle_seconds=args.idle_seconds)
    else:
        if not args.config:
            parser.error('router requires --config')
        from .router import create_router
        app = create_router(json.loads(args.config.read_text()))
    import uvicorn
    sockets = []
    try:
        for host in hosts:
            address = ipaddress.ip_address(host)
            listener = socket.socket(socket.AF_INET6 if address.version == 6 else socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(listener)
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if address.version == 6:
                listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            listener.bind((host, args.port))
            listener.listen(128)
            listener.setblocking(False)
        # One app and event loop own all listeners and one shared capacity pool.
        server = uvicorn.Server(uvicorn.Config(app, access_log=False))
        asyncio.run(server.serve(sockets=sockets))
    finally:
        for listener in sockets: listener.close()

if __name__ == '__main__':
    main()
