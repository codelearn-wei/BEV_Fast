#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Send frame packet json files to realtime tcp source.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=17999)
    parser.add_argument('--packet-dir', required=True)
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--loop', action='store_true')
    return parser.parse_args()


def iter_packets(packet_dir: Path):
    return sorted(packet_dir.glob('*.json'), key=lambda p: p.name)


def main():
    args = parse_args()
    packet_dir = Path(args.packet_dir)
    if not packet_dir.exists():
        raise FileNotFoundError(f'packet directory not found: {packet_dir}')

    interval = 1.0 / max(args.fps, 1e-3)

    with socket.create_connection((args.host, args.port)) as conn:
        print(f'[sender] connected to {args.host}:{args.port}')
        while True:
            packets = iter_packets(packet_dir)
            if not packets:
                print('[sender] no packet json found')
                return

            for path in packets:
                line = path.read_text(encoding='utf-8').strip()
                conn.sendall(line.encode('utf-8') + b'\n')
                print(f'[sender] sent {path.name}')
                time.sleep(interval)

            if not args.loop:
                break


if __name__ == '__main__':
    main()
