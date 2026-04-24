from __future__ import annotations

import json
import queue
import socketserver
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Set

from .schemas import FramePacket


class FrameSource(ABC):

    @abstractmethod
    def next_frame(self, timeout_s: float = 1.0) -> Optional[FramePacket]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class DirectoryFrameSource(FrameSource):

    def __init__(self, frame_dir: Path, poll_interval_s: float = 0.2) -> None:
        self.frame_dir = frame_dir
        self.poll_interval_s = poll_interval_s
        self._consumed: Set[Path] = set()

    def next_frame(self, timeout_s: float = 1.0) -> Optional[FramePacket]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not self.frame_dir.exists():
                time.sleep(self.poll_interval_s)
                continue

            candidates = sorted(
                self.frame_dir.glob('*.json'),
                key=lambda p: (p.stat().st_mtime, p.name),
            )
            for json_path in candidates:
                if json_path in self._consumed:
                    continue
                self._consumed.add(json_path)
                return FramePacket.from_json_file(json_path)
            time.sleep(self.poll_interval_s)
        return None

    def close(self) -> None:
        return


class _QueueTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, frame_queue):
        super().__init__(server_address, RequestHandlerClass)
        self.frame_queue = frame_queue


class _FrameTCPHandler(socketserver.StreamRequestHandler):

    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                break
            line = raw.decode('utf-8').strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                packet = FramePacket.from_mapping(payload)
                self.server.frame_queue.put(packet)
            except Exception as exc:  # pylint: disable=broad-except
                print(f'[tcp_source] dropped invalid payload: {exc}')


class TCPFrameSource(FrameSource):

    def __init__(self, host: str, port: int) -> None:
        self._queue: queue.Queue[FramePacket] = queue.Queue()
        self._server = _QueueTCPServer((host, port), _FrameTCPHandler,
                                       self._queue)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f'[tcp_source] listening on {host}:{port}')

    def next_frame(self, timeout_s: float = 1.0) -> Optional[FramePacket]:
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)
