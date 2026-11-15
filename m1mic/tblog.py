"""Minimal TensorBoard scalar writer (stdlib only: no tensorboard, tensorflow or torch).

Writes the standard event-file format (TFRecord framing with masked CRC32C around Event protos) to
``<log_dir>/events.out.tfevents.<time>.<host>.<pid>.0``.
"""

from __future__ import annotations

import os
import socket
import struct
import time


def _crc32c_table():
    t = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        t.append(c)
    return t


_T = _crc32c_table()


def _crc32c(b: bytes) -> int:
    c = 0xFFFFFFFF
    for x in b:
        c = _T[(c ^ x) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def _masked(b: bytes) -> int:
    c = _crc32c(b)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _len_field(num: int, payload: bytes) -> bytes:
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def _event(step: int, wall: float, summary: bytes = b"", version: str = "") -> bytes:
    e = struct.pack("<Bd", 0x09, wall) + _varint(2 << 3) + _varint(step)
    if version:
        e += _len_field(3, version.encode())
    if summary:
        e += _len_field(5, summary)
    return e


class TBWriter:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        name = f"events.out.tfevents.{int(time.time())}.{socket.gethostname()}.{os.getpid()}.0"
        self.path = os.path.join(log_dir, name)
        self.f = open(self.path, "ab")
        self._write(_event(0, time.time(), version="brain.Event:2"))

    def _write(self, data: bytes):
        n = struct.pack("<Q", len(data))
        self.f.write(n + struct.pack("<I", _masked(n)) + data + struct.pack("<I", _masked(data)))

    def scalars(self, step: int, values: dict[str, float], wall: float | None = None):
        """One event per call holding every finite scalar in ``values`` (tag -> value)."""
        summary = b""
        for tag, v in values.items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v != v or v in (float("inf"), float("-inf")):
                continue
            summary += _len_field(1, _len_field(1, tag.encode()) + struct.pack("<Bf", 0x15, v))
        if summary:
            self._write(_event(int(step), time.time() if wall is None else wall, summary))
            self.f.flush()

    def close(self):
        self.f.close()
