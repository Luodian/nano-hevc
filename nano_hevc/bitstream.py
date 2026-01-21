"""
Bitstream writer for HEVC NAL units.

Provides bit-level output with support for:
- Arbitrary bit writes (1-32 bits)
- Exponential-Golomb codes (ue(v), se(v))
- Byte alignment
- Start code emulation prevention (0x000003 stuffing)
"""

from __future__ import annotations
from typing import List
import numpy as np


class BitstreamWriter:
    """
    Bit-level output buffer for HEVC bitstream generation.

    Accumulates bits in a byte buffer with MSB-first ordering.
    Handles HEVC-required emulation prevention bytes.
    """

    __slots__ = ("_buffer", "_byte", "_bits_in_byte")

    def __init__(self):
        self._buffer: List[int] = []
        self._byte: int = 0
        self._bits_in_byte: int = 0

    def write_bits(self, value: int, num_bits: int) -> None:
        """Write num_bits from value (MSB first)."""
        if num_bits == 0:
            return
        if num_bits < 0:
            raise ValueError(f"num_bits must be non-negative, got {num_bits}")

        if num_bits > 32:
            high_bits = num_bits - 32
            self.write_bits(value >> 32, high_bits)
            self.write_bits(value & 0xFFFFFFFF, 32)
            return

        for i in range(num_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self._byte = (self._byte << 1) | bit
            self._bits_in_byte += 1

            if self._bits_in_byte == 8:
                self._buffer.append(self._byte)
                self._byte = 0
                self._bits_in_byte = 0

    def write_bit(self, value: int) -> None:
        """Write a single bit."""
        self.write_bits(value & 1, 1)

    def write_byte(self, value: int) -> None:
        """Write a full byte."""
        self.write_bits(value & 0xFF, 8)

    def write_ue(self, value: int) -> None:
        """
        Write unsigned Exp-Golomb code (ue(v)).

        Encoding: value -> binary code
        0 -> 1
        1 -> 010
        2 -> 011
        3 -> 00100
        etc.
        """
        if value < 0:
            raise ValueError(f"ue(v) requires non-negative value, got {value}")

        value_plus_one = value + 1
        num_bits = value_plus_one.bit_length()
        leading_zeros = num_bits - 1

        self.write_bits(0, leading_zeros)
        self.write_bits(value_plus_one, num_bits)

    def write_se(self, value: int) -> None:
        """
        Write signed Exp-Golomb code (se(v)).

        Mapping: 0->0, 1->1, -1->2, 2->3, -2->4, ...
        """
        if value <= 0:
            code_num = -2 * value
        else:
            code_num = 2 * value - 1
        self.write_ue(code_num)

    def byte_align(self) -> None:
        """Pad with zeros to byte boundary."""
        if self._bits_in_byte > 0:
            remaining = 8 - self._bits_in_byte
            self.write_bits(0, remaining)

    def write_rbsp_trailing_bits(self) -> None:
        """Write RBSP trailing bits (1 followed by zeros to byte align)."""
        self.write_bit(1)
        self.byte_align()

    def get_bytes(self) -> bytes:
        """Get the written data as bytes (flushes partial byte with zeros)."""
        result = self._buffer.copy()
        if self._bits_in_byte > 0:
            remaining = 8 - self._bits_in_byte
            result.append(self._byte << remaining)
        return bytes(result)

    def get_bytes_with_emulation_prevention(self) -> bytes:
        """
        Get bytes with emulation prevention (0x03) inserted.

        HEVC requires 0x03 after 0x0000 if followed by 0x00, 0x01, 0x02, or 0x03.
        """
        raw = self.get_bytes()
        result = []
        zero_count = 0

        for byte in raw:
            if zero_count >= 2 and byte <= 0x03:
                result.append(0x03)
                zero_count = 0

            result.append(byte)

            if byte == 0x00:
                zero_count += 1
            else:
                zero_count = 0

        return bytes(result)

    @property
    def bit_count(self) -> int:
        """Total bits written so far."""
        return len(self._buffer) * 8 + self._bits_in_byte

    @property
    def byte_count(self) -> int:
        """Complete bytes written so far."""
        return len(self._buffer)


def write_start_code_prefix(writer: BitstreamWriter, long_prefix: bool = True) -> None:
    """Write NAL unit start code prefix (0x000001 or 0x00000001)."""
    if long_prefix:
        writer.write_byte(0x00)
    writer.write_byte(0x00)
    writer.write_byte(0x00)
    writer.write_byte(0x01)
