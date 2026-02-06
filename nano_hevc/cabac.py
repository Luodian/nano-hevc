"""
CABAC (Context-Adaptive Binary Arithmetic Coding) for HEVC.

Implements the entropy coding engine per ITU-T H.265 Section 9.3.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from nano_hevc.bitstream import BitstreamWriter


CABAC_LPS_RANGE = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [128, 150, 178, 205],
    [128, 142, 169, 195],
    [128, 135, 160, 185],
    [128, 128, 152, 175],
    [128, 122, 144, 166],
    [128, 116, 137, 158],
    [128, 110, 130, 150],
    [128, 104, 123, 142],
    [128, 99, 117, 135],
    [128, 94, 111, 128],
    [128, 89, 105, 122],
    [128, 85, 100, 116],
    [128, 80, 95, 110],
    [128, 76, 90, 104],
    [128, 72, 85, 99],
    [128, 69, 81, 94],
    [128, 65, 77, 89],
    [128, 62, 73, 85],
    [128, 59, 69, 80],
    [128, 56, 66, 76],
    [128, 53, 63, 72],
    [128, 51, 59, 69],
    [128, 48, 56, 65],
    [128, 46, 53, 62],
    [128, 43, 50, 58],
    [128, 41, 48, 55],
    [128, 39, 45, 52],
    [128, 37, 43, 50],
    [128, 35, 41, 47],
    [128, 33, 39, 45],
    [128, 32, 37, 43],
    [128, 30, 35, 40],
    [128, 29, 33, 38],
    [128, 27, 31, 36],
    [128, 26, 30, 35],
    [128, 24, 29, 33],
    [128, 23, 27, 31],
    [128, 22, 26, 30],
    [128, 21, 25, 28],
    [128, 20, 23, 27],
    [128, 19, 22, 25],
    [128, 18, 21, 24],
    [128, 17, 20, 23],
    [128, 16, 19, 22],
    [128, 15, 18, 21],
    [128, 14, 17, 20],
    [128, 14, 16, 19],
    [128, 13, 15, 18],
    [128, 12, 14, 17],
    [128, 12, 14, 16],
    [128, 11, 13, 15],
    [128, 11, 12, 14],
    [128, 10, 12, 14],
    [128, 10, 11, 13],
    [128, 9, 11, 12],
    [128, 9, 10, 12],
    [128, 8, 10, 11],
    [128, 8, 9, 11],
    [128, 7, 9, 10],
    [128, 7, 8, 10],
    [128, 7, 8, 9],
]

CABAC_TRANS_LPS = [
    0,
    0,
    1,
    2,
    2,
    4,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    11,
    11,
    12,
    13,
    13,
    15,
    15,
    16,
    16,
    18,
    18,
    19,
    19,
    21,
    21,
    22,
    22,
    23,
    24,
    24,
    25,
    26,
    26,
    27,
    27,
    28,
    29,
    29,
    30,
    30,
    30,
    31,
    32,
    32,
    33,
    33,
    33,
    34,
    34,
    35,
    35,
    35,
    36,
    36,
    36,
    37,
    37,
    37,
    38,
    38,
    63,
]

CABAC_TRANS_MPS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    62,
    63,
]


class CabacContext:
    """Single CABAC context with state and MPS."""

    __slots__ = ("state", "mps")

    def __init__(self, init_value: int = 64):
        slope = (init_value >> 4) * 5 - 45
        offset = ((init_value & 15) << 3) - 16
        init_state = min(max(1, ((slope * 26 + offset) >> 4) + 1), 126)

        if init_state >= 64:
            self.state = init_state - 64
            self.mps = 1
        else:
            self.state = 63 - init_state
            self.mps = 0


class CabacEncoder:
    """CABAC arithmetic encoder."""

    __slots__ = (
        "_low",
        "_range",
        "_bits_left",
        "_buffer",
        "_num_buffered_bytes",
        "_buffered_byte",
    )

    def __init__(self):
        self._low = 0
        self._range = 510
        self._bits_left = 23
        self._buffer: List[int] = []
        self._num_buffered_bytes = 0
        self._buffered_byte = 0xFF

    def encode_bin(self, bin_val: int, ctx: CabacContext) -> None:
        """Encode a single bin using context."""
        q_range_idx = (self._range >> 6) & 3
        lps_range = CABAC_LPS_RANGE[ctx.state][q_range_idx]

        self._range -= lps_range

        if bin_val != ctx.mps:
            self._low += self._range
            self._range = lps_range

            if ctx.state == 0:
                ctx.mps = 1 - ctx.mps
            ctx.state = CABAC_TRANS_LPS[ctx.state]
        else:
            ctx.state = CABAC_TRANS_MPS[ctx.state]

        self._renormalize()

    def encode_bypass(self, bin_val: int) -> None:
        """Encode a bin with uniform (0.5) probability."""
        self._low <<= 1
        if bin_val:
            self._low += self._range
        self._bits_left -= 1

        if self._bits_left < 12:
            self._write_out()

    def encode_bypass_bins(self, bins: int, num_bins: int) -> None:
        """Encode multiple bypass bins."""
        for i in range(num_bins - 1, -1, -1):
            self.encode_bypass((bins >> i) & 1)

    def encode_terminate(self, bin_val: int) -> None:
        """Encode terminating bin (end of slice)."""
        self._range -= 2

        if bin_val:
            self._low += self._range
            self._range = 2
            self._renormalize()
            self._encode_final()
        else:
            self._renormalize()

    def _renormalize(self) -> None:
        """Renormalize after encoding."""
        while self._range < 256:
            self._range <<= 1
            self._low <<= 1
            self._bits_left -= 1

            if self._bits_left < 12:
                self._write_out()

    def _write_out(self) -> None:
        """Write buffered bytes."""
        lead_byte = self._low >> (24 - self._bits_left)
        self._bits_left += 8
        self._low &= 0xFFFFFFFF >> self._bits_left

        if lead_byte == 0xFF:
            self._num_buffered_bytes += 1
        else:
            if self._num_buffered_bytes > 0:
                carry = lead_byte >> 8
                byte_val = self._buffered_byte + carry
                self._buffer.append(byte_val)

                while self._num_buffered_bytes > 1:
                    byte_val = (0xFF + carry) & 0xFF
                    self._buffer.append(byte_val)
                    self._num_buffered_bytes -= 1
                self._num_buffered_bytes = 0

            self._buffered_byte = lead_byte & 0xFF

    def _encode_final(self) -> None:
        """Finalize encoding."""
        if self._low >> (32 - self._bits_left):
            self._buffer.append(self._buffered_byte + 1)
            while self._num_buffered_bytes > 1:
                self._buffer.append(0x00)
                self._num_buffered_bytes -= 1
        else:
            if self._num_buffered_bytes > 0:
                self._buffer.append(self._buffered_byte)
            while self._num_buffered_bytes > 1:
                self._buffer.append(0xFF)
                self._num_buffered_bytes -= 1

        bits_left = 24 - self._bits_left
        self._buffer.append((self._low >> 8) & 0xFF)
        if bits_left > 8:
            self._buffer.append(self._low & 0xFF)

    def get_bytes(self) -> bytes:
        """Get encoded bytes."""
        return bytes(self._buffer)

    def reset(self) -> None:
        """Reset encoder state."""
        self._low = 0
        self._range = 510
        self._bits_left = 23
        self._buffer = []
        self._num_buffered_bytes = 0
        self._buffered_byte = 0xFF


def init_contexts_for_slice(slice_type: int, qp: int) -> dict:
    """Initialize all CABAC contexts for a slice."""
    def contexts_with_init(init_value: int, count: int) -> list[CabacContext]:
        return [CabacContext(init_value) for _ in range(count)]

    contexts = {}

    contexts["split_cu_flag"] = [
        CabacContext(139),
        CabacContext(141),
        CabacContext(157),
    ]
    contexts["skip_flag"] = [CabacContext(197), CabacContext(185)]
    contexts["merge_flag"] = [CabacContext(110)]
    contexts["merge_idx"] = [CabacContext(122)]
    contexts["pred_mode"] = [CabacContext(149)]
    contexts["part_mode"] = [
        CabacContext(154),
        CabacContext(139),
        CabacContext(154),
        CabacContext(154),
    ]
    contexts["intra_luma_pred_mode"] = [CabacContext(183)]
    contexts["intra_chroma_pred_mode"] = [CabacContext(152), CabacContext(139)]
    contexts["cbf_luma"] = [CabacContext(111), CabacContext(141)]
    contexts["cbf_chroma"] = [
        CabacContext(94),
        CabacContext(138),
        CabacContext(182),
        CabacContext(154),
    ]
    contexts["transform_skip"] = [CabacContext(139), CabacContext(139)]
    contexts["last_sig_coeff_x_prefix"] = contexts_with_init(125, 18)
    contexts["last_sig_coeff_y_prefix"] = contexts_with_init(125, 18)
    contexts["coded_sub_block_flag"] = [
        CabacContext(121),
        CabacContext(140),
        CabacContext(61),
        CabacContext(154),
    ]
    contexts["sig_coeff_flag"] = contexts_with_init(170, 42)
    contexts["coeff_abs_level_greater1"] = contexts_with_init(154, 24)
    contexts["coeff_abs_level_greater2"] = contexts_with_init(154, 6)

    return contexts


def encode_coeff_abs_level_remaining(
    encoder: CabacEncoder, level: int, rice_param: int
) -> None:
    """Encode coefficient absolute level minus threshold using bypass bins."""
    threshold = 3 << rice_param

    if level < threshold:
        length = level >> rice_param
        for _ in range(length):
            encoder.encode_bypass(1)
        encoder.encode_bypass(0)
        encoder.encode_bypass_bins(level & ((1 << rice_param) - 1), rice_param)
    else:
        length = threshold >> rice_param
        for _ in range(length):
            encoder.encode_bypass(1)
        encoder.encode_bypass(0)

        remaining = level - threshold
        exp_golomb_order = rice_param + 1

        while remaining >= (1 << exp_golomb_order):
            remaining -= 1 << exp_golomb_order
            exp_golomb_order += 1
            encoder.encode_bypass(1)

        encoder.encode_bypass(0)
        encoder.encode_bypass_bins(remaining, exp_golomb_order)
