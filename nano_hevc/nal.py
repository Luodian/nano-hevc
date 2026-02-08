"""
NAL unit generation for HEVC bitstream.

Implements VPS, SPS, PPS, and slice header generation per ITU-T H.265.
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field

from nano_hevc.bitstream import BitstreamWriter, write_start_code_prefix


NAL_TRAIL_N = 0
NAL_TRAIL_R = 1
NAL_TSA_N = 2
NAL_TSA_R = 3
NAL_STSA_N = 4
NAL_STSA_R = 5
NAL_RADL_N = 6
NAL_RADL_R = 7
NAL_RASL_N = 8
NAL_RASL_R = 9
NAL_BLA_W_LP = 16
NAL_BLA_W_RADL = 17
NAL_BLA_N_LP = 18
NAL_IDR_W_RADL = 19
NAL_IDR_N_LP = 20
NAL_CRA_NUT = 21
NAL_VPS = 32
NAL_SPS = 33
NAL_PPS = 34
NAL_AUD = 35
NAL_EOS = 36
NAL_EOB = 37
NAL_FD = 38
NAL_PREFIX_SEI = 39
NAL_SUFFIX_SEI = 40


PROFILE_MAIN = 1
PROFILE_MAIN_10 = 2
PROFILE_MAIN_STILL = 3

LEVEL_1 = 30
LEVEL_2 = 60
LEVEL_2_1 = 63
LEVEL_3 = 90
LEVEL_3_1 = 93
LEVEL_4 = 120
LEVEL_4_1 = 123
LEVEL_5 = 150
LEVEL_5_1 = 153
LEVEL_5_2 = 156
LEVEL_6 = 180
LEVEL_6_1 = 183
LEVEL_6_2 = 186


@dataclass
class HEVCConfig:
    """HEVC encoder configuration."""

    width: int = 1920
    height: int = 1080
    fps: int = 30
    qp: int = 27
    bit_depth: int = 8
    profile: int = PROFILE_MAIN
    level: int = LEVEL_4
    max_cu_size: int = 64
    min_cu_size: int = 8
    max_tu_size: int = 32
    min_tu_size: int = 4
    max_transform_hierarchy_depth_inter: int = 2
    max_transform_hierarchy_depth_intra: int = 2
    amp_enabled: bool = False
    sao_enabled: bool = False
    pcm_enabled: bool = False
    strong_intra_smoothing: bool = False
    sign_data_hiding: bool = False
    transform_skip_enabled: bool = False


def write_nal_unit_header(
    writer: BitstreamWriter, nal_type: int, temporal_id: int = 0
) -> None:
    """Write 2-byte NAL unit header."""
    writer.write_bit(0)
    writer.write_bits(nal_type, 6)
    writer.write_bits(0, 6)
    writer.write_bits(temporal_id + 1, 3)


def write_profile_tier_level(
    writer: BitstreamWriter, config: HEVCConfig, max_sub_layers: int = 1
) -> None:
    """Write profile_tier_level syntax."""
    profile_idc = max(1, min(int(config.profile), 31))
    writer.write_bits(0, 2)  # general_profile_space
    writer.write_bit(0)  # general_tier_flag (Main tier)
    writer.write_bits(profile_idc, 5)  # general_profile_idc

    compatibility_flags = 0
    compatibility_flags |= 1 << (31 - profile_idc)
    writer.write_bits(compatibility_flags, 32)

    writer.write_bit(1)
    writer.write_bit(0)
    writer.write_bit(0)
    writer.write_bit(1)
    writer.write_bits(0, 44)

    writer.write_bits(config.level, 8)


def write_vps(writer: BitstreamWriter, config: HEVCConfig) -> None:
    """Write Video Parameter Set."""
    write_start_code_prefix(writer)
    write_nal_unit_header(writer, NAL_VPS)

    writer.write_bits(0, 4)
    writer.write_bits(3, 2)
    writer.write_bits(0, 6)
    writer.write_bits(0, 3)
    writer.write_bit(1)
    writer.write_bits(0xFFFF, 16)

    write_profile_tier_level(writer, config)

    writer.write_bit(1)  # vps_sub_layer_ordering_info_present_flag
    writer.write_ue(2)  # vps_max_dec_pic_buffering_minus1[0]
    writer.write_ue(0)  # vps_max_num_reorder_pics[0]
    writer.write_ue(1)  # vps_max_latency_increase_plus1[0]
    writer.write_bits(0, 6)  # vps_max_layer_id
    writer.write_ue(0)  # vps_num_layer_sets_minus1
    writer.write_bit(0)  # vps_timing_info_present_flag
    writer.write_bit(0)  # vps_extension_flag

    writer.write_rbsp_trailing_bits()


def write_sps(writer: BitstreamWriter, config: HEVCConfig) -> None:
    """Write Sequence Parameter Set."""
    write_start_code_prefix(writer)
    write_nal_unit_header(writer, NAL_SPS)

    writer.write_bits(0, 4)
    writer.write_bits(0, 3)
    writer.write_bit(1)

    write_profile_tier_level(writer, config)

    writer.write_ue(0)
    writer.write_ue(1)

    writer.write_ue(config.width)
    writer.write_ue(config.height)

    writer.write_bit(0)

    writer.write_ue(config.bit_depth - 8)
    writer.write_ue(config.bit_depth - 8)

    log2_max_poc = 8
    writer.write_ue(log2_max_poc - 4)

    writer.write_bit(1)
    writer.write_ue(0)
    writer.write_ue(0)
    writer.write_ue(0)

    log2_min_cb = 3
    log2_diff_max_min_cb = int(config.max_cu_size).bit_length() - 1 - log2_min_cb
    log2_min_tb = 2
    log2_diff_max_min_tb = int(config.max_tu_size).bit_length() - 1 - log2_min_tb

    writer.write_ue(log2_min_cb - 3)
    writer.write_ue(log2_diff_max_min_cb)
    writer.write_ue(log2_min_tb - 2)
    writer.write_ue(log2_diff_max_min_tb)
    writer.write_ue(config.max_transform_hierarchy_depth_inter)
    writer.write_ue(config.max_transform_hierarchy_depth_intra)

    writer.write_bit(0)
    writer.write_bit(config.amp_enabled)
    writer.write_bit(config.sao_enabled)
    writer.write_bit(config.pcm_enabled)

    writer.write_ue(0)

    writer.write_bit(0)
    writer.write_bit(0)
    writer.write_bit(0)
    writer.write_bit(0)

    writer.write_bit(0)

    writer.write_rbsp_trailing_bits()


def write_pps(writer: BitstreamWriter, config: HEVCConfig) -> None:
    """Write Picture Parameter Set."""
    write_start_code_prefix(writer)
    write_nal_unit_header(writer, NAL_PPS)

    writer.write_ue(0)
    writer.write_ue(0)

    writer.write_bit(0)
    writer.write_bit(0)
    writer.write_bits(0, 3)

    writer.write_bit(config.sign_data_hiding)
    writer.write_bit(0)

    writer.write_ue(0)
    writer.write_ue(0)
    writer.write_se(0)

    writer.write_bit(0)
    writer.write_bit(config.transform_skip_enabled)
    writer.write_bit(0)
    writer.write_se(0)
    writer.write_se(0)
    writer.write_bit(0)
    writer.write_bit(0)
    writer.write_bit(0)
    writer.write_bit(0)
    writer.write_bit(0)
    writer.write_bit(0)
    writer.write_bit(1)
    writer.write_bit(0)
    writer.write_bit(0)
    writer.write_bit(0)
    writer.write_ue(0)
    writer.write_bit(0)
    writer.write_bit(0)

    writer.write_rbsp_trailing_bits()


def write_slice_header(
    writer: BitstreamWriter,
    config: HEVCConfig,
    is_first_slice: bool = True,
    slice_type: int = 2,
    poc: int = 0,
) -> None:
    """
    Write slice header.

    slice_type: 0=B, 1=P, 2=I
    """
    writer.write_bit(is_first_slice)

    if is_first_slice:
        nal_type = NAL_IDR_N_LP if poc == 0 else NAL_TRAIL_R
        if nal_type >= NAL_BLA_W_LP and nal_type <= NAL_CRA_NUT:
            writer.write_bit(0)

    writer.write_ue(0)
    if not is_first_slice:
        writer.write_bit(0)  # dependent_slice_segment_flag

    writer.write_ue(slice_type)
    # num_extra_slice_header_bits from PPS is zero in this encoder.

    if slice_type != 2:
        log2_max_poc = 8
        writer.write_bits(poc & ((1 << log2_max_poc) - 1), log2_max_poc)
        writer.write_bit(1)

    if config.sao_enabled:
        writer.write_bit(0)
        writer.write_bit(0)

    writer.write_se(config.qp - 26)

    if config.sao_enabled:
        writer.write_bit(0)

    # pps_loop_filter_across_slices_enabled_flag is 1 in write_pps(), and
    # deblocking is implicitly enabled, so this flag is present.
    writer.write_bit(1)


def create_parameter_sets(config: HEVCConfig) -> bytes:
    """Create VPS, SPS, and PPS as a single byte sequence."""
    writer = BitstreamWriter()
    write_vps(writer, config)
    write_sps(writer, config)
    write_pps(writer, config)
    # Parameter sets are emitted as Annex-B NAL units with explicit start codes.
    # Applying emulation-prevention over the whole byte stream would corrupt those
    # start codes (e.g., 00 00 01 -> 00 00 03 01), so keep raw bytes here.
    return writer.get_bytes()


def create_slice_nal_unit(
    config: HEVCConfig,
    slice_data: bytes,
    is_first_slice: bool = True,
    slice_type: int = 2,
    poc: int = 0,
) -> bytes:
    """Create complete slice NAL unit with header and data."""

    def apply_emulation_prevention(rbsp: bytes) -> bytes:
        out = bytearray()
        zero_count = 0
        for byte in rbsp:
            if zero_count >= 2 and byte <= 0x03:
                out.append(0x03)
                zero_count = 0
            out.append(byte)
            if byte == 0x00:
                zero_count += 1
            else:
                zero_count = 0
        return bytes(out)

    writer = BitstreamWriter()
    write_start_code_prefix(writer)

    if poc == 0:
        write_nal_unit_header(writer, NAL_IDR_N_LP)
    else:
        write_nal_unit_header(writer, NAL_TRAIL_R)

    write_slice_header(writer, config, is_first_slice, slice_type, poc)
    # byte_alignment(): one '1' bit then zero bits to next byte boundary.
    if writer.bit_count % 8 != 0:
        writer.write_bit(1)
        while writer.bit_count % 8 != 0:
            writer.write_bit(0)

    header_bytes = writer.get_bytes()
    if len(header_bytes) < 6:
        return header_bytes + slice_data

    start_code = header_bytes[:4]
    nal_header = header_bytes[4:6]
    rbsp = header_bytes[6:] + slice_data
    escaped_rbsp = apply_emulation_prevention(rbsp)
    return start_code + nal_header + escaped_rbsp
