"""nano-hevc: A minimal, educational HEVC encoder in Python."""

__version__ = "0.1.0"

from nano_hevc.frame import Plane, Frame, PackedFrame, FrameBufferPool
from nano_hevc.block import BlockView, iterate_blocks
from nano_hevc.intra import (
    INTRA_PRED_ANGLE,
    intra_dc_predict_4x4,
    intra_dc_predict,
    intra_planar_predict,
    intra_angular_predict,
    residual_block,
    reconstruct_block,
    clip_to_pixel_range,
)
from nano_hevc.transform import (
    forward_transform,
    inverse_transform,
    forward_transform_4x4,
    inverse_transform_4x4,
    forward_transform_8x8,
    inverse_transform_8x8,
    forward_transform_16x16,
    inverse_transform_16x16,
    forward_transform_32x32,
    inverse_transform_32x32,
    DCT4,
    DCT8,
    DCT16,
    DCT32,
    DST4,
)
from nano_hevc.quant import (
    quantize,
    dequantize,
    quantize_block,
    dequantize_block,
    QUANT_SCALE,
    DEQUANT_SCALE,
)
from nano_hevc.metrics import (
    psnr,
    mse,
    sad,
    satd_4x4,
    residual_energy,
)
from nano_hevc.scan import (
    scan_block,
    inverse_scan_block,
    get_scan_order,
    get_scan_type_for_intra_mode,
    SCAN_DIAG,
    SCAN_HOR,
    SCAN_VER,
)
from nano_hevc.bitstream import BitstreamWriter
from nano_hevc.cabac import CabacEncoder, CabacContext, init_contexts_for_slice
from nano_hevc.nal import HEVCConfig, create_parameter_sets
__all__ = [
    "Plane",
    "Frame",
    "PackedFrame",
    "FrameBufferPool",
    "BlockView",
    "iterate_blocks",
    "INTRA_PRED_ANGLE",
    "intra_dc_predict_4x4",
    "intra_dc_predict",
    "intra_planar_predict",
    "intra_angular_predict",
    "residual_block",
    "reconstruct_block",
    "clip_to_pixel_range",
    "forward_transform",
    "inverse_transform",
    "forward_transform_4x4",
    "inverse_transform_4x4",
    "forward_transform_8x8",
    "inverse_transform_8x8",
    "forward_transform_16x16",
    "inverse_transform_16x16",
    "forward_transform_32x32",
    "inverse_transform_32x32",
    "DCT4",
    "DCT8",
    "DCT16",
    "DCT32",
    "DST4",
    "quantize",
    "dequantize",
    "quantize_block",
    "dequantize_block",
    "QUANT_SCALE",
    "DEQUANT_SCALE",
    "psnr",
    "mse",
    "sad",
    "satd_4x4",
    "residual_energy",
    "scan_block",
    "inverse_scan_block",
    "get_scan_order",
    "get_scan_type_for_intra_mode",
    "SCAN_DIAG",
    "SCAN_HOR",
    "SCAN_VER",
    "BitstreamWriter",
    "CabacEncoder",
    "CabacContext",
    "init_contexts_for_slice",
    "HEVCConfig",
    "create_parameter_sets",
    "encode_frame",
    "encode_video",
    "encode_video_nano",
    "encode_video_ffmpeg",
    "decode_video_nano",
]

_LAZY_ENCODER_EXPORTS = {
    "encode_frame",
    "encode_video",
    "encode_video_nano",
    "encode_video_ffmpeg",
    "decode_video_nano",
}


def __getattr__(name: str):
    if name in _LAZY_ENCODER_EXPORTS:
        from nano_hevc import encoder as _encoder

        return getattr(_encoder, name)
    raise AttributeError(f"module 'nano_hevc' has no attribute {name!r}")
