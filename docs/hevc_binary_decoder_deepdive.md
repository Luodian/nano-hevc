# HEVC C Binary Decoder Deep-Dive: OneVision-Encoder 的 Codec 数据提取协议

本文档是对 OneVision-Encoder 中 HEVC C 二进制解码器（`dataloader/decoder/bin/hevc`）输出协议的完整逆向工程记录。覆盖每个字段的字节级布局、HEVC 编码语义、当前使用状态，以及未使用字段的研究潜力分析。

---

## 1. C Binary 的身份与架构

### 1.1 它是什么

`dataloader/decoder/bin/hevc` 是一个基于 **openHEVC** 的预编译二进制文件（Linux/ARM 架构）。openHEVC 本身是 FFmpeg libavcodec HEVC 解码器的独立分支，专门优化为可嵌入式使用。

它的角色很单一：**接受标准 HEVC 比特流输入，在 stdout 上输出解码后的像素数据和编码器决策元数据的二进制流。**

### 1.2 源文件与头文件

repo 中只保留了头文件和预编译的二进制，没有 C 源码：

| 文件 | 内容 |
|------|------|
| `dataloader/decoder/bin/hevc` | 预编译二进制（Linux/ARM，macOS 上无法直接运行） |
| `dataloader/decoder/include/openHevcWrapper.h` | C API 头文件，定义 `OpenHevc_Frame_cpy` 结构体 |
| `dataloader/decoder/include/hevcpred.h` | HEVC 预测上下文（intra_pred 函数指针表） |
| `dataloader/decoder/include/hevcdsp.h` | HEVC DSP 上下文（DCT/DST、SAO、Loop Filter、MC 函数指针表） |

### 1.3 关键 C 结构体

`openHevcWrapper.h` 第 71-84 行定义了输出帧的 buffer 布局：

```c
typedef struct OpenHevc_Frame_cpy {
    void *pvY;    // Y 平面（重建像素）
    void *pvU;    // U 平面
    void *pvV;    // V 平面
    void *pvMV;   // 运动向量 + 参考偏移 + CU 大小 + padding（打包在一个 buffer 中）
    void *pvYR;   // Y 残差平面
    void *pvUR;   // U 残差
    void *pvVR;   // V 残差
} OpenHevc_Frame_cpy;
```

注意：`pvMV` 不仅仅包含运动向量 -- 它是一个打包 buffer，依次包含 MV L0、MV L1、参考偏移、CU 大小和对齐 padding。

---

## 2. 二进制输出协议：逐字节解析

C binary 在 stdout 上按帧输出二进制数据。Python 端的 `HevcFeatureReader._read_frame_data()` 方法（`hevc_feature_decoder_mv.py` 第 572-629 行）精确地解析了这个协议。

### 2.1 每帧总大小公式

对于分辨率为 W×H 的视频：

```
total_bytes_per_frame = YUV420(recon) + pvMV_block + META + YUV420(residual)

其中：
  YUV420       = H×W + 2×(H/2)×(W/2) = H×W×1.5
  pvMV_block   = 3×H×W/4              （固定大小，内含 MV + ref_off + size + padding）
  META         = H×W/4
  YUV420(res)  = H×W×1.5

total = H×W×1.5 + 3×H×W/4 + H×W/4 + H×W×1.5
      = H×W × (1.5 + 0.75 + 0.25 + 1.5)
      = H×W × 4.0
```

**每帧的总输出大小恰好是 4×H×W 字节。**

### 2.2 完整的字段布局（按 stdout 顺序）

以下表格按照 C binary 实际输出的字节顺序排列。所有偏移量基于帧起始位置。

| # | 字段名 | 大小（字节） | 数据类型 | 维度 | 字节偏移（从帧头开始） |
|---|--------|------------|---------|------|----------------------|
| 1 | **Y（重建）** | H×W | uint8 | (H, W) | 0 |
| 2 | **U（重建）** | (H/2)×(W/2) | uint8 | (H/2, W/2) | H×W |
| 3 | **V（重建）** | (H/2)×(W/2) | uint8 | (H/2, W/2) | H×W + H×W/4 |
| 4 | **MVX_L0** | (H/4)×(W/4)×2 | int16 | (H/4, W/4) | H×W×1.5 |
| 5 | **MVY_L0** | (H/4)×(W/4)×2 | int16 | (H/4, W/4) | +pvMV_size |
| 6 | **MVX_L1** | (H/4)×(W/4)×2 | int16 | (H/4, W/4) | +pvMV_size×2 |
| 7 | **MVY_L1** | (H/4)×(W/4)×2 | int16 | (H/4, W/4) | +pvMV_size×3 |
| 8 | **REF_OFF_L0** | (H/4)×(W/4) | uint8 | (H/4, W/4) | +pvMV_size×4 |
| 9 | **REF_OFF_L1** | (H/4)×(W/4) | uint8 | (H/4, W/4) | +pvMV_size×4+pvOFF_size |
| 10 | **CU Size Map** | (H/8)×(W/8) | uint8 | (H/8, W/8) | +pvMV_size×4+pvOFF_size×2 |
| 11 | **Padding** | pvOffset 字节 | -- | -- | （对齐到 3×H×W/4） |
| 12 | **META** | H×W/4 | uint8 | flat | H×W×1.5 + 3×H×W/4 |
| 13 | **Y（残差）** | H×W | uint8 | (H, W) | +meta_bytes |
| 14 | **U（残差）** | (H/2)×(W/2) | uint8 | (H/2, W/2) | ... |
| 15 | **V（残差）** | (H/2)×(W/2) | uint8 | (H/2, W/2) | ... |

### 2.3 精确的大小计算

```python
pvY_size  = W * H                          # Y 平面字节数
pvU_size  = (W >> 1) * (H >> 1)            # U 平面字节数
pvV_size  = (W >> 1) * (H >> 1)            # V 平面字节数
pvMV_size = (W >> 2) * (H >> 2) * 2        # 单个 MV 分量（int16，故 ×2）
pvOFF_size = (W >> 2) * (H >> 2)           # 单个参考偏移（uint8）
pvSize_size = (W >> 3) * (H >> 3)          # CU 大小图（uint8）

# pvMV 块的固定总大小
pvMV_block = 3 * W * H // 4                # = 3×H×W/4（固定）

# padding 用于补齐 pvMV_block
pvOffset = pvMV_block - (pvMV_size * 5 + pvOFF_size * 2)

# 注意：pvMV_size×5 而非 ×4，是因为 CU Size Map 借用了 pvMV 的 buffer 空间
# 实际布局：MV_L0_x(pvMV) + MV_L0_y(pvMV) + MV_L1_x(pvMV) + MV_L1_y(pvMV) 
#          + REF_OFF_L0(pvOFF) + REF_OFF_L1(pvOFF)
#          + CU_SIZE(pvSize) + PADDING(pvOffset)
# 但 CU Size Map 的读取是从 pvMV buffer 的第 5 个 pvMV_size 偏移开始的

meta_bytes = pvY_size >> 2                  # = H×W/4
```

### 2.4 具体数值示例（1920×1080 视频）

```
W = 1920, H = 1080

pvY_size   = 2,073,600 bytes (1920 × 1080)
pvU_size   =   518,400 bytes (960 × 540)
pvV_size   =   518,400 bytes (960 × 540)
pvMV_size  =   259,200 bytes (480 × 270 × 2)
pvOFF_size =   129,600 bytes (480 × 270)
pvSize_size =   32,400 bytes (240 × 135)
pvMV_block = 1,555,200 bytes (3 × 1920 × 1080 / 4)
pvOffset   = 1,555,200 - (259,200×5 + 129,600×2) = 1,555,200 - 1,555,200 = 0
meta_bytes =   518,400 bytes (2,073,600 / 4)

total_per_frame = 2,073,600 + 518,400 + 518,400 + 1,555,200 + 518,400 
                + 2,073,600 + 518,400 + 518,400
                = 8,294,400 bytes
                = 4 × 1920 × 1080 ✓
```

对于 1920×1080，padding 恰好是 0。这不是巧合 -- pvMV_block 的大小被设计为恰好容纳所有子字段。

---

## 3. 每个字段的 HEVC 编码语义

### 3.1 YUV420 重建像素（字段 #1-3）

**含义**：经过完整 HEVC 解码流水线后的输出帧。这是 HEVC 解码器最终产出的视觉像素。

**编码细节**：
- Y 是亮度平面，U/V 是色度平面
- YUV420p 格式：色度分辨率为亮度的一半（水平和垂直各减半）
- 每个像素 8 bit（uint8），值域 0-255
- 这些像素已经过 HEVC 的去块滤波（Deblocking Filter）和 SAO（Sample Adaptive Offset）后处理

**OneVision-Encoder 中的使用**：
- `UMT_HEVC_Y_ONLY=1` 时（默认），只使用 Y 平面做灰度输入
- `UMT_HEVC_Y_ONLY=0` 时，Y/U/V 转换为 BGR 输入
- 用途：作为 ViT 的 patch embedding 输入像素

### 3.2 MV_L0（运动向量，参考列表 0）（字段 #4-5）

**含义**：HEVC 解码器在参考列表 0（List 0，通常是前向参考）中，每个 4×4 块找到的最佳匹配位移。

**编码细节**：
- 分辨率：H/4 × W/4（每个值对应一个 4×4 像素块）
- 数据类型：int16（有符号 16 位整数）
- **单位：四分之一像素（quarter-pel）**
  - 值 4 = 位移 1 个整像素
  - 值 1 = 位移 0.25 个像素（亚像素精度）
  - 值 -8 = 向左/上位移 2 个整像素
- MVX = 水平位移，MVY = 垂直位移
- 对于 I 帧，所有 MV 值为 0

**HEVC 中的 List 0**：
- P 帧（单向预测）：List 0 是唯一的参考列表，通常指向前面的帧
- B 帧（双向预测）：List 0 通常指向时间上较早的参考帧（但不总是，HEVC 允许灵活的参考列表构造）

**OneVision-Encoder 中的使用**：✅ **核心特征**
- `_mv_energy_norm()` 计算 `sqrt(mvx² + mvy²)` 得到每个 4×4 块的运动能量
- 减去全局运动模型（相似变换或仿射变换）后的残余运动作为局部显著性
- 最终映射到 patch 级别（通常 14×14 或 16×16 像素），取 patch 内最大/均值作为 patch 运动得分

### 3.3 MV_L1（运动向量，参考列表 1）（字段 #6-7）

**含义**：HEVC 解码器在参考列表 1（List 1，通常是后向参考）中的最佳匹配位移。

**编码细节**：
- 格式与 L0 完全一致：H/4 × W/4，int16，quarter-pel 单位
- 仅在 B 帧中有意义值（P 帧和 I 帧中为 0）
- B 帧双向预测时，最终预测像素 = `(L0_pred + L1_pred + 1) >> 1`

**HEVC 中的 List 1**：
- B 帧专用，通常指向时间上较晚的参考帧
- 但 HEVC 支持 Low-Delay B 模式（LDB），此时 List 1 也可以指向前面的帧
- List 1 MV 与 List 0 MV 的不一致性是遮挡（occlusion）的强信号

**OneVision-Encoder 中的使用**：⚠️ **提取但未使用**
- C binary 正确输出了 L1 MV
- Python reader 正确读取并存入 `frame_tuple[5]`（mvx_L1）和 `frame_tuple[6]`（mvy_L1）
- 但 `_mv_energy_norm()` 只使用 L0：`mvx, mvy = frame_tuple[3], frame_tuple[4]`
- L1 完全被忽略

### 3.4 REF_OFF_L0 / REF_OFF_L1（参考帧偏移）（字段 #8-9）

**含义**：每个 4×4 块实际参考的是 DPB（Decoded Picture Buffer）中的哪一帧。

**编码细节**：
- 分辨率：H/4 × W/4（与 MV 一致）
- 数据类型：uint8
- 值表示参考帧在 DPB 中的索引偏移（不是帧号差距）
  - 0 = 最近的参考帧（List 0 的第一个条目）
  - 1 = 次近的参考帧
  - 以此类推
- 对于 I 帧，值无意义

**HEVC 中的参考帧管理**：
- HEVC 维护一个 DPB，包含已解码的参考帧
- 编码器通过 RPS（Reference Picture Set）机制精确控制哪些帧保留在 DPB 中
- ref_off 越大，说明编码器需要参考更远的帧才能找到好的匹配 -- 这通常意味着场景变化或快速运动

**OneVision-Encoder 中的使用**：⚠️ **提取但未使用**
- 存入 `frame_tuple[7]`（ref_off_L0）和 `frame_tuple[8]`（ref_off_L1）
- 在整个 pipeline 中没有任何代码读取这些值

### 3.5 CU Size Map（编码单元大小图）（字段 #10）

**含义**：HEVC 编码器对每个 8×8 区域选择的编码单元（CU）大小。

**编码细节**：
- 分辨率：H/8 × W/8（每个值对应一个 8×8 像素区域）
- 数据类型：uint8
- 值表示该区域所属 CU 的边长（或其 log2 编码）
  - HEVC CU 大小层级：8×8, 16×16, 32×32, 64×64
  - 值可能是：8, 16, 32, 64（直接边长）或 3, 4, 5, 6（log2）

**HEVC 中的 CU 划分**：
- HEVC 编码的核心决策之一就是确定 CU 划分
- 编码器从 64×64 CTU 开始，使用率失真优化（RDO）决定是否递归细分
- **小 CU 表示该区域复杂度高**：编码器需要更细的粒度才能有效预测
- **大 CU 表示该区域简单**：平坦、低纹理、或运动一致的区域

**这是 HEVC 编码器最昂贵的决策之一** -- 编码器在每个 CTU 上可能花费 80% 的编码时间来做这个划分决策。

**OneVision-Encoder 中的使用**：⚠️ **提取但未使用**
- 存入 `frame_tuple[9]`
- 没有代码使用它

### 3.6 META Buffer（元数据块）（字段 #12）

**含义**：包含帧级元数据和 CTU 级四叉树结构信息的打包 buffer。

**编码细节**：
- 总大小：H×W/4 字节
- 内部布局：

```
偏移      内容                        大小
──────────────────────────────────────────────
[0]       magic/version = 4           1 byte
[1]       magic/version = 2           1 byte
[2]       frame_type                  1 byte
            0 = IDR (瞬时解码刷新)
            1 = CRA (清洁随机访问, IRAP)
            2 = P (单向预测)
            3 = B (双向预测)
[3..1023] 保留 / 未使用               1021 bytes
[1024..1024+nb_ctus*12)
          quadtree_stru               12 bytes per CTU
[1024+nb_ctus*12..end)
          未使用填充                   剩余字节
```

其中 `nb_ctus = ceil(W/64) * ceil(H/64)`。

**frame_type 的 HEVC 含义**：
- **IDR (0)**：立即解码刷新帧。完全清空 DPB，后续帧不能参考 IDR 之前的任何帧。这是最强的随机访问点。
- **CRA (1)**：清洁随机访问帧。也是一个随机访问点，但允许 Leading Pictures（在显示顺序上位于 CRA 之前但解码顺序在之后的帧）参考 CRA 之前的帧。HEVC 新增的概念。
- **P (2)**：单向预测帧。只使用 List 0（通常是前向参考）。
- **B (3)**：双向预测帧。使用 List 0 和 List 1。

### 3.7 Quadtree Structure（四叉树结构）（META 内嵌字段）

**含义**：每个 CTU 的递归四叉树分割决策的编码表示。

**编码细节**：
- 每个 CTU 12 字节
- 从 META buffer 的偏移 1024 开始
- 12 字节编码了 CTU → CU 的完整四叉树拓扑

**HEVC 四叉树的含义**：

一个 64×64 的 CTU 可以被递归四叉分割，最多 4 层深度：

```
深度 0: 64×64（1 个 CU）
         │
    ┌────┼────┬────┐   split_flag[0] = 1
    │    │    │    │
深度 1: 32×32（4 个 CU）
         │
    ┌────┼────┬────┐   split_flag[1][0] = 1（第一个 32×32 继续分割）
    │    │    │    │
深度 2: 16×16（4 个 CU）
         │
    ┌────┼────┬────┐   split_flag[2][0] = 1
    │    │    │    │
深度 3: 8×8（4 个 CU，最小）
```

每一层的 split flag 决定该 CU 是否继续细分为 4 个子 CU。12 字节需要编码：
- 深度 0：1 bit（split or not）
- 深度 1：最多 4 bit
- 深度 2：最多 16 bit
- 深度 3：最多 64 bit（8×8 是最小 CU，不再分割）
- 总计最多 85 个节点的 split 决策

12 字节 = 96 bit，足够编码完整的四叉树拓扑（85 个 split flag + 部分附加信息）。

**OneVision-Encoder 中的使用**：⚠️ **提取但未使用**
- 存入 `frame_tuple[1]`
- 没有代码使用它

### 3.8 YUV420 残差（字段 #13-15）

**含义**：每个像素的预测残差，即 `original - prediction` 的结果。

**编码细节**：
- 格式与重建像素完全一致：Y(H,W) + U(H/2,W/2) + V(H/2,W/2)
- 数据类型：uint8
- **重要**：残差被 centered at 128
  - 值 128 = 残差为 0（预测完全正确）
  - 值 > 128 = 正残差（原始 > 预测）
  - 值 < 128 = 负残差（原始 < 预测）
  - 实际残差 = pixel_value - 128
- 这种编码方式避免了有符号数据类型，同时保留了残差信息

**HEVC 中的残差**：
- 残差是编码效率的核心指标
- 高残差区域 = 预测效果差 = 编码器需要更多 bit 来表示该区域
- 残差大的区域通常是：
  - 新出现的物体（运动补偿无法预测）
  - 遮挡边界（物体遮挡关系变化）
  - 复杂纹理变化
  - 场景切换

**OneVision-Encoder 中的使用**：✅ **核心特征**
- `_residual_energy_norm()` 计算 `|pixel - 128|` 的能量
- 映射到 patch 级别后作为 patch 残差得分
- 与 MV 能量融合：`energy = alpha * mv_energy + (1-alpha) * residual_energy`

---

## 4. frame_tuple 与 meta dict：Python 侧的数据结构

### 4.1 frame_tuple（11 元素）

`HevcFeatureReader._readFrame()` 返回的帧数据元组：

```python
frame_tuple = (
    frame_type,      # [0] int: 0=IDR, 1=CRA, 2=P, 3=B
    quadtree_stru,   # [1] np.ndarray(uint8): 12 bytes × nb_ctus
    rgb,             # [2] np.ndarray(uint8): 重建帧（Y-only 或 BGR）
    mv_x_L0,         # [3] np.ndarray(int16, shape=(H/4, W/4)): 水平 MV L0
    mv_y_L0,         # [4] np.ndarray(int16, shape=(H/4, W/4)): 垂直 MV L0
    mv_x_L1,         # [5] np.ndarray(int16, shape=(H/4, W/4)): 水平 MV L1
    mv_y_L1,         # [6] np.ndarray(int16, shape=(H/4, W/4)): 垂直 MV L1
    ref_off_L0,      # [7] np.ndarray(uint8, shape=(H/4, W/4)): 参考帧偏移 L0
    ref_off_L1,      # [8] np.ndarray(uint8, shape=(H/4, W/4)): 参考帧偏移 L1
    size,            # [9] np.ndarray(uint8, shape=(H/8, W/8)): CU 大小图
    residual,        # [10] np.ndarray(uint8): 残差帧（Y-only 或 BGR）
)
```

### 4.2 meta dict

`HevcFeatureReader.nextFrameEx()` 返回的帧级元数据：

```python
meta = {
    "frame_index": int,          # 解码顺序中的帧序号
    "gop_id": int,               # 当前 GOP 编号
    "is_i_frame": bool,          # 是否为 I 帧（IDR 或 CRA）
    "frame_type": int,           # 0=IDR, 1=CRA, 2=P, 3=B
    "frame_type_str": str,       # "IDR", "CRA", "P", "B"
    "timestamp": float | None,   # 帧时间戳（如果可用）
    "i_cache_key": str | None,   # I 帧缓存键
    "width": int,                # 帧宽度
    "height": int,               # 帧高度
    "gop_pos": [int, int],       # [gop_id, pos_in_gop]
    "frame_hash": str | None,    # 帧哈希值
    "i_rgb_suppressed": bool,    # I 帧 RGB 是否被置零（节省内存）
}
```

### 4.3 环境变量控制

| 环境变量 | 默认值 | 作用 |
|---------|-------|------|
| `UMT_HEVC_Y_ONLY` | "1" | 只提取 Y 通道（灰度）vs 完整 BGR |
| `UMT_HEVC_SUPPRESS_I_RGB` | "0" | 置零 I 帧 RGB 以节省内存 |
| `UMT_HEVC_STRICT_I` | "0" | 仅用 frame_type 判断 I 帧（不用启发式后备） |
| `UMT_HEVC_I_TYPES` | "0" | 逗号分隔的 frame_type 值，视为 I 帧 |
| `UMT_HEVC_DEBUG` | "0" | 启用调试日志 |
| `HEVC_FEAT_DECODER` | `"../dataloader/decoder/bin/hevc"` | C binary 路径 |
| `HEVC_PREFIX_FAST` | "1" | 启用顺序读取快速路径 |

---

## 5. 使用状态总览

```
             已提取的 HEVC 字段

    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │   ✅ 活跃使用                ⚠️  提取但未使用     │
    │   ─────────                ───────────────      │
    │   • Y 重建像素              • quadtree_stru     │
    │   • MV_L0 (x, y)          • MV_L1 (x, y)      │
    │   • 残差 Y                  • REF_OFF_L0        │
    │   • frame_type             • REF_OFF_L1        │
    │   • GOP 结构               • CU Size Map       │
    │                            • timestamp          │
    │                                                 │
    └─────────────────────────────────────────────────┘
```

**5 个字段活跃使用，6 个字段提取但完全未使用。**

---

## 6. 未使用字段的研究价值分析

### 6.1 Quadtree Structure — 编码器的"注意力图"

**当前状态**：12 bytes/CTU 被读取后直接丢弃。

**它本质上是什么**：
这是 HEVC 编码器经过完整 RDO（率失真优化）后做出的空间复杂度判断。编码器在每个 CTU 上尝试所有可能的分割方案，选择 R-D cost 最优的那个。这个过程在 x265 中占据了 60-80% 的编码时间。

换句话说，**quadtree 是编码器花费最大计算量得出的"注意力图"**。

**可以做什么**：

1. **替代 CU Size Map 的更精细版本**
   - CU Size Map 只告诉你每个 8×8 区域的 CU 大小
   - Quadtree 告诉你完整的分割拓扑 -- 不仅是叶节点大小，还有分割路径
   - 例如：一个 32×32 区域被分为 4 个 16×16，其中 3 个保持 16×16 但 1 个被进一步分为 4 个 8×8 -- 这种不对称分割模式包含丰富的语义信息

2. **构建层级化 patch 重要性权重**
   - 深度 0 区域（64×64 不分割）→ 权重最低，极度均匀
   - 深度 3 区域（分到 8×8）→ 权重最高，内容复杂
   - 中间深度 → 线性或非线性插值
   - 这个权重可以直接替代或增强当前的 MV+残差能量融合

3. **跨帧一致性信号**
   - 同一空间位置的 quadtree 分割在相邻帧之间通常高度一致
   - 突变意味着场景变化、遮挡、或新物体出现
   - 这比 MV 的突变更可靠，因为 MV 可能因为搜索范围限制而不连续

4. **研究价值：完全未被探索的领域**
   - CoViAR、CoPE-VideoLM、C2MAE 等现有工作都没有使用四叉树结构
   - 这是一个全新的研究信号

**实现难度**：中等。需要写一个 12-byte → quadtree 的解析器，然后将叶节点深度映射为 H/8×W/8 的权重图。

### 6.2 MV L1 — B 帧的后向运动

**当前状态**：读取后存入 frame_tuple[5:7]，无代码使用。

**它本质上是什么**：
B 帧双向预测中指向后方参考帧的运动向量。与 L0（前向）配合，给出完整的时间运动场。

**可以做什么**：

1. **更鲁棒的运动能量**
   - 当前：`energy = sqrt(mvx_L0² + mvy_L0²)`
   - 改进：`energy = max(|MV_L0|, |MV_L1|)` 或 `energy = (|MV_L0| + |MV_L1|) / 2`
   - B 帧中，某些区域可能 L0 MV 很小但 L1 MV 很大（物体从后方进入画面）

2. **遮挡检测**
   - L0 和 L1 指向不同方向时，如果 `L0_pred ≠ L1_pred`，说明该区域在两个参考帧中看到的内容不同
   - 这是遮挡的直接信号
   - 遮挡区域的 patch 应该得到更高的权重（内容不确定，需要更多 token 来表示）

3. **运动一致性校验**
   - 对于无遮挡的匀速运动区域，L0 和 L1 的 MV 应该大致相反且等长
   - `|MV_L0 + MV_L1| ≈ 0` 表示运动一致
   - `|MV_L0 + MV_L1| >> 0` 表示该区域有复杂运动或遮挡

**实现难度**：极低。只需在 `_mv_energy_norm()` 中多读两个字段。

### 6.3 Reference Offsets — 时间距离信号

**当前状态**：读取后存入 frame_tuple[7:9]，无代码使用。

**它本质上是什么**：
每个 4×4 块的运动向量指向的参考帧在 DPB 中的位置。值越大，说明编码器需要看更远的帧才能找到匹配。

**可以做什么**：

1. **MV 归一化**
   - 当前 MV 能量不考虑时间距离：指向 1 帧前的 MV=4 和指向 5 帧前的 MV=4 被等同对待
   - 但实际上后者表示更慢的运动（像素每帧只移动 4/5 = 0.8 quarter-pel）
   - 归一化：`normalized_mv = |MV| / ref_offset` 得到每帧运动速度

2. **长距离参考异常检测**
   - 如果大部分块参考帧 0（最近），但某些块参考帧 3 或更远
   - 说明这些区域的近期帧不包含好的匹配 -- 可能是新出现的物体、遮挡恢复、或重复纹理
   - 这些"异常"区域值得更多关注

3. **时间连贯性加权**
   - 参考帧距离可以作为时间连贯性的代理指标
   - 在视频理解任务中，不同帧的 token 重要性应该考虑时间距离

**实现难度**：极低。

### 6.4 CU Size Map — 编码器复杂度判断的直接读出

**当前状态**：读取后存入 frame_tuple[9]，无代码使用。

**它本质上是什么**：
编码器 RDO 决策的直接输出。每个 8×8 区域所属 CU 的大小。

**可以做什么**：

1. **最简单的编码器先验**
   - 直接计算每个 patch 的"小 CU 比例"：`small_cu_ratio = count(size <= 16) / total_blocks_in_patch`
   - 高比例 → patch 内容复杂 → 需要更多 token
   - 低比例 → patch 内容简单 → 可以跳过

2. **与 MV/残差能量的互补**
   - MV 和残差是帧间信号（temporal）
   - CU 大小是帧内+帧间的综合决策（spatial + temporal）
   - 对于 I 帧，MV=0、残差无意义，但 CU 大小仍然有效

3. **I 帧的唯一可用先验**
   - 当前 I 帧处理完全依赖静态启发式（中心先验 + 均匀采样）
   - CU 大小图可以给 I 帧提供基于编码器决策的 patch 选择先验
   - 这可能是提升 I 帧质量最简单的方法

**实现难度**：极低。数据已经在 frame_tuple[9] 中，只需几行代码加入能量融合。

---

## 7. 从当前状态到"Fully HEVC"的路线图

### Stage 2.5: 启用已提取但未使用的数据（工作量：小时级）

零新增提取，纯 Python 端改动：

| 任务 | 改动位置 | 预期收益 |
|------|---------|---------|
| `_mv_energy_norm()` 加入 L1 | `ap_dataloader_dali_codec.py:573` | B 帧运动能量更准确 |
| 能量融合加入 CU Size Map | `ap_dataloader_dali_codec.py:181` | I 帧获得编码器先验 |
| MV 归一化使用 ref_offset | `ap_dataloader_dali_codec.py:573` | 跨 GOP 运动能量可比 |
| L0/L1 一致性 → 遮挡检测 | 新增 ~30 行 | 遮挡区域权重提升 |

### Stage 3: 学习化门控（工作量：天级）

用已有的 6 个信号（MV_L0, MV_L1, residual, CU_size, ref_off, frame_type）训练一个轻量门控网络替代硬编码启发式：

```python
# 当前：硬编码融合
energy = alpha * mv_energy + (1-alpha) * residual_energy

# Stage 3：学习融合
gate_input = concat([mv_L0_energy, mv_L1_energy, residual_energy, 
                     cu_size_ratio, ref_offset_mean, frame_type_onehot])
gate_weight = MLP(gate_input)  # 输出 per-patch 重要性
```

### Stage 4: 深度 HEVC 语法提取（工作量：周级）

需要修改 C binary 源码（或找到 openHEVC 源码重新编译），新增提取：

| 信号 | HEVC 含义 | 研究价值 |
|------|----------|---------|
| **QP Map** | 每个 CU 的量化参数 | 编码器 bit 分配策略的直接读出 |
| **Intra Mode Map** | 每个 PU 的 35 种 intra 预测模式 | 空间方向性先验（边缘方向、纹理方向） |
| **PU Mode** | 2Nx2N / NxN / 非对称分割 | 运动复杂度的更细粒度信号 |
| **TU Partition** | 变换单元的划分 | 残差能量分布的精细结构 |
| **DCT 系数能量** | 变换域的频率分布 | 纹理复杂度 vs 边缘复杂度 |
| **Deblocking Strength** | 去块滤波强度 | 块边界不连续度 |
| **SAO Parameters** | 自适应偏移类型和值 | 像素级的编码器矫正决策 |
| **Weighted Prediction** | 加权预测参数 | 亮度渐变检测（渐入渐出） |

### Stage 5: 完全 Codec-Native Vision（工作量：月级）

将 ViT 的 patch embedding 完全替换为 codec feature embedding，直接在压缩域操作，跳过像素重建：

```
当前: bitstream → [C decoder] → pixels → patch_embed → ViT
目标: bitstream → [C decoder] → codec_features → codec_embed → ViT
```

这将是真正的"Fully HEVC"视觉编码器。

---

## 8. 附录：ASCII 格式的二进制布局图

```
┌──────────────────────────────────────────────────────────────────┐
│                    C Binary stdout (per frame)                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────┐                            │
│  │     YUV420 Reconstructed        │  H×W×1.5 bytes             │
│  │  ┌────────────┐                 │                            │
│  │  │ Y  (H×W)   │  uint8          │                            │
│  │  ├────────────┤                 │                            │
│  │  │ U  (H/2×   │  uint8          │                            │
│  │  │    W/2)    │                 │                            │
│  │  ├────────────┤                 │                            │
│  │  │ V  (H/2×   │  uint8          │                            │
│  │  │    W/2)    │                 │                            │
│  │  └────────────┘                 │                            │
│  └─────────────────────────────────┘                            │
│                                                                  │
│  ┌─────────────────────────────────┐                            │
│  │     pvMV Block                  │  3×H×W/4 bytes (fixed)     │
│  │  ┌────────────┐                 │                            │
│  │  │ MVX_L0     │  int16 (H/4×W/4)│                           │
│  │  ├────────────┤                 │                            │
│  │  │ MVY_L0     │  int16 (H/4×W/4)│                           │
│  │  ├────────────┤                 │                            │
│  │  │ MVX_L1     │  int16 (H/4×W/4)│                           │
│  │  ├────────────┤                 │                            │
│  │  │ MVY_L1     │  int16 (H/4×W/4)│                           │
│  │  ├────────────┤                 │                            │
│  │  │ REF_OFF_L0 │  uint8 (H/4×W/4)│                           │
│  │  ├────────────┤                 │                            │
│  │  │ REF_OFF_L1 │  uint8 (H/4×W/4)│                           │
│  │  ├────────────┤                 │                            │
│  │  │ CU_SIZE    │  uint8 (H/8×W/8)│                           │
│  │  ├────────────┤                 │                            │
│  │  │ PADDING    │  pvOffset bytes  │                           │
│  │  └────────────┘                 │                            │
│  └─────────────────────────────────┘                            │
│                                                                  │
│  ┌─────────────────────────────────┐                            │
│  │     META Buffer                 │  H×W/4 bytes               │
│  │  [0] = 4 (magic)               │                            │
│  │  [1] = 2 (magic)               │                            │
│  │  [2] = frame_type (0/1/2/3)    │                            │
│  │  [1024..] = quadtree_stru      │  12 bytes × nb_ctus        │
│  └─────────────────────────────────┘                            │
│                                                                  │
│  ┌─────────────────────────────────┐                            │
│  │     YUV420 Residual             │  H×W×1.5 bytes             │
│  │  (centered at 128)             │                            │
│  │  ┌────────────┐                 │                            │
│  │  │ Y_res      │  uint8          │                            │
│  │  ├────────────┤                 │                            │
│  │  │ U_res      │  uint8          │                            │
│  │  ├────────────┤                 │                            │
│  │  │ V_res      │  uint8          │                            │
│  │  └────────────┘                 │                            │
│  └─────────────────────────────────┘                            │
│                                                                  │
│  Total: 4 × H × W bytes per frame                              │
└──────────────────────────────────────────────────────────────────┘
```

---

*文档基于 `OneVision-Encoder/dataloader/hevc_feature_decoder_mv.py` 的完整逆向工程分析。C binary 基于 openHEVC（FFmpeg HEVC 解码器分支），预编译为 Linux/ARM 架构。*
