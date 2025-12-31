# 🎬 nano-hevc 第二期：数据的布局

上一篇文章我们聊了视频编码为什么选择 YUV 而不是 RGB。但理解「YUV 是什么」是一回事，如何在工程代码中合理地表示 YUV 数据结构又是另一回事。主包觉得这个话题值得单独水一期。

nano-hevc 这个项目的目标不仅仅是实现算法的 pseudocode 式模拟。主包更希望探索的是：在 Python 语言下，如何贴合 HEVC 标准的设计思想，合理利用语言特性，写出具有工业级代码质量的实现。

## 📦 数据的物理形态

在动手写代码之前，我们需要先想清楚一个问题：在 HEVC 视角下，一帧视频在内存里应该长什么样？

这个问题看起来很简单，但实际上隐藏着不少工程 tradeoff。不同的数据布局会直接影响后续算法的实现难度和运行效率。

### HWC 格式 vs Planar 格式

在聊视频编码的数据布局之前，有必要先理清两种常见的多通道数据存储方式。

以一张 RGB 图像为例，每个像素有 R、G、B 三个值。这三个值在内存中怎么排列？有两种截然不同的策略。

这里我们用一个 2×2 的小图来展示两种不同的存储方式：

![像素布局（逻辑视角）](./assets/pixel_layout.png)

HWC 格式下，内存中的实际排列是按「像素优先」的顺序，先存完一个像素的所有通道，再存下一个像素。

![HWC 内存布局](./assets/hwc_memory_layout.png)

HWC（又称 Packed 或 Interleaved）格式按「像素优先」存储：先存完一个像素的所有通道，再存下一个像素。这种布局的好处是什么？访问单个像素的全部通道时，数据在内存中紧挨着，cache 友好。图像渲染、显示输出等场景经常需要同时读取一个位置的所有通道，所以 OpenCV、PIL 等图像库默认使用这种格式。

Planar（又称 CHW）格式则完全不同，它按「通道优先」存储：先存完整个 R 平面，再存完整个 G 平面，最后存 B 平面：

![Planar 内存布局](./assets/planar_memory_layout.png)

这两种格式各有适用场景，选择取决于后续操作的访问模式。值得一提的是，现代深度学习框架如 PyTorch 默认使用 NCHW（即 Planar）格式，因为 GPU 上的矩阵运算对这种布局有更好的优化。

那视频编码为什么选择 Planar？

视频编码器 90% 的时间都在单独处理 Y 平面或 U/V 平面。做预测、变换、量化时，每次操作的都是单个通道的像素块。如果用 HWC 格式，每次读取 Y 平面的一个 8×8 块时，内存中实际是 Y0U0V0Y1U1V1... 这样交错的，需要「跳着读」，每读一个 Y 就要跳过 U 和 V，cache 命中率直接腰斩。

还有一个问题：在 4:2:0 采样下，U/V 平面的尺寸只有 Y 的 1/4（水平和垂直各减半）。如果用交错存储，每 4 个 Y 像素才对应 1 组 UV，数据结构会变得异常复杂。而 Planar 格式下，三个平面各自独立，尺寸不同也毫无问题。

更关键的是，HEVC 允许 luma 和 chroma 在变换单元（TU）层面拥有不同的划分深度（这一特性在 H.264 中并不存在）。比如同一个 CU 内，Luma 的 TU 可以细分到 4×4，而 Chroma 受限于最小变换单元大小，可能只划分到对应 Luma 8×8 区域的 4×4 块就不再细分了。如果 Y/U/V 混在一起存储，这种灵活性就难以实现。

所以从第一行代码开始，我们就需要把一帧图像拆解为三个正交的二维平面，各自独立、互不干扰。

### Plane：单个颜色平面

理解了为什么要分离 Y/U/V 之后，我们来看看 nano-hevc 中 `Plane` 的设计。

这个类的职责很简单：封装一个二维 NumPy 数组，表示单个颜色通道。注意这里用的是 `__slots__` 而非普通的 class 或 dataclass：

```python
class Plane:
    """一个颜色平面（Y、U 或 V）"""
    __slots__ = ('data',)  # 👈 省内存的关键

    def __init__(self, data: np.ndarray):
        self.data = data

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @classmethod
    def zeros(cls, height: int, width: int, dtype: np.dtype = np.int16) -> Plane:
        return cls(data=np.zeros((height, width), dtype=dtype, order='C'))
```

代码很短，但有几个值得注意的细节。

### 🤔 何为 `__slots__`？

这是本文的核心优化之一，展开聊聊。

Python 中每个对象默认都有一个 `__dict__` 属性来存储实例属性。这个设计非常灵活，你可以随时给对象添加新属性，甚至在运行时动态修改对象结构。但灵活性是有代价的。

`__dict__` 本质上是一个哈希表。即使只存一个属性，也要维护 bucket 数组、哈希种子、负载因子等元数据。在 CPython 3.11 中，一个空的 `__dict__` 就要占用约 48 字节，加上对象头开销，每个实例至少需要 100～200 字节的额外内存。

对于普通业务代码，这点开销完全可以忽略。但视频编码的场景确实不同，需要创建的对象数量非常庞大。

算一笔账：一帧 1080p 视频，用 8×8 的块来处理，Y 平面就有 $(1080/8) \times (1920/8) = 32400$ 个块。加上 U/V 平面（尺寸减半），总共约 40000 个块，每个块都要创建一个 `BlockView` 对象。如果每个对象需要 200 字节额外内存，40000 个对象就是 8MB。这还只是单帧的开销。

`__slots__` 的作用就是告诉 Python：「这个类的实例只会有这几个属性，不需要 `__dict__`」。Python 会把属性存储在固定大小的数组里，通过偏移量直接访问，就像 C 的 struct 一样。属性访问从查字典变成按偏移量访问，速度更快，内存占用大幅下降。

这里我们可以用 `BlockView` 举例

![Python 对象内存布局对比](./assets/slots_memory_comparison.png)

实际节省比例因 Python 版本和属性数量而异。属性越少，`__dict__` 的固定开销占比越大，节省比例越高。对于视频压缩这种需要创建几万个 `BlockView` 对象的场景，这个优化效果相当可观。

另一个细节是 `zeros` 工厂方法中的 `order='C'`。NumPy 数组有两种内存布局：C order（行优先）和 Fortran order（列优先）。视频编码通常按行扫描处理数据，所以 C order 能获得更好的 cache locality。这个参数容易被忽略，但在大规模数据处理中，cache 命中率的差异可能带来显著的性能差距。

顺便提一下，nano-hevc 的 `Plane` 实现假设图像数据是紧凑排列的（width × height）。但在真实的工业级编码器（如 x265、FFmpeg）中，为了利用 SIMD 指令集（AVX2/NEON）加速，每一行像素通常需要内存对齐（比如对齐到 32 或 64 字节）。这时候就需要引入 `stride`（跨度）的概念：`stride` 表示相邻两行之间的字节偏移，可能大于 `width`。

### Frame：三个平面的容器

有了 `Plane`，`Frame` 的实现就很自然了。它只是三个 `Plane` 的容器，同样用 `__slots__`：

```python
class Frame:
    """YUV420p 格式的视频帧"""
    __slots__ = ('y', 'u', 'v')

    def __init__(self, y: Plane, u: Plane, v: Plane):
        self.y = y
        self.u = u
        self.v = v

    @classmethod
    def zeros(cls, height: int, width: int, dtype: np.dtype = np.int16) -> Frame:
        return cls(
            y=Plane.zeros(height, width, dtype),
            u=Plane.zeros(height // 2, width // 2, dtype),  # UV 是 Y 的 1/4
            v=Plane.zeros(height // 2, width // 2, dtype),
        )

    @classmethod
    def from_yuv420p(cls, buffer: bytes, height: int, width: int) -> Frame:
        """从原始 YUV420p 字节流构建 Frame"""
        y_size = height * width
        uv_height, uv_width = height // 2, width // 2
        uv_size = uv_height * uv_width
        return cls(
            y=Plane.from_buffer(buffer[:y_size], height, width),
            u=Plane.from_buffer(buffer[y_size:y_size + uv_size], uv_height, uv_width),
            v=Plane.from_buffer(buffer[y_size + uv_size:], uv_height, uv_width),
        )
```

这里我们需要注意 `zeros` 方法中 U 和 V 平面的尺寸是 Y 的一半。也就是上一篇文章提到的 4:2:0 采样：色度分辨率在水平和垂直方向各减半，总数据量只有 Y 的 1/4。三个平面加起来，数据量是 Y 的 1.5 倍，相比 RGB 格式的 3 倍数据量减少了 50%。

`from_yuv420p` 方法展示了如何从原始字节流构建 Frame。YUV420p 这个名字怎么理解？拆开来看：YUV 表示颜色空间（Y 是亮度，U/V 是色度）；420 是色度采样格式（4:2:0 表示每 4 个亮度像素共享 1 组色度值，色度在水平和垂直方向都是亮度的一半）；p 代表 planar（平面式），表示 Y/U/V 三个通道分开存储。

一帧 1080p 的 YUV420p 数据在字节流中的布局是：

| 平面 | 尺寸 | 字节范围 | 大小 |
|------|------|----------|------|
| Y | 1920×1080 | `[0, 2073600)` | 2,073,600 bytes |
| U | 960×540 | `[2073600, 2592000)` | 518,400 bytes |
| V | 960×540 | `[2592000, 3110400)` | 518,400 bytes |

总计约 3MB。先存完整个 Y 平面，再存 U，最后存 V。这种布局正好符合我们的数据结构设计，`from_yuv420p` 只需要按偏移量切片即可。

在 `__main__.py` 的实现中可以看到，我们对 Y、U、V 三个平面分别处理：

```python
for plane_name, src_plane, dst_plane in [
    ("Y", frame.y, recon.y),
    ("U", frame.u, recon.u),
    ("V", frame.v, recon.v),
]:
    for blk in iterate_blocks(src_plane, block_size):
        ...  # 每个平面独立处理
```

### 🚀 进阶：PackedFrame 与内存池

其实上面的 `Frame` 实现已经足够教学使用了。但如果需要进一步优化性能，我们还可以进一步的做些设计。

#### PackedFrame：单次内存分配

普通 Frame 创建三个独立的 NumPy 数组，意味着三次内存分配。每次 `np.zeros()` 都可能触发一次 `malloc()` 系统调用，分配器还要维护内存块的元数据。

升级为 PackedFrame 的思路很简单：既然我们知道 Y/U/V 的总大小，为什么不一次性分配一块连续内存，然后用 view 切出三个平面？

```python
class PackedFrame:
    """Y/U/V 在连续内存中的视频帧"""
    __slots__ = ('_buffer', 'y', 'u', 'v', 'height', 'width', '_y_size', '_uv_size')

    def __init__(self, height: int, width: int, dtype: np.dtype = np.int16):
        # 一次分配所有内存
        total = height * width + 2 * (height // 2) * (width // 2)
        self._buffer = np.zeros(total, dtype=dtype, order='C')

        # Y/U/V 都是 view，零拷贝
        self.y = self._buffer[:height * width].reshape(height, width)
        ...
```

这种设计的好处：一次 syscall vs 三次 syscall，减少系统调用开销；Y/U/V 在内存中物理连续，可能获得更好的 cache locality（取决于访问模式）；导出 YUV420p 字节流时直接 `self._buffer.tobytes()` 即可，零拷贝。

当然，优化收益取决于具体场景。如果只处理几帧视频，节省的几微秒影响不大。但对于实时编码器，每一帧都要创建和销毁大量 buffer，累积起来的开销就变得显著了。

#### FrameBufferPool：对象池模式

当然我们还可以继续思考，比减少内存分配更有效的优化是什么？其实还可以完全避免分配。

因为视频编码器有一个特点：它会反复处理相同分辨率的帧。每一帧都分配新 buffer，处理完又释放，那这个时候这种 malloc/free 的开销就是被浪费的。

对象池（Object Pool）模式的思路很简单：预先分配一批 buffer，需要时从池中取，用完放回。这样内存分配只发生在初始化阶段，后续编码过程中完全避免了系统调用。

```python
pool = FrameBufferPool(1080, 1920, pool_size=4)
idx, frame = pool.acquire()  # 从池中获取
# ... 使用 frame ...
pool.release(idx)  # 归还到池中
```

这个模式在工业级编码器中非常常见，x264、x265 等主流编码器都有类似的 buffer 管理机制。nano-hevc 提供了一个简化版本，帮助理解设计思路。当需要处理 4K 60fps 的视频流时，这些优化累积起来可能会对整体性能产生决定性影响。（主包计划最后尝试一下真正去编码 4K 60fps 的视频流🐱）

## ⚠️ 存储格式 vs 计算格式

讨论完数据的「形状」之后，我们来分析数据的「类型」。这是一个容易踩坑的地方，值得单独说明。

在数据表示层，我们的数值范围通常在 8 到 10 bit。这里我们采用 8 bit 这一更常见的 setting，对应的最小可以选择的数据单位就是 `uint8`：每个像素一个字节，值域 0～255，正好对应人眼能分辨的亮度级别。

解决了存储的问题，但实际在编码器的内部计算链路中，`uint8` 是不够用的，这里有两个原因。

首先，残差可能是负数。

视频编码的核心思想是「预测 + 残差」。编码器先猜测当前块长什么样（预测），然后只传输猜错的部分（残差）。

数学表达式很简单：$Residual = Original - Prediction$

问题来了：如果预测值比原始值大怎么办？

比如原始像素值是 100，预测值是 150，残差就是 -50。如果用 `uint8` 存储，100 - 150 会得到 206（因为 uint8 的减法会 wrap around），而不是我们期望的 -50。

这种 underflow 一旦发生，后续的变换、量化、反量化、重建全都会出错，最终解码出来的图像会出现严重的色块和失真。而且这种错误很难 debug：数值看起来都是「正常」的 0～255 范围，但就是不对。

其次，DCT 变换后动态范围会扩大。

即使残差本身在 -255～255 范围内（用 int16 能轻松表示），经过 DCT 变换后，情况会变得更复杂。

DCT（离散余弦变换）的作用是把空域信号转换到频域。对于一个 N×N 的块，DCT 系数的理论最大值是原始像素值的 N 倍。

以 8×8 块为例：如果所有像素都是 255，DC 系数（左上角）会是 $255 \times 8 = 2040$。如果考虑到残差可能是负的（-255），DC 系数的范围就是 -2040～2040。

这还只是 DC 系数。AC 系数虽然通常较小，但在某些极端纹理下也可能很大。HEVC 标准规定，变换系数的位深最多比输入位深多 7 bit，也就是说 8 bit 输入可能产生 15 bit 的系数。

一旦在中间环节发生 overflow 或截断，最终画面就会出现无法挽回的色彩断层：某些区域突然变成纯黑或纯白，或者出现奇怪的马赛克。更糟糕的是，由于 HEVC 使用预测编码，一个块的错误会传播到后续所有依赖它的块。

因此，`int16` 是数据存储的安全底线。它能表示 -32768～32767 的范围，对于残差和大多数中间结果是足够的。

但要注意：DCT 变换的中间计算可能会溢出 int16。以 8×8 块为例，变换矩阵乘法涉及 8 个元素的累加，如果每个残差值接近 ±255，中间累加结果可能突破 int16 的边界。因此，变换核心计算应使用 `int32`，计算完成后再转回 `int16` 存储。这也是 HEVC 参考软件 HM 的做法。

在 nano-hevc 中，我们严格区分了存储格式（`uint8`）与计算格式（`int16`/`int32`）。注意 `Plane.zeros` 默认使用 `np.int16`。输入数据用 uint8 读取后立即转成有符号类型进行计算，最后输出时再 clip 回 0～255 的 uint8。这种「宽进窄出」的策略确保了计算过程中不会丢失精度。

对于 10 bit 视频（HDR 内容常用），需要用 `int32` 来保证安全。经过主包研究后发现，真正在生产环境使用的编码器通常会根据位深调整内部计算精度，比如 x264 在编译时通过 `--bit-depth` 选项确定（8-bit 和 10-bit 是不同的二进制），而 x265 则支持运行时通过 `--output-depth` 动态选择。

## 🧱 核心计算单元：Block

讨论完 Frame 和 Plane 这些「容器」之后，我们来分析编码器的核心计算单元：Block。

HEVC 与现代 Vision Transformer 或 CNN 的主要区别在于，它不是对整张图像进行均匀处理，而是严格基于 block 为单位进行计算的，而且 block 的大小可以动态变化。为什么？因为图像的不同区域复杂度差异很大。天空、墙壁这种平坦区域，用很粗糙的描述就够了；而人脸、文字这种细节丰富的区域，需要更精细的处理。通过自适应的块划分，编码器可以把更多的 bit 分配给复杂区域，在保证质量的同时压缩体积。

HEVC 采用四叉树递归划分：先把图像切成固定大小的 CTU（Coding Tree Unit，最大 64×64），然后根据内容复杂度动态细分。平坦区域可能保持 64×64 的大块，而纹理丰富的区域会一路细分到 8×8 甚至更小。同一帧画面中，不同位置的块大小可能完全不同。

这套机制（CTU → CU → PU → TU）确实非常复杂，我们会在后续文章中详细讲解。现在只需要知道：编码器需要频繁地在各种尺寸的像素块上进行操作，而且块的大小是动态变化的。

这就带来了一个重要的工程问题：如何高效地访问这些计算单元？

### BlockView：把块变成对象

最直接的做法是到处写 `frame_data[y:y+size, x:x+size]`，但这样 Block 的概念会散落在代码各处，坐标维护容易出错，边界处理逻辑也没有统一归属。

nano-hevc 的解法是引入 `BlockView` 类，把「一个块」抽象成对象：

```python
class BlockView:
    __slots__ = ('plane', 'x', 'y', 'size')

    def __init__(self, plane: Plane, x: int, y: int, size: int):
        self.plane, self.x, self.y, self.size = plane, x, y, size

    @property
    def pixels(self) -> np.ndarray:
        return self.plane.data[self.y:self.y + self.size, self.x:self.x + self.size]

    def get_top_neighbors(self) -> np.ndarray:
        if self.y == 0:
            return np.full(self.size, 128, dtype=self.plane.data.dtype)  # 边界填充
        return self.plane.data[self.y - 1, self.x:self.x + self.size].copy()
```

这样一来，函数签名从 `process_block(data, x, y, size)` 简化为 `process_block(blk)`，边界处理也被封装在 `get_top_neighbors()` 内部。调用方不用关心当前块是否在图像边缘。

这里用 128 作为边界默认值是一个简化处理。HEVC 标准的参考像素填充逻辑其实更复杂：如果 Top 不存在但 Left 存在，会用 Left 的最上像素去填充；只有当 Top 和 Left 都不存在（比如第一帧的第一个块）时，才 fallback 到 `1 << (bit_depth - 1)`（8 bit 下即 128）。对于教学项目，直接用 128 可以接受，但严格实现时需要补全这套逻辑。

遍历所有块也变得很简洁：

```python
for blk in iterate_blocks(plane, block_size):
    top = blk.get_top_neighbors()
    left = blk.get_left_neighbors()
    pred = intra_dc_predict(top, left, block_size)  # 预测函数专注于数学
```

## 🗺️ 全景图：HEVC 编码流水线

当我们把 Frame、Plane 和 BlockView 组合在一起时，一个符合 HEVC 标准的编码 pipeline 就自然浮现了。每一帧的处理流程如下：

<!-- Image prompt for nano-banana (google/gemini-3-pro-image-preview):
A futuristic 3D FRONT-FACING technical flowchart visualization in sleek glassmorphism style, clean white background with HIGH CONTRAST elements.

Main structure: HEVC video encoding pipeline displayed as floating semi-transparent frosted glass boxes, viewed from DIRECT FRONT ANGLE (not isometric).

The diagram shows two parallel paths. The top encoding path has 5 glass boxes connected by BOLD glowing cyan arrows flowing left to right: Prediction, Residual, Transform, Quantization, and Entropy. Each box contains BOLD BLACK text. The Entropy box leads to an output labeled Bitstream.

From the Quantization box, a BOLD downward arrow branches to the bottom reconstruction path with 3 glass boxes: Dequantization, Inverse Transform, and Reconstruction. The Reconstruction box leads to an output labeled Reconstructed Frame.

A curved BOLD glowing arrow connects from Reconstruction back up to Prediction, labeled reference for next block in BOLD DARK text. This feedback loop is essential for the encoding process.

All boxes have BOLD glowing cyan beveled edges with strong contrast and defined inner shadows. Arrows are BRIGHT CYAN with high visibility.

Top title plate with DARK BACKGROUND: HEVC ENCODING PIPELINE in bold WHITE text with high contrast.

Clean lines, strong shadows, HIGH CONTRAST colors, high-tech minimalist aesthetic. 4K resolution, ultra detailed, sharp edges.
-->

![HEVC Encoding Pipeline](./assets/hevc_pipeline.png)

用 README.md 中的代码示例来对应这个流程：

```python
# 1. 预测：用邻居像素猜测当前块
pred = intra_dc_predict(top, left, size=4)

# 2. 残差：原始 - 预测
residual = residual_block(orig, pred)

# 3. 变换：空域 -> 频域
# DST-VII 仅用于 4x4 Luma Intra 块，其他情况一律用 DCT-II
is_4x4_luma_intra = (size == 4) and is_luma and is_intra
coeff = forward_transform(residual, use_dst=is_4x4_luma_intra)

# 4. 量化：压缩系数
levels = quantize_block(coeff, qp=22)

# 5. 解码端（编码器内部也要做，为了重建）
recon_coeff = dequantize_block(levels, qp=22)
recon_residual = inverse_transform(recon_coeff, use_dst=is_4x4_luma_intra)
recon = np.clip(pred + recon_residual, 0, 255).astype(np.uint8)  # 重建块，必须 clip
```

为什么 4×4 Intra 要用 DST 而不是 DCT？这是 HEVC 相比 H.264 的一个改进。Intra 预测的残差有一个特点：靠近参考边界的地方残差通常较小（因为预测更准），离边界越远残差越大。这种「从边界逐渐增大」的形态恰好接近正弦波的半周期，所以 DST 比 DCT 能更高效地压缩能量。但这个效果在大块上不那么明显，所以 HEVC 只对 4×4 Luma Intra 启用 DST。

这里的 `np.clip` 是本质性的步骤，绝对不能省略。预测值和重建残差相加后，结果很可能超出 `[0, 255]` 范围：比如 `pred = 250`，`recon_residual = 20`，加起来就是 270。如果不做 clip，后续块用这个「溢出」的值做预测参考，错误会沿着扫描方向传播到整帧，最终产生严重的色块。

注意最后写回的是「重建帧」而非原始帧。后续块的预测要基于重建帧，这确保了编解码器的一致性：解码器没有原始帧，只能用重建帧做预测。

这个流程对每一帧的每一个块重复执行。一帧 1080p 视频约 32000 个 8×8 块，30fps 就是每秒近百万次迭代，这就是为什么内存布局和对象开销如此重要。

## ✨ 写在最后

深入 HEVC 这样复杂的工业标准时，凭着主包个人粗浅的经验，学习路径基本不可能从公式和 bitstream 操作开始（非常的劝退），而是必须得先看一些偏科普的讲义[1,2]，再思考一下标准文档中那些核心 entity（Frame、Plane、Block、Neighbor）之间的拓扑关系，反向思考一下它们被设计出来的原因。

比如本文中的例子，frames 和 planes 的 `__slots__`、NumPy 的 view 机制、以及对象池模式，都得是我们深入理解了设计模式后，才能够更容易理解 HEVC 接下来更复杂的操作。

接下来的 nano-hevc 系列博客将按照编码流水线的顺序展开，欢迎关注🐱：

| # | 主题 | 对应代码 | 状态 |
|---|------|----------|------|
| 1 | YUV 色彩空间 | - | ✅ 这里写完了|
| 2 | Frame/Plane/Block 数据结构 | `frame.py`, `block.py` | ✅ 这里写完了|
| 3 | 帧内预测 (DC/Planar/Angular) | `intra.py` | 📝 |
| 4 | 变换与量化 | `transform.py`, `quant.py` | 📝 |
| 5 | 扫描模式 | `scan.py` | 📝 |
| 6 | CABAC 熵编码 | `cabac.py` | 📝 |
| 7 | NAL 单元与比特流 | `nal.py` | 📝 |
| 8 | 完整编码器与率失真优化 | `encoder.py` | 📝 |

下一篇，我们将基于这套数据结构，开始实现 HEVC 真正的重磅核心模块：帧内预测。这也是视频编码最核心的部分。仅利用像素之间微小的空间相关性，到底能把数据压缩到什么程度？我们会实现 DC 预测、Planar 预测，以及 HEVC 标准规定的 33 种角度预测模式。敬请期待！

## 参考资料

[1] [HEVC/H.265 Video Coding Standard Tutorial](https://www.youtube.com/watch?v=Fawcboio6g4)

[2] [H.265/HEVC Tutorial (MIT/ISCAS 2014)](https://eems.mit.edu/wp-content/uploads/2014/06/H.265-HEVC-Tutorial-2014-ISCAS.pdf)
