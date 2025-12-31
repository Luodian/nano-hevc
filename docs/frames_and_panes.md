# 🎬 手写视频编码器 (2)：使用 Python `__slots__` 优化内存占用

上一篇文章我们讨论了视频编码为什么选择 YUV 而不是 RGB。但理解「YUV 是什么」是一回事，如何在工程代码中合理地表示 YUV 数据结构又是另一回事，这个话题值得单独讨论。

在 nano-hevc 这个教学项目中，我们的目标不仅仅是实现算法的 pseudocode 式模拟，而是希望深入研究如何在 Python 语言下，贴合 HEVC 标准的设计思想，合理利用 Python 的语言特性，构建一个具有工业级代码质量的 HEVC 实现。

## 📦 数据的物理形态

在动手写代码之前，我们需要先理解：在 HEVC 视角下，一帧视频在内存里应该长什么样？

这个问题看起来很简单，但实际上隐藏着很多工程上的 tradeoff。不同的数据布局会直接影响后续算法的实现难度和运行效率。

### 为什么视频编码不用 HWC 格式？

做深度学习的同学肯定习惯了 `(H, W, C)` 这种 Packed 格式，毕竟卷积核需要同时看 RGB 三个 channel 的 feature。在 PyTorch 里，一张 1080p 的图片就是一个 `(1080, 1920, 3)` 的 tensor，三个 channel 的数据在内存中是交错存放的。

但在视频编码领域，情况有所不同。

让我们从 HEVC 编码器的工作流程说起。当编码器拿到一帧图像后，它要做的事情大致是这样的：

1. 把图像切成一个个小块（block）
2. 对每个块做预测（猜测这个块长什么样）
3. 计算预测误差（残差）
4. 对残差做变换和量化
5. 把量化后的系数编码成比特流

需要注意的是，上述步骤中 90% 的计算都在单个平面内独立完成。预测、变换、量化等操作，Y、U、V 三个通道各自独立处理，互不干扰。

这意味着，如果使用 HWC 格式存储数据，每次访问 Y 平面的一个 8×8 块时，实际上需要跳过中间的 U 和 V 数据，这会导致 cache 命中率下降。

此外，在 HEVC 的某些高级模式下，luma channel 的四叉树划分结构与 chroma channel 可以是完全不同的。这一特性在早期的 H.264 标准中并不存在，但在 HEVC 的 I 帧中，为了进一步提高压缩效率，标准允许亮度和色度拥有各自独立的四叉树划分。

> 💬 具体来说：亮度可能被划分为 8×8 的块，而色度可能仍然保持 16×16 的块大小。

如果 Y/U/V 混在一起存储，这种灵活性就难以实现。每次单独操作某个平面时，都需要先将数据拆分，处理完成后再合并，这会增加不必要的计算开销。

所以从第一行代码开始，我们就需要把一帧图像拆解为三个正交的二维平面。每个平面都是一个独立的二维数组，互不干扰。

### Plane：单个颜色平面

理解了为什么要分离 Y/U/V 之后，我们来看 nano-hevc 中 `Plane` 的设计。

这个类的职责很简单：封装一个二维 NumPy 数组，表示单个颜色通道。注意我们用的是 `__slots__` 而非普通的 class 或 dataclass：

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

代码很短，但里面有几个值得注意的细节。

### 🤔 为什么用 `__slots__`？

这是本文的核心优化之一，值得展开讲讲。

在 Python 中，每个对象默认都有一个 `__dict__` 属性，用来存储实例的属性。这个设计非常灵活——你可以随时给对象添加新属性，甚至可以在运行时动态修改对象的结构。

但灵活性是有代价的。

`__dict__` 本质上是一个哈希表，即使只存一个属性，也要维护哈希表的各种元数据：bucket 数组、哈希种子、负载因子等等。在 CPython 3.11 的实现中，一个空的 `__dict__` 就要占用约 48 字节，加上 Python 对象头的开销，每个实例至少需要 100~200 字节的额外内存。

对于普通的业务代码，这点开销完全可以忽略不计。但视频编码的场景有所不同，因为需要创建的对象数量非常庞大。

算一笔账：一帧 1080p 的视频，如果用 8×8 的块来处理，Y 平面就有 $(1080/8) \times (1920/8) = 32400$ 个块。加上 U 和 V 平面（尺寸减半），总共约 40000 个块。每个块要创建一个 `BlockView` 对象来表示……

如果每个对象需要 200 字节的额外内存，40000 个对象就是 8MB。这还只是单帧的开销。如果需要同时处理多帧（例如 B 帧需要参考前后帧），内存占用会进一步增加。

`__slots__` 的作用就是告诉 Python：「这个类的实例只会有这几个属性，不需要 `__dict__`」。Python 会把属性存储在一个固定大小的数组里，通过偏移量直接访问，就像 C 语言的 struct 一样。

用 `__slots__` 之后：
- 内存省 40~50%
- 属性访问从查字典变成直接按偏移量访问，更快

当需要创建几万个 `BlockView` 对象时，这个优化可以大大节省存储开销。

当然，`__slots__` 也有代价：你不能再动态添加属性了。但对于数据结构类来说，这通常不是问题——我们本来就不希望在运行时随意修改对象结构。

> 💡 另外需要注意 `zeros` 工厂方法中的 `order='C'`。NumPy 数组有两种内存布局：C order（行优先）和 Fortran order（列优先）。视频编码中，我们通常按行扫描处理数据，所以 C order 能获得更好的 cache locality。这个参数虽然容易被忽略，但在大规模数据处理中，cache 命中率的差异可能带来 2~3 倍的性能差距。

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

注意 `zeros` 方法中 U 和 V 平面的尺寸是 Y 的一半。这就是上一篇文章提到的 4:2:0 采样——色度分辨率在水平和垂直方向各减半，总数据量只有 Y 的 1/4。三个平面加起来，数据量是 Y 的 1.5 倍，相比 RGB 格式的 3 倍数据量减少了 50%。

`from_yuv420p` 方法展示了如何从原始字节流构建 Frame。YUV420p 是最常见的 raw video 格式，数据布局是先存完整个 Y 平面，再存 U 平面，最后存 V 平面。这种「planar」布局正好符合我们的数据结构设计。

在 `__main__.py` 的编码循环中可以看到，我们对 Y、U、V 三个平面分别遍历：

```python
for plane_name, src_plane, dst_plane in [
    ("Y", frame.y, recon.y),
    ("U", frame.u, recon.u),
    ("V", frame.v, recon.v),
]:
    for blk in iterate_blocks(src_plane, block_size):
        ...  # 每个平面独立处理
```

这种设计让代码结构非常清晰：三个平面的处理逻辑完全对称，只是数据不同。

### 🚀 进阶：PackedFrame 与内存池

上面的 `Frame` 实现已经足够教学使用，但如果需要进一步优化性能，nano-hevc 还提供了两个高级抽象。

#### PackedFrame：单次内存分配

普通 Frame 创建三个独立的 NumPy 数组，意味着三次内存分配。在操作系统层面，每次 `np.zeros()` 都可能触发一次 `malloc()` 系统调用，分配器还要维护内存块的元数据。

PackedFrame 的思路是：既然我们知道 Y/U/V 的总大小，为什么不一次性分配一块连续内存，然后用 view 切出三个平面呢？

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

这种设计的优势包括：
- 一次 syscall vs 三次 syscall，减少系统调用开销
- Y/U/V 在内存中物理连续，可能获得更好的 cache locality（取决于访问模式）
- 导出 YUV420p 字节流时，直接 `self._buffer.tobytes()` 即可，零拷贝

当然，这个优化的收益取决于具体场景。如果只处理几帧视频，节省的几微秒可能影响不大。但对于实时编码器，每一帧都要创建和销毁大量 buffer，累积起来的开销会变得显著。

#### FrameBufferPool：对象池模式

比减少内存分配更有效的优化是——完全避免分配。

视频编码器有一个特点：它会反复处理相同分辨率的帧。每一帧都分配新 buffer，处理完又释放，这种 malloc/free 的开销完全是浪费。

对象池（Object Pool）模式的思路很简单：预先分配一批 buffer，需要时从池中取，用完放回。这样内存分配只发生在初始化阶段，后续编码过程中完全避免了系统调用。

```python
pool = FrameBufferPool(1080, 1920, pool_size=4)
idx, frame = pool.acquire()  # 从池中获取
# ... 使用 frame ...
pool.release(idx)  # 归还到池中
```

这个模式在工业级编码器中非常常见。x264、x265 等主流编码器都有类似的 buffer 管理机制。在 nano-hevc 的教学代码中，我们提供了一个简化版本，让大家理解设计思路。

这些优化在教学代码中是可选的，但它们展示了工业级编码器的设计思路。当需要处理 4K 60fps 的视频流时，这些优化累积起来可能会对整体性能产生决定性影响。

## ⚠️ 数据的存储格式和计算格式

讨论完数据的「形状」之后，我们来分析数据的「类型」。这是一个容易出错的地方，值得单独说明。

在数据表示层，我们的数值范围通常在 8 到 10 bit。这里我们采用 8 bit 这一更常见的 setting，对应就是 `uint8`——每个像素一个字节，值域 0~255，正好对应人眼能分辨的亮度级别。

解决了存储的问题，但实际在编码器的内部计算链路中，`uint8` 是不够用的。这里有两个原因，下面分别进行说明。

### 问题 1：残差可能是负数

视频编码的核心思想是「预测 + 残差」。编码器先猜测当前块长什么样（预测），然后只传输猜错的部分（残差）。

数学表达式很简单：$Residual = Original - Prediction$

问题来了：如果预测值比原始值大怎么办？

比如原始像素值是 100，预测值是 150，残差就是 -50。如果用 `uint8` 存储，100 - 150 会得到 206（因为 uint8 的减法会 wrap around），而不是我们期望的 -50。

这种 underflow 一旦发生，后续的变换、量化、反量化、重建全都会出错，最终解码出来的图像会出现严重的色块和失真。而且这种错误很难 debug——数值看起来都是「正常」的 0~255 范围，但就是不对。

### 问题 2：DCT 变换后动态范围扩大

即使残差本身在 -255~255 范围内（用 int16 能轻松表示），经过 DCT 变换后，情况会变得更复杂。

DCT（离散余弦变换）的作用是把空域信号转换到频域。对于一个 N×N 的块，DCT 系数的理论最大值是原始像素值的 N 倍。

以 8×8 块为例：如果所有像素都是 255，DC 系数（左上角）会是 $255 \times 8 = 2040$。如果考虑到残差可能是负的（-255），DC 系数的范围就是 -2040~2040。

这还只是 DC 系数。AC 系数虽然通常较小，但在某些极端纹理下也可能很大。HEVC 标准规定，变换系数的位深最多比输入位深多 7 bit，也就是说 8 bit 输入可能产生 15 bit 的系数。

一旦在中间环节发生 overflow 或截断，最终画面就会出现无法挽回的色彩断层——某些区域突然变成纯黑或纯白，或者出现奇怪的马赛克。更糟糕的是，由于 HEVC 使用预测编码，一个块的错误会传播到后续所有依赖它的块。

因此，`int16` 是内存表示的安全底线。它能表示 -32768~32767 的范围，足够容纳 8 bit 视频的所有中间计算结果。

在 nano-hevc 中，大家可以发现，我们严格区分了存储格式（`uint8`）与计算格式（`int16`）——注意 `Plane.zeros` 默认使用 `np.int16`。输入数据用 uint8 读取，立即转换成 int16 进行计算，最后输出时再 clip 回 0~255 的 uint8。这种「宽进窄出」的策略确保了计算过程中不会丢失精度。

对于 10 bit 视频（HDR 内容常用），需要用 `int32` 来保证安全。这也是为什么专业编码器通常会根据输入位深动态选择内部计算精度。

## 🧱 核心计算单元：Block

讨论完 Frame 和 Plane 这些「容器」之后，我们来分析编码器的核心计算单元：Block。

HEVC 与现代 Vision Transformer 或 CNN 的主要区别在于：它不是对整张图像进行均匀处理，而是严格基于 block 进行计算的。

为什么要切块？因为图像的不同区域复杂度差异很大。天空、墙壁这种平坦区域，用很粗糙的描述就够了；而人脸、文字这种细节丰富的区域，需要更精细的处理。通过自适应的块划分，编码器可以把更多的 bit 分配给复杂区域，在保证质量的同时压缩体积。

HEVC 标准把图像递归切分为 CTU（Coding Tree Unit），然后进一步细化为 CU、PU 和 TU。这套四叉树划分机制非常复杂，我们会在后续文章中详细讲解。现在只需要知道：编码器的主要工作就是在大量的 8×8、16×16 或 64×64 像素块上进行各种操作。

这就带来了一个重要的工程问题：如何高效地访问这些计算单元？

### 直接实现方式的问题

最直接的做法是到处写 `frame_data[y:y+size, x:x+size]`，同时维护 x、y、size 这三个变量。但这样做有几个麻烦：

第一，Block 的概念散落在代码各处，没有一个统一的抽象。在 `predict.py` 中会看到 `data[y:y+size, x:x+size]`，在 `transform.py` 中又看到相同的模式，在 `entropy.py` 中再次出现……每个地方都需要仔细维护坐标信息，一旦某处出错，问题定位会比较困难。

第二，邻居访问逻辑没有归属。帧内预测需要参考当前块上方和左侧的像素。但如果当前块在图像边缘怎么办？上方没有像素了。这种边界情况需要特殊处理（通常用固定值填充），但这个逻辑应该写在哪里？如果每个用到邻居像素的地方都自己处理边界，代码会变得很冗余，而且容易出现不一致。

第三，四叉树递归时，需要手动在递归函数中传递多个坐标参数。HEVC 的 CTU 划分是递归进行的：一个 64×64 的 CTU 可以分成四个 32×32 的 CU，每个 CU 又可以继续划分……递归函数的签名会变成 `process_cu(data, x, y, size, depth, ...)`，参数数量不断增加，代码可读性随之下降。

### ✨ BlockView：把块变成对象

nano-hevc 的解法是引入 `BlockView` 类。这个类的核心思想是：与其到处传递坐标，不如把「一个块」抽象成一个对象。

同样使用 `__slots__`：

```python
class BlockView:
    """Plane 内某个矩形块的视图（零拷贝）"""
    __slots__ = ('plane', 'x', 'y', 'size')

    def __init__(self, plane: Plane, x: int, y: int, size: int):
        self.plane = plane
        self.x = x
        self.y = y
        self.size = size

    @property
    def pixels(self) -> np.ndarray:
        return self.plane.data[self.y:self.y + self.size,
                               self.x:self.x + self.size]

    def copy_pixels(self) -> np.ndarray:
        return self.pixels.copy()

    def write_pixels(self, data: np.ndarray) -> None:
        self.plane.data[self.y:self.y + self.size,
                        self.x:self.x + self.size] = data
```

这个设计有几个值得注意的特点。

`BlockView` 持有一个 `Plane` 的引用和坐标信息，但不拥有任何像素数据。`pixels` 属性返回的是 NumPy 数组的一个 view（视图），指向原始 Plane 数据的对应区域。这意味着：

- 创建 BlockView 不会触发内存拷贝，开销几乎为零
- 通过 BlockView 修改像素会直接反映到原始 Plane 上
- 多个 BlockView 可以指向同一个 Plane 的不同区域

### 🎯 BlockView 真正解决的问题

需要澄清一点：`BlockView` 的核心价值不在于「实现零拷贝」。

NumPy 的切片赋值本来就是直接 modify 原数组，`frame_data[y:y+size, x:x+size] = something` 本身就没有 copy。所以从性能角度看，BlockView 并没有带来多大改进。

`BlockView` 真正解决的问题是封装和职责划分：

1️⃣ Block 变成了一个 object

一个 block 就代表「这一块区域」。你可以把它传给任何函数，函数内部通过 `blk.pixels` 访问数据、通过 `blk.x, blk.y` 知道自己在哪里、通过 `blk.size` 知道自己多大。不用再到处传递 x、y、size 三个参数了。

这种封装让函数签名变得简洁。比较一下：

```python
# 之前
def process_block(data, x, y, size, plane_ref):
    ...

# 之后
def process_block(blk: BlockView):
    ...
```

参数少了，函数的意图也更清晰了。

2️⃣ 邻居访问逻辑有了归属

边界检查和 padding 处理被封装在 BlockView 内部：

```python
def get_top_neighbors(self, count: Optional[int] = None) -> np.ndarray:
    """获取上方一行参考像素"""
    n = count if count is not None else self.size
    if self.y == 0:
        return np.full(n, 128, dtype=self.plane.data.dtype)  # 边界填充 128
    return self.plane.data[self.y - 1, self.x:self.x + n].copy()

def get_left_neighbors(self, count: Optional[int] = None) -> np.ndarray:
    """获取左侧一列参考像素"""
    n = count if count is not None else self.size
    if self.x == 0:
        return np.full(n, 128, dtype=self.plane.data.dtype)
    return self.plane.data[self.y:self.y + n, self.x - 1].copy()
```

这两个方法封装了所有边界处理逻辑。调用方完全不需要知道当前块是否位于边缘，直接调用 `get_top_neighbors()` 即可获得正确的结果。如果当前块位于边缘，方法返回填充值 128（8 bit 视频的中间值）；如果不在边缘，则返回真正的邻居像素。

为什么用 128 作为填充值？因为 128 是 0~255 的中点，代表「中等亮度」。对于帧内预测来说，这是一个相对「安全」的假设——在没有任何先验信息的情况下，假设边界外的像素是中等亮度，误差通常不会太大。HEVC 标准也是这样规定的。

3️⃣ 四叉树递归更自然

当我们把一个 64×64 的块分成四个 32×32 的子块时，可以直接创建四个子 BlockView：

```python
def split_quad(blk: BlockView) -> List[BlockView]:
    half = blk.size // 2
    return [
        BlockView(blk.plane, blk.x,        blk.y,        half),  # 左上
        BlockView(blk.plane, blk.x + half, blk.y,        half),  # 右上
        BlockView(blk.plane, blk.x,        blk.y + half, half),  # 左下
        BlockView(blk.plane, blk.x + half, blk.y + half, half),  # 右下
    ]
```

每个子块自己知道自己的位置和大小，递归函数只需要处理 BlockView，不用关心坐标计算。这种设计让四叉树划分的实现变得非常自然。

### 遍历所有块

有了 BlockView，遍历整个平面的所有块变得很简洁：

```python
def iterate_blocks(plane: Plane, block_size: int) -> Iterator[BlockView]:
    """遍历平面内所有非重叠块"""
    for y in range(0, plane.height, block_size):
        for x in range(0, plane.width, block_size):
            actual_size = min(block_size, plane.height - y, plane.width - x)
            if actual_size == block_size:
                yield BlockView(plane=plane, x=x, y=y, size=block_size)
```

这个生成器函数按光栅扫描顺序（从左到右、从上到下）遍历平面中所有完整的块。边缘不完整的块暂时跳过——在真实编码器中，边缘块需要特殊处理，但那是另一个话题了。

调用方只需要这样写，完全不用关心坐标计算和边界处理：

```python
for blk in iterate_blocks(plane, block_size):
    orig = blk.copy_pixels()
    top = blk.get_top_neighbors()      # 边界处理已封装
    left = blk.get_left_neighbors()    # 边界处理已封装

    # DC 预测：直接用邻居像素，不用关心边界
    dc_pred = intra_dc_predict(top, left, block_size)
```

得益于 BlockView 的封装，这些边界处理被下沉到数据结构层。`intra.py` 中的预测函数可以专注于数学公式本身：

```python
def intra_dc_predict(top: np.ndarray, left: np.ndarray, size: int) -> np.ndarray:
    """DC 预测：用邻居像素的均值填充整个块"""
    dc_value = (int(top.sum()) + int(left.sum()) + size) // (2 * size)
    return np.full((size, size), dc_value, dtype=np.int16)
```

这个函数完全不知道 boundary 的存在——它只知道自己会收到正确的邻居像素 array，然后计算均值、填充整个块。这种「关注点分离」（Separation of Concerns）让代码更容易理解、测试和维护。

## 🗺️ 全景图：HEVC 编码流水线

当我们把 Frame（数据容器）、Plane（解耦通道）和 BlockView（零拷贝视图）组合在一起时，一个符合 HEVC 标准定义的编码 pipeline 就自然浮现了。

请看这张数据流图，它概括了 nano-hevc 每一帧的 lifecycle：

```
┌─────────────────────────────────────────────────────────────────────┐
│                       HEVC 编码核心环路                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入视频帧 (YUV420p bytes)                                          │
│       │                                                             │
│       ▼                                                             │
│  Frame.from_yuv420p() ──→ 构建 Frame (Y/U/V Planes)                  │
│       │                                                             │
│       ▼                                                             │
│  1. 块扫描（Raster Scan）──→ iterate_blocks() 生成 BlockView          │
│       │                                                             │
│       ▼                                                             │
│  2. 预测 ──→ get_top/left_neighbors()，生成预测值                     │
│       │                                                             │
│       ▼                                                             │
│  3. 残差计算 ──→ 原始值减去预测值（需 int16 精度）                     │
│       │                                                             │
│       ▼                                                             │
│  4. 变换与量化 ──→ DCT 变换（频域分析）                               │
│       │                                                             │
│       ▼                                                             │
│  5. 重建 ──→ 预测值加上残差（反向恢复）                               │
│       │                                                             │
│       ▼                                                             │
│  6. 写入 ──→ BlockView.write_pixels() 原子化更新 Plane               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

让我解释一下这个流程中每个步骤与我们数据结构的对应关系：

步骤 1 是数据加载。`Frame.from_yuv420p()` 把原始字节流解析成三个 Plane，建立起我们的数据骨架。

步骤 2 是块扫描。`iterate_blocks()` 生成器按光栅顺序遍历每个平面，为每个块创建 BlockView。注意 Y、U、V 三个平面是分别处理的——这正是我们选择 planar 布局的原因。

步骤 3~4 是核心算法，包括预测、残差计算、变换和量化。这些操作都是在 int16 精度下进行的，避免溢出。BlockView 的 `get_top_neighbors()` 和 `get_left_neighbors()` 方法提供了预测所需的参考像素。

步骤 5 是重建。编码器需要维护一个「重建帧」，它是解码器能看到的画面。重建帧 = 预测 + 量化后的残差。后续块的预测要基于重建帧，而不是原始帧——这确保了编解码器的一致性。

步骤 6 是写回。`BlockView.write_pixels()` 把重建的像素写入 Plane。由于 BlockView 是 view 而不是 copy，写回操作直接修改原始 Plane 数据，为下一个块的预测提供参考。

这个流程会对每一帧的每一个块重复执行。一帧 1080p 视频有约 40000 个块，一秒 30 帧就是 120 万次迭代。这就是为什么我们要如此在意内存布局和对象开销——任何微小的低效都会被放大百万倍。

## ✨ 写在最后

> 好的架构，应该让代码读起来像 pseudocode。

当我们深入 HEVC 这样复杂的工业标准时，不要急着去堆砌 DCT 公式或 bitstream 操作。先停下来，思考标准文档中那些核心 entity（Frame、Plane、Block、Neighbor）之间的拓扑关系。

一旦这些关系被正确地映射到代码结构中（如 nano-hevc 所示），后面的算法实现就是水到渠成的事情了。在这个过程中，Python 的 `__slots__`、NumPy 的 view 机制、以及对象池模式，都是我们在保持代码清晰度的同时不牺牲性能的有力工具。

回顾一下本文的要点：

1. Y/U/V 分离存储（Planar）比交错存储（Packed/HWC）更适合视频编码，因为 90% 的计算都在单个平面内完成
2. `__slots__` 可以省 40~50% 的对象内存开销，对于海量小对象的场景至关重要
3. 存储用 uint8，计算用 int16，避免残差和 DCT 系数溢出
4. BlockView 把「块」抽象成对象，封装了坐标管理和边界处理，让算法代码更专注于数学本身

如果遇到类似的内存问题，这些技巧也是值得尝试的。

## 📌 下一篇预告

我们将基于这套数据结构，开始实现 HEVC 的核心模块：帧内预测。

仅仅利用像素之间微小的空间相关性，到底能把数据压缩到什么程度？我们会实现 DC 预测、Planar 预测，以及 HEVC 标准规定的 33 种角度预测模式。

敬请期待 🚀

如果这篇对你有帮助，点个赞让更多人看到吧！有问题欢迎评论区讨论 💬
