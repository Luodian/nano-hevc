# birds.mp4 Video Structure Analysis

---

## 1. Container Layer: MP4

MP4 is not a codec — it's a **container**. It packages video streams, audio streams, subtitles, etc. together and records each frame's byte offset so the player can seek.

```
┌─ birds.mp4  ──────────────────────────────────────┐
│                                                   │
│  ┌─ ftyp ─┐  ┌─ moov (metadata)  ──────────────┐  │
│  │ isom   │  │ ┌─ trak #1 (video)  ──────────┐ │  │
│  │ mp41   │  │ │ codec: H.264 High@5.1       │ │  │
│  └────────┘  │ │ 3840x1620, 25fps, 765 frms  │ │  │
│              │ │ stts/stss/stco → seek table │ │  │
│              │ └─────────────────────────────┘ │  │
│              │ ┌─ trak #2 (audio)  ──────────┐ │  │
│              │ │ AAC / ...                   │ │  │
│              │ └─────────────────────────────┘ │  │
│              └─────────────────────────────────┘  │
│                                                   │
│  ┌─ mdat (actual data)  ──────────────────────┐   │
│  │ H.264 NAL units, audio samples, ...        │   │
│  │ (99%+ of file size)                        │   │
│  └────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────┘
```

Analogy: MP4 = shipping box, H.264 = the goods inside.

---

## 2. Codec Layer: H.264 Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| codec | H.264 High @ Level 5.1 | Standard profile for 4K, Level 5.1 allows up to 4096x2160@30fps |
| resolution | 3840x1620 | 64:27 ultrawide (cinematic aspect ratio) |
| framerate | 25 fps | 765 frames, 30.6 seconds |
| bitrate | ~5.96 Mbps | |
| pixel format | yuv420p | chroma resolution is half of luma (see below) |
| color space | bt709, tv range | |
| has_b_frames | 2 | B-frame reorder depth, not the number of B-frames |
| refs | 1 | reference frame count |

### YUV420p Pixel Layout

```
  Y (luma) 3840 x 1620          U (chroma) 1920 x 810     V (chroma) 1920 x 810
  ┌──────────────────────┐      ┌───────────┐              ┌───────────┐
  │                      │      │           │              │           │
  │  one Y value per     │      │  every    │              │  every    │
  │  pixel (0-255)       │      │  2x2 luma │              │  2x2 luma │
  │                      │      │  pixels   │              │  pixels   │
  │  3840 x 1620 = 6.2M  │      │  share    │              │  share    │
  │                      │      │  one U    │              │  one V    │
  │                      │      │  1920x810 │              │  1920x810 │
  │                      │      │  = 1.6M   │              │  = 1.6M   │
  └──────────────────────┘      └───────────┘              └───────────┘

  total per frame = 6.2M + 1.6M + 1.6M = 9.4M bytes
                  = 50% of RGB (RGB needs 3840x1620x3 = 18.7M)
```

---

## 3. Frame Types: I / P / B

### I-Frame = Keyframe = Self-Contained Picture

- References no other frame, can be decoded independently
- Therefore the largest: avg 79KB
- Video seek can only jump to I-frame positions

```
  I-frame: can only use already-reconstructed neighbors for prediction

  reconstructed (available)                not yet processed
  ┌─────────────────────────────────┬────────────────┐
  │  ## ## ## ## ## ## ## ## ## ##   │                │
  │  ## ## ## ## ## ## ## ## ## ##   │                │
  │  ## ## ## ## ## ## top[0..N]    │                │
  │  ## ## ## ## ## ## ──────────── │                │
  │  ## ## ## ## ## left │current│  │                │
  │  ## ## ## ## ##  [0] │ block │  │                │
  │  ## ## ## ## ##  [1] │  pred │  │                │
  │  ## ## ## ## ##  ... │=DC/Pl │  │                │
  │  ## ## ## ## ## ──── └───────┘  │                │
  │                                │                │
  └────────────────────────────────┴────────────────┘
                      scan direction -->
```

### P-Frame = Forward Prediction Frame

```
P-frame encoding = motion vector (MV) + residual

process:
  1. split current frame into blocks (4x4 to 64x64)
  2. for each block, search the reference frame: "which position looks most like me?"
  3. record the displacement --> motion vector (MV)
  4. use that position's pixels as prediction, subtract from actual --> residual
  5. transmit only MV + residual
```

```
  P-frame motion search:

  reference frame (decoded)         current frame (encoding)
  ┌────────────────────┐           ┌────────────────────┐
  │                    │           │                    │
  │    ┌────┐          │           │         ┌────┐     │
  │    │best│          │           │         │curr│     │
  │    │mtch│·····MV=(+3,+1)······│·········│blk │     │
  │    └────┘          │           │         └────┘     │
  │                    │           │                    │
  └────────────────────┘           └────────────────────┘

  transmitted data = MV(+3,+1) + residual(curr - best_match)
  NOT the raw pixel values of the block
```

P-frames are smaller than I-frames: avg 43KB (54% of I-frame)

### B-Frame = Bidirectional Prediction Frame

Predicts from both past and future reference frames simultaneously:

```
    ref frame (past)         B-frame (current)       ref frame (future)
    ┌──────────┐            ┌──────────┐            ┌──────────┐
    │          │─── MV_L0 ──>│  this    │<── MV_L1 ──│          │
    │          │            │  block's │            │          │
    └──────────┘            │ predict  │            └──────────┘
                            └──────────┘

    final prediction = (L0_pred + L1_pred + 1) >> 1
```

Two directions guessing together = more accurate prediction = smaller residual. B-frames are the smallest: avg 25KB

---

## 4. GOP Structure

GOP = Group of Pictures, from one I-frame to the next.

```
I-frame positions: [0, 128, 240, 440, 640]
GOP sizes:         [128, 112, 200, 200]   <-- unequal spacing, scene-adaptive

first 32 frames pattern:
  IBBBPBBBPBBBPBBBPBBBPBBBPBBBPBBB
  ^       ^       ^
  I-frm   P-frm   P-frm
  (anchor)

mini-GOP = [P B B B], length 4
```

```
  GOP timeline (unequal spacing, scene-adaptive):

  frame  0         128   240                440                640   765
         |          |     |                  |                  |     |
         v          v     v                  v                  v     v
         I----------I-----I------------------I------------------I-----|
         |<- 128  ->| 112 |<-    200       ->|<-    200       ->| 125 |
         |          |  ^  |                  |                  |     |
         |  GOP 0   |  |  |     GOP 2       |     GOP 3       |GOP 4|
         |          |scene|                  |                  |     |
         |          | cut |                  |                  |     |
                       ^
              88 frames shorter than standard GOP = content changed here
```

Unequal GOP spacing is caused by libx264's scenecut detector, which inserts an I-frame when content changes dramatically. The gap between frame 128 and 240 is only 112 frames (instead of the usual 200), indicating a scene change near frame 240.

### Decode Order != Display Order

```
display order:  I_0  B_1  B_2  B_3  P_4  B_5  B_6  B_7  P_8
decode order:   I_0  P_4  B_1  B_2  B_3  P_8  B_5  B_6  B_7
                      ^
                      P_4 must be decoded first because B_1-3 need it
                      as a backward reference
```

---

## 5. Bit Allocation

| Frame Type | Count | % of Frames | Avg Size | Total Size | % of Bitrate |
|------------|-------|-------------|----------|------------|-------------|
| I | 5 | 0.7% | 79 KB | 394 KB | 1.7% |
| P | 192 | 25.1% | 43 KB | 8,254 KB | 36.2% |
| B | 568 | 74.2% | 25 KB | 14,143 KB | 62.1% |

```
  average frame size comparison:

  I ########################################  79 KB   borrows from 0 refs
  P #####################                    43 KB   borrows from 1 ref
  B #############                            25 KB   borrows from 2 refs
    |-----|-----|-----|-----|-----|-----|-----|-----|
    0    10    20    30    40    50    60    70   80 KB
```

Size ratio I:P:B = 3.2 : 1.7 : 1 — directly reflects "how much can be borrowed from other frames."

### I-Frame Sizes at Each Position

```
idx=  0: I  69,801 bytes
idx=128: I  85,764 bytes
idx=240: I  77,585 bytes
idx=440: I  64,601 bytes  <-- smallest, simplest scene
idx=640: I  96,604 bytes  <-- largest, most complex scene
```

```
  per I-frame size (direct indicator of scene complexity):

  idx=  0 ####################################              69.8 KB
  idx=128 ###########################################       85.8 KB
  idx=240 ########################################          77.6 KB
  idx=440 #################################                 64.6 KB  <-- simplest
  idx=640 ################################################# 96.6 KB  <-- most complex
          |------|------|------|------|------|
          0     20     40     60     80    100 KB
```

Frame 640's I-frame is the largest, and the B-frames immediately following it are also larger (~30KB vs avg 25KB), indicating the most visually complex moment in the video.

---

## 6. Implications for Neural Video Compression

### I-Frame Placement

Encoder I-frame placement is not fixed-interval but content-adaptive. Useful signals:
- MV energy spike --> motion discontinuity
- residual energy spike --> prediction failure
- P-frame pkt_size spike --> compression efficiency drop

### P-Frame Prediction Target Hierarchy

```
  abstraction          target                            what the network learns
  ───────────────────────────────────────────────────────────────────────────
      ^
      |   Layer 4    z_P - warp(z_I, MV)                latent-space residual
      |             ┌──────────────────────┐            (hardest, highest info density)
      |             │ warp + residual in   │
      |             │ learned feature space│
      |             └──────────────────────┘
      |
      |   Layer 3    MV + pixel residual                motion + residual
      |             ┌──────────────────────┐
      |             │ predict "how to move"│
      |             │ then compensate diff │
      |             └──────────────────────┘
      |
      |   Layer 2    MV_L0 field                        optical flow / motion field
      |             ┌──────────────────────┐
      |             │ learn motion only,   │
      |             │ ignore residual      │
      |             └──────────────────────┘
      |
      |   Layer 1    raw P-frame pixels                 end-to-end pixel prediction
      |             ┌──────────────────────┐            (easiest, most info waste)
      |             │ directly predict     │
      |             │ full frame pixels    │
      |             └──────────────────────┘
      v
  abstraction
```

### Characteristics of This Video

- P/I compression ratio is only 1.8:1 (bird non-rigid motion, block-matching is mediocre)
- B-frames are extremely efficient (25KB, 1/3 of I) -- bidirectional prediction works well for animal motion
- GOP of 200 frames means 1 I-frame must "support" a prediction chain spanning 8 seconds

```
  prediction chain within 1 GOP (200 frames = 8 seconds):

  I_0 --ref--> P_4 --ref--> P_8 --ref--> ... --ref--> P_196
  |             |  \          |  \                      |  \
  |      +--ref-+   ref--+   |   ref--+          +--ref-+   ref--+
  v      v        v       v   v       v          v        v       v
  B_1   B_2      B_3     B_5  B_6    B_7  ...  B_193   B_194   B_195

  I_0's reconstruction quality sets the ceiling for the entire chain.
  as the chain lengthens, quantization error accumulates per frame (drift).
  this is why periodic I-frame insertion is needed to "reset" reference quality.
```

---

## 7. Frame Representation: What Gets Stored

### I-Frame: Dense Spatial Encoding

Every block is independently coded. No block can be skipped.

```
  I-frame block payload:
  ┌─────────────────────────────────┐
  │  intra_pred_mode  (DC/Planar/..)│   prediction direction
  │  cbp  (coded block pattern)     │   which sub-blocks have coefficients
  │  qp_delta                       │   quantization step change
  │  DCT coefficients  ← 90%+ cost │   transform-domain residual
  └─────────────────────────────────┘

  3840/16 × 1620/16 = 24,480 macroblocks
  info density: 69,801 bytes / 24,480 = 2.85 bytes/block  (dense)
```

### P-Frame: Sparse Delta Encoding

P-frame does not store pixels. It stores **edit instructions** against a reference frame.

```
  P-frame block payload (3 modes):

  ┌── skip (majority of blocks) ────┐
  │  skip_flag = 1                  │   1 bit, copy from ref, zero cost
  └─────────────────────────────────┘

  ┌── MV only ──────────────────────┐
  │  skip_flag = 0                  │
  │  MV(dx, dy)       ← few bits    │   shift from ref, residual ≈ 0
  │  cbf = 0                        │
  └─────────────────────────────────┘

  ┌── MV + residual ────────────────┐
  │  skip_flag = 0                  │
  │  MV(dx, dy)                     │   shift from ref
  │  cbf = 1                        │
  │  DCT coefficients ← sparse      │   compensate remaining difference
  └─────────────────────────────────┘

  info density: 30,314 bytes / 24,480 = 1.24 bytes/block  (sparse, 43% of I)
```

Spatial distribution of block types for birds.mp4 frame 4:

```
  ┌──────────────────────────────────────────┐
  │  skip  skip  skip  skip  skip  skip      │   sky: identical to reference
  │  skip  skip  skip  skip  skip  skip      │   zero cost
  │  skip  skip  skip  skip  skip  skip      │
  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ───     │
  │  mv    mv    MV+R  MV+R  mv    skip      │   tree branch: slight sway
  │  skip  mv    MV+R  MV+R  MV+R  mv        │   small MV, small residual
  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ───     │
  │  MV+R  MV+R  MV+R  MV+R  MV+R  MV+R      │   bird: non-rigid motion
  │  MV+R  MV+R  MV+R  MV+R  MV+R  MV+R      │   large MV, significant residual
  │  MV+R  MV+R  MV+R  MV+R  MV+R  MV+R      │   (all bytes concentrate here)
  └──────────────────────────────────────────┘
```

### B-Frame: Very Sparse Bidirectional Encoding

```
  B-frame block payload:
  ┌─────────────────────────────────────┐
  │  MV_L0(dx,dy)  → ref past  (I_0)   │   warp from past
  │  MV_L1(dx,dy)  → ref future (P_4)  │   warp from future
  │  pred = (L0 + L1 + 1) >> 1         │   average cancels opposite errors
  │  residual = orig - pred             │   even sparser than P-frame
  └─────────────────────────────────────┘

  info density: 12,934 bytes / 24,480 = 0.53 bytes/block  (19% of I)
```

### Summary

```
  what each frame type transmits:

  I:  Σ [ pred_mode + DCT(orig − spatial_pred) ]            DENSE
  P:  Σ [ MV + DCT(orig − warp(ref, MV)) ]                  SPARSE
  B:  Σ [ MV₀ + MV₁ + DCT(orig − avg(warp₀, warp₁)) ]     VERY SPARSE

  I ████████████████████████████████████  69,801 bytes
  P ████████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  30,314 bytes  (43% of I)
  B ████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  12,934 bytes  (19% of I)
       ████ actual info   ▒▒▒ zeros (skip blocks)
```

---

## 8. Autoregressive Prediction Target: Codec Token Sequences

Sections 1–7 describe the traditional codec pipeline. This section bridges to neural compression by showing that the codec's output is a **token sequence** with a well-defined autoregressive probability structure — the same structure that LLMs use.

### 8.1 Codec Output as Token Sequence

The H.264 encoder does not output raw pixels. It outputs a sequence of **syntax elements** (decisions) that the decoder replays to reconstruct the frame. We extract these decisions directly from the compressed bitstream via FFmpeg's motion vector export API — no re-encoding needed. Each decision is captured as a `SymbolTrace`:

```python
class SymbolTrace(NamedTuple):
    poc: int          # frame index
    cu_x: int         # macroblock x coordinate (pixels)
    cu_y: int         # macroblock y coordinate (pixels)
    cu_size: int      # block size (16 for H.264 macroblocks)
    plane: str        # "Y"
    token_type: str   # syntax element name
    value: int        # ground-truth value
    vocab_size: int   # number of possible values
```

Token vocabulary (5 types, all categorical):

| token_type | vocab | values | when emitted |
|---|---|---|---|
| frame_type | 3 | I=0, P=1, B=2 | 1 per frame |
| mb_mode | 3 | intra=0, inter=1, skip=2 | 1 per macroblock |
| partition_mode | 4 | 16x16=0, 16x8=1, 8x16=2, 8x8=3 | inter MBs only |
| mv_x | 256 | nonlinear 256-bin discretization | per sub-partition |
| mv_y | 256 | nonlinear 256-bin discretization | per sub-partition |

Per-frame token sequence = frame_type, then per-MB in raster scan order:

```
z_frame = (frame_type,
           mb_mode₀, [partition_mode₀, mv_x₀₀, mv_y₀₀, …]?,
           mb_mode₁, [partition_mode₁, mv_x₁₀, mv_y₁₀, …]?,
           …)
```

where `?` denotes conditional presence — skip/intra MBs emit only `mb_mode` (1 token), inter MBs emit `mb_mode + partition_mode + MV pairs` (variable length, 4–11 tokens depending on partition).

Per-MB token count by mode:

```
skip:    mb_mode                                            = 1 token
intra:   mb_mode                                            = 1 token
inter:   mb_mode + partition_mode + N × (mv_x, mv_y)       = 3–11 tokens
         where N = {1, 2, 2, 4} for {16x16, 16x8, 8x16, 8x8}
```

The complete frame token sequence is the concatenation:

$$
\mathbf{z} = (z_1, z_2, \dots, z_T), \quad T = 1 + \sum_{b=1}^{N_{\text{MB}}} |\mathbf{z}_b|
$$

### 8.2 Autoregressive Factorization

An LLM factorizes text as:

$$
P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{<t})
$$

A codec language model factorizes the token sequence identically:

$$
P(\mathbf{z}) = \prod_{t=1}^{T} P(z_t \mid z_{<t})
$$

The current CTP model conditions on previous tokens and positional features (macroblock coordinates, POC). Reference frame pixels are not explicitly fed to the model — the codec's macroblock decisions already encode motion information implicitly.

The negative log-likelihood equals the total bitrate in bits:

$$
R = -\sum_{t=1}^{T} \log_2 P(z_t \mid z_{<t})
$$

Normalized per token, this is **Bits Per Token (BPT)** — the codec analogue of perplexity:

$$
\text{BPT} = \frac{R}{T} = -\frac{1}{T} \sum_{t=1}^{T} \log_2 P(z_t \mid z_{<t})
$$

Lower BPT = better model = fewer bits = better compression.

### 8.3 Conditional Probability Chain for One Macroblock

Expanding $P(\mathbf{z}_b \mid \text{ctx})$ for a single macroblock, where $\text{ctx} = (z_{<t},\; V_{\text{ref}},\; \text{position})$:

$$
P(\mathbf{z}_b \mid \text{ctx}) = P(\text{mb\_mode} \mid \text{ctx}) \;\times\;
\begin{cases}
1 & \text{if mode} \in \{\text{skip, intra}\} \\[6pt]
P(\text{part} \mid \text{ctx}) \;\times\; \displaystyle\prod_{k=1}^{N_{\text{part}}} P(\text{mv\_x}_k \mid \text{ctx}) \cdot P(\text{mv\_y}_k \mid \text{ctx})
& \text{if mode} = \text{inter}
\end{cases}
$$

where $N_{\text{part}} \in \{1, 2, 2, 4\}$ for partition modes $\{$16x16, 16x8, 8x16, 8x8$\}$.

Each token type maps to a categorical distribution:

| token_type | vocab | distribution | role |
|---|---|---|---|
| frame_type | 3 | $\text{Categorical}_3$ | I=0, P=1, B=2; 1 per frame, sets context for all MBs |
| mb_mode | 3 | $\text{Categorical}_3$ | intra=0, inter=1, skip=2; skip/intra terminate the MB immediately |
| partition_mode | 4 | $\text{Categorical}_4$ | sub-block layout: 16x16, 16x8, 8x16, 8x8; determines MV count |
| mv_x | 256 | $\text{Categorical}_{256}$ | horizontal motion, nonlinear bins (fine near zero, coarse in tails) |
| mv_y | 256 | $\text{Categorical}_{256}$ | vertical motion, same binning as mv_x |

The MV binning is nonlinear to match the real distribution of motion vectors — most are near zero (static/slow motion), few are large (fast motion):

```
bins   0-127:  values [-64, +63]     at 1x resolution (quarter-pel)
bins 128-191:  values [-192, -65]    at 2x step
               values [+64, +191]
bins 192-255:  values [-704, -193]   at 8x step
               values [+192, +703]
```

The **variable-length** property is key: skip and intra MBs cost 1 token, while a complex 8x8-partitioned inter MB costs 11 tokens. This naturally allocates modeling capacity to motion-rich regions — the same principle as HEVC's skip flag, but at macroblock granularity.

### 8.4 Traditional Context Models vs Neural LM

H.264's entropy coder (CABAC in High profile) is already an autoregressive model — a very simple one. It estimates $P(z_t \mid \text{ctx})$ using a 64-state lookup table indexed by a handful of handcrafted context features:

```
  CABAC context for P(mb_mode):
    features = (neighbor MB modes, slice type, position)
    → index into 64-state probability table
    → outputs one categorical probability

  context window: ~3 neighboring MBs
  model capacity: 64 states × ~50 context indices ≈ 3,200 parameters
```

A neural codec LM replaces this with a Transformer:

```
  Neural context for P(z_t):
    ┌─────────────────────────────────────────────────────┐
    │  previous_tokens       z_{<t} (causal attention)    │
    │  type_embedding        token_type identity          │
    │  position_embedding    (mb_x, mb_y, poc)            │
    │          ↓                                          │
    │  Transformer(causal sequence)                       │
    │          ↓                                          │
    │  h_t → type-conditioned prediction head             │
    └─────────────────────────────────────────────────────┘

  context window: all previous tokens in the frame
  model capacity: ~10M–1B parameters
```

Both optimize the same objective:

$$
\min_\theta \;-\sum_{t=1}^{T} \log_2 P_\theta(z_t \mid z_{<t})
$$

The difference is capacity and context window size.

### 8.5 Training Objective

Given a dataset of videos and their extracted token traces:

$$
\mathcal{L}(\theta) = -\mathbb{E}_{V \sim \text{data}} \left[ \sum_{t=1}^{T} \log P_\theta(z_t \mid z_{<t}) \right]
$$

All 5 MPEG-4 token types are categorical, so the per-token loss is cross-entropy with type-conditioned heads:

$$
\ell_t = -\log \text{softmax}(g_\theta^{(\tau_t)}(\mathbf{h}_t))_{z_t}
$$

where $\tau_t$ is the token type at position $t$, and $g_\theta^{(\tau)}$ is the prediction head for type $\tau$. Each head outputs logits of dimension equal to that type's vocabulary size (3, 3, 4, 256, or 256). All heads share the same Transformer backbone $\mathbf{h}_t$.

### 8.6 Comparison: GPT vs Codec LM

| | GPT | Codec LM |
|---|---|---|
| token source | BPE tokenizer on text | H.264 macroblock decisions from bitstream |
| conditioning | $w_{<t}$ | $z_{<t}$ (previous codec tokens) |
| vocabulary | fixed (32K–128K) | **mixed**: 5 token types with vocab 3–256 |
| prediction head | single softmax | **multi-head**: type-conditioned dispatch |
| metric | perplexity (PPL) | bits per token (BPT) |
| objective | $\min -\sum \log P(w_t \mid w_{<t})$ | $\min -\sum \log P(z_t \mid z_{<t})$ |

### 8.7 Concrete Example: birds.mp4 Frame 4 (P-Frame)

Applying the formulation to the actual P-frame. Frame starts with `frame_type=1` (P), then per-MB in raster scan:

```
frame_type=P                                                    1 token
MB(0,0)  sky:     z = [mb_mode=skip]                            1 token
MB(1,0)  sky:     z = [mb_mode=skip]                            1 token
  ...
MB(5,8)  bird:    z = [mb_mode=inter, partition=8x8,            11 tokens
                       mv_x₀=bin(+2), mv_y₀=bin(-1),
                       mv_x₁=bin(+3), mv_y₁=bin(0),
                       mv_x₂=bin(+1), mv_y₂=bin(-2),
                       mv_x₃=bin(+2), mv_y₃=bin(-1)]
  ...
```

```
  cost per MB type:

  sky MB:      P(mb_mode=skip | ctx) ≈ 0.99
               → −log₂(0.99) ≈ 0.015 bits          virtually free (1 token)

  bird MB:     11 tokens, each 0.5–8 bits
               → ~20–50 bits per MB                   all budget here
```

A good neural model should achieve:
- near-zero BPT on sky MBs (trivial to predict skip)
- low BPT on bird MBs by learning motion patterns from preceding tokens
- total BPT well below the uniform baseline (which ignores all context)

### 8.8 Extension to B-Frames

The same factorization applies. B-frames use `frame_type=2` and the motion vectors extracted from the bitstream already reflect bidirectional prediction. FFmpeg's motion vector export reports L0 and L1 motion vectors as separate entries per macroblock — we flatten them into the same `(mv_x, mv_y)` token pairs, and the model learns the L0/L1 distinction implicitly from sequence position and frame_type context.

$$
P(\mathbf{z}^B) = P(\text{frame\_type}=B) \;\times\; \prod_{b=1}^{N_{\text{MB}}} P(\mathbf{z}_b \mid z_{<b},\; \text{frame\_type}=B)
$$

In practice, B-frame MBs have higher skip rates (bidirectional prediction is more accurate), so the token sequences are shorter — matching the bit allocation pattern from section 5 where B-frames are 1/3 the size of I-frames.
