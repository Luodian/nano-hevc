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

## 8. Autoregressive Prediction Target for P-Frames

Sections 1–7 describe the traditional codec pipeline. This section bridges to neural compression by showing that the codec's output is a **token sequence** with a well-defined autoregressive probability structure — the same structure that LLMs use.

### 8.1 Codec Output as Token Sequence

The HEVC encoder does not output raw pixels. It outputs a sequence of **syntax elements** (decisions) that the decoder replays to reconstruct the frame. nano-hevc's `SymbolTrace` captures exactly this sequence:

```python
class SymbolTrace(NamedTuple):
    poc: int          # frame index
    cu_x: int         # block x coordinate
    cu_y: int         # block y coordinate
    cu_size: int      # block size
    plane: str        # "Y", "U", "V"
    token_type: str   # syntax element name
    value: int        # ground-truth value
    vocab_size: int   # number of possible values (0 = open-ended)
```

For an I-frame block, the token sub-sequence is:

```
z_block^I = (split_cu, part_mode, intra_mode, cbf,
             last_x, last_y, sig₀, sig₁, …, gt1₀, …,
             gt2₀, …, sign₀, …, remaining₀, …)
```

For a P-frame block, the prediction source changes from spatial neighbors to temporal reference + motion vector:

```
z_block^P = (skip_flag,
             merge_flag?, merge_idx?,
             mvd_x?, mvd_y?,
             cbf,
             last_x?, last_y?, sig?, gt1?, gt2?, sign?, remaining?)
```

where `?` denotes conditional presence — `skip=1` terminates the sequence immediately, `cbf=0` omits all coefficient tokens.

The complete P-frame token sequence is the concatenation over all blocks in raster scan order:

$$
\mathbf{z} = (z_1, z_2, \dots, z_T), \quad T = \sum_{b=1}^{N_{\text{blocks}}} |\mathbf{z}_b|
$$

### 8.2 Autoregressive Factorization

An LLM factorizes text as:

$$
P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{<t})
$$

A codec language model factorizes the P-frame token sequence identically, with one addition — conditioning on the reference frame $V_{\text{ref}}$:

$$
P(\mathbf{z} \mid V_{\text{ref}}) = \prod_{t=1}^{T} P(z_t \mid z_{<t},\; V_{\text{ref}})
$$

The negative log-likelihood equals the total bitrate in bits:

$$
R = -\sum_{t=1}^{T} \log_2 P(z_t \mid z_{<t},\; V_{\text{ref}})
$$

Normalized per token, this is **Bits Per Token (BPT)** — the codec analogue of perplexity:

$$
\text{BPT} = \frac{R}{T} = -\frac{1}{T} \sum_{t=1}^{T} \log_2 P(z_t \mid z_{<t},\; V_{\text{ref}})
$$

Lower BPT = better model = fewer bits = better compression.

### 8.3 Conditional Probability Chain for One P-Frame Block

Expanding $P(\mathbf{z}_b \mid \text{ctx})$ for a single block, where $\text{ctx} = (z_{<t},\; V_{\text{ref}},\; \text{position})$:

$$
P(\mathbf{z}_b \mid \text{ctx}) = P(\text{skip} \mid \text{ctx}) \;\times\;
\begin{cases}
1 & \text{if skip}=1 \\[6pt]
P(\text{merge} \mid \text{ctx}) \;\times\; P(\mathbf{mv} \mid \text{ctx}) \;\times\; P(\text{cbf} \mid \text{ctx}) \;\times\;
\begin{cases}
1 & \text{if cbf}=0 \\[4pt]
\displaystyle\prod_{i} P(c_i \mid \text{ctx}) & \text{if cbf}=1
\end{cases}
& \text{if skip}=0
\end{cases}
$$

where $\{c_i\}$ is the coefficient token sub-sequence (last_x, last_y, sig flags, gt1, gt2, signs, remaining).

Each token maps to a specific distribution family based on its `vocab_size`:

| token_type | vocab | distribution | role |
|---|---|---|---|
| skip_flag | 2 | $\text{Bernoulli}$ | block identical to ref? if 1, entire block done — 1 bit, zero cost |
| merge_flag | 2 | $\text{Bernoulli}$ | reuse a neighbor block's MV? avoids transmitting MV explicitly |
| merge_idx | 5 | $\text{Categorical}_5$ | which neighbor's MV to reuse (left / above / above-right / below-left / temporal) |
| mvd_x | 0 (open) | $\text{DiscretizedLogistic}(\mu, s)$ | MV residual in x: actual_MV_x = predicted_MV_x + mvd_x |
| mvd_y | 0 (open) | $\text{DiscretizedLogistic}(\mu, s)$ | MV residual in y: actual_MV_y = predicted_MV_y + mvd_y |
| cbf | 2 | $\text{Bernoulli}$ | any non-zero DCT coefficient in this block? if 0, no coefficients transmitted |
| last_sig_x | $K$ | $\text{Categorical}_K$ | x-coordinate of the last non-zero coeff in scan order; everything after is zero |
| last_sig_y | $K$ | $\text{Categorical}_K$ | y-coordinate of the last non-zero coeff; together with last_x, sets the scan endpoint |
| sig_coeff_flag | 2 | $\text{Bernoulli}$ | is this scan position non-zero? one flag per position from last_pos back to 0 |
| coeff_gt1 | 2 | $\text{Bernoulli}$ | is \|coeff\| > 1? most quantized coefficients are ±1, so gt1=0 is dominant |
| coeff_gt2 | 2 | $\text{Bernoulli}$ | is \|coeff\| > 2? peels another layer; only for coefficients already known > 1 |
| coeff_sign | 2 | $\text{Bernoulli}(0.5)$ | positive or negative? statistically near-uniform, coded in CABAC bypass mode |
| coeff_remaining | 0 (open) | $\text{DiscretizedLogistic}(\mu, s)$ | leftover magnitude: \|coeff\| − 3; only when \|coeff\| > 2; Rice-Golomb coded |

Token types are ordered by the encoding decision tree. A P-frame block with quantized coefficients `[+5, -2, 0, +1, 0, ...]` produces:

```
skip=0 → merge=0 → mvd=(+1,+1) → cbf=1
  → last_x=3, last_y=0                  locate scan endpoint
  → sig=[0,1,1]                         which positions are non-zero
  → gt1=[1,1,0]                         peel layer: |coeff|>1?
  → gt2=[1,0]                           peel layer: |coeff|>2?
  → sign=[0,1,0]                        +5 is positive, -2 is negative, +1 is positive
  → remaining=[2]                        |5|-3=2
  = 19 tokens total

  decoder reconstructs:
    sig=1, gt1=1, gt2=1, remaining=2, sign=0 → +1+1+1+2 = +5  ✓
    sig=1, gt1=1, gt2=0,             sign=1 → -(1+1)   = -2  ✓
    sig=0                                   →            =  0  ✓
    sig=1, gt1=0,                     sign=0 → +1       = +1  ✓
```

The "onion peeling" strategy is critical: most coefficients are 0 (eliminated by sig_coeff_flag), most non-zero coefficients are ±1 (eliminated by coeff_gt1), and only a tiny fraction reach coeff_remaining. Each layer acts as a **sparsity filter**, with the early binary tokens handling the common case cheaply.

### 8.4 CABAC as a Minimal Autoregressive Model

CABAC is already an autoregressive model — a very simple one. It estimates $P(z_t \mid \text{ctx})$ using a 64-state lookup table indexed by a handful of handcrafted context features:

```
  CABAC context for P(sig_coeff_flag_i):
    features = (neighbor significance flags, scan position, block size)
    → index into 64-state probability table
    → outputs one Bernoulli probability

  context window: ~3 neighboring flags
  model capacity: 64 states × ~50 context indices ≈ 3,200 parameters
```

A neural codec LM replaces this with a Transformer:

```
  Neural context for P(z_t):
    ┌─────────────────────────────────────────────────────┐
    │  ref_frame_patches     V_ref → patch tokens (prefix)│
    │  previous_tokens       z_{<t} (causal attention)    │
    │  position_embedding    (cu_x, cu_y, plane, type)    │
    │          ↓                                          │
    │  Transformer(prefix + causal sequence)              │
    │          ↓                                          │
    │  h_t → type-conditioned prediction head             │
    └─────────────────────────────────────────────────────┘

  context window: entire reference frame + all previous tokens
  model capacity: ~10M–1B parameters
```

Both optimize the same objective:

$$
\min_\theta \;-\sum_{t=1}^{T} \log_2 P_\theta(z_t \mid z_{<t},\; V_{\text{ref}})
$$

The difference is capacity and context window size.

### 8.5 Training Objective

Given a dataset of video frames and their encoder traces:

$$
\mathcal{L}(\theta) = -\mathbb{E}_{V \sim \text{data}} \left[ \sum_{t=1}^{T} \log P_\theta(z_t \mid z_{<t},\; V_{\text{ref}}) \right]
$$

The per-token loss dispatches on `token_type`:

$$
\ell_t =
\begin{cases}
-\log \sigma(f_\theta(\mathbf{h}_t)) \cdot z_t - \log(1 - \sigma(f_\theta(\mathbf{h}_t))) \cdot (1-z_t)
  & \text{if } \text{vocab} = 2 \;\text{(BCE)} \\[6pt]
-\log \text{softmax}(g_\theta(\mathbf{h}_t))_{z_t}
  & \text{if } \text{vocab} > 2 \;\text{(CE)} \\[6pt]
-\log p_{\text{DL}}(z_t \mid \mu_\theta(\mathbf{h}_t),\; s_\theta(\mathbf{h}_t))
  & \text{if } \text{vocab} = 0 \;\text{(open-ended)}
\end{cases}
$$

where $p_{\text{DL}}$ is the discretized logistic likelihood, and $f_\theta, g_\theta, \mu_\theta, s_\theta$ are type-conditioned heads sharing the same Transformer backbone.

### 8.6 Comparison: GPT vs Codec LM

| | GPT | Codec LM |
|---|---|---|
| token source | BPE tokenizer on text | CABAC syntax elements from encoder |
| conditioning | $w_{<t}$ | $z_{<t}$ + $V_{\text{ref}}$ (reference frame) |
| vocabulary | fixed (32K–128K) | **mixed**: each `token_type` has different vocab |
| prediction head | single softmax | **multi-head**: type-conditioned dispatch |
| metric | perplexity (PPL) | bits per token (BPT) |
| objective | $\min -\sum \log P(w_t \mid w_{<t})$ | $\min -\sum \log P(z_t \mid z_{<t}, V_{\text{ref}})$ |

### 8.7 Concrete Example: birds.mp4 Frame 4 (P-Frame)

Applying the formulation to the actual P-frame:

```
block(0,0) sky:     z = [skip=1]                               1 token
block(0,1) sky:     z = [skip=1]                               1 token
  ...
block(5,8) bird:    z = [skip=0, merge=0, mvd=(+2,-1),        ~20 tokens
                         cbf=1, last_x=3, last_y=2,
                         sig=1,0,1,0,…, gt1=1,0,…,
                         sign=0,1,…, remaining=2,…]
  ...
```

```
  cost per block type:

  sky block:     P(skip=1 | ctx) ≈ 0.99
                 → −log₂(0.99) ≈ 0.015 bits        virtually free

  bird block:    ~20 tokens × 0.5–8 bits each
                 → ~30–80 bits per block              all budget here
```

A good neural model should achieve:
- near-zero BPT on sky blocks (trivial to predict skip=1)
- low BPT on bird blocks by leveraging reference frame attention (implicit motion compensation)
- total BPT well below the CABAC baseline (which uses only local context)

### 8.8 Extension to B-Frames

The same factorization applies. The token sequence gains a second MV:

$$
\mathbf{z}_b^B = (\text{skip},\; \mathbf{mv}_{L0}?,\; \mathbf{mv}_{L1}?,\; \text{cbf},\; \{c_i\}?)
$$

and the conditioning includes two reference frames:

$$
P(\mathbf{z}^B \mid V_{\text{ref}}^{L0},\; V_{\text{ref}}^{L1}) = \prod_{t=1}^{T} P(z_t \mid z_{<t},\; V_{\text{ref}}^{L0},\; V_{\text{ref}}^{L1})
$$

For the iGPT-style architecture, both references become prefix tokens:

```
  input sequence:
  [ref_L0_patch₁, …, ref_L0_patchN, SEP,
   ref_L1_patch₁, …, ref_L1_patchN, SEP,
   z₁, z₂, …, z_T]
   \___ prefix (bidirectional) ___/  \_ causal generate _/
```

The Transformer's attention pattern implicitly learns which reference to rely on per block — equivalent to the L0/L1/bi-pred mode selection in traditional codecs, but without hard-coded averaging.
