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
