# comfyui-lrw-nodes

[![License: BSL-1.1](https://img.shields.io/badge/License-BSL%201.1-yellow.svg)](LICENSE)

ComfyUI custom nodes for **latent-space geometry, geodesic interpolation, and WAN2.2 video guidance** powered by [latent-riemannian-world](https://github.com/lajjadred/latent-riemannian-world).

This project provides practical nodes for experimenting with latent-space paths in ComfyUI, including:

- geodesic and SLERP-style latent interpolation
- curvature and latent-distance analysis
- parallel transport style/vector operations
- WAN2.2 First-Last Frame video workflows with **LRW-guided latent blending**

> **WAN2.2 note:**  
> The WAN2.2 integration is designed as a practical, VRAM-safe workflow.  
> Direct WAN2.2 First-Last Frame generation remains the main continuity path, while LRW geodesic keyframes are softly blended into the video latent before sampling.  
> This allows the geodesic structure to influence intermediate motion without sacrificing first-to-last continuity.

---

## Demo

### Direct WAN2.2 FLF vs LRW-Guided Latent Blend

In both comparison videos:

- **Left:** Direct WAN2.2 First-Last Frame baseline
- **Right:** WAN2.2 with LRW geodesic-guided latent blend

The right-side result uses LRW geodesic keyframes as a latent guide before KSampler.

#### 4-step comparison

[Watch 4-step comparison](examples/media/output_compare_side_by_side.mp4)

```text
KSampler steps: 4
Left: Direct WAN2.2 FLF
Right: LRW-guided latent blend
```

#### 6-step comparison

[Watch 6-step comparison](examples/media/output_compare_side_by_side01.mp4)

```text
KSampler steps: 6
Left: Direct WAN2.2 FLF
Right: LRW-guided latent blend
```

Recommended demo workflow:

```text
examples/workflows/wan22_practical_direct_flf_plus_lrw_guided_blend.json
```

---

## What the WAN2.2 LRW workflow does

The practical WAN2.2 workflow uses a two-branch structure.

```text
Main continuity branch:

first image + last image
→ WanFirstLastFrameToVideo
→ base video latent

LRW guide branch:

first latent + last latent
→ LRW_WanTemporalMetric
→ LRW_WanCurvatureGuide
→ LRW_WanGeodesicKeyframes
→ geodesic guide latent

Final sampling branch:

base video latent + geodesic guide latent
→ LRW_WanLatentGuideBlend
→ KSampler
→ VAEDecode
→ VHS_VideoCombine
```

This keeps the target last frame active while allowing the LRW geodesic branch to influence the intermediate latent motion.

---

## Node List

### WAN2.2 Video Nodes (`lrw/wan`) ★ Core Feature

Designed for WAN2.2 First-Last Frame video generation.

| Node | Description |
|---|---|
| **LRW WAN Temporal Metric** | Creates a lightweight LRW-compatible metric descriptor for WAN latent-space analysis. Avoids WAN VAE decoder Jacobian paths for VRAM safety. |
| **LRW WAN Curvature Guide** | Estimates latent distance / curvature-like complexity between first and last latent states. Useful for choosing steps, frame count, and keyframe strategy. |
| **LRW WAN Geodesic Keyframes** | Generates LRW geodesic-style keyframe latents between first and last latent states. Supports WAN-safe 5D latent output. |
| **LRW Latent Keyframe Picker** | Selects one geodesic keyframe from a stacked keyframe latent batch. Can output WAN-compatible 5D latent format. |
| **LRW WAN Latent Guide Blend** | Blends a selected LRW geodesic guide into the actual WAN video latent before KSampler. This is the main practical node for making LRW affect video generation. |

### Core Bridge Nodes (`lrw/core`)

| Node | Description |
|---|---|
| **VAE Decoder Bridge** | Wraps a decoder/VAE-style function for LRW metric experiments. |
| **Latent Blend / Frame Select** | Selects or blends latent frames from a latent path. |
| **Latent Vector from Diff** | Creates a latent direction vector from two latents. |
| **Apply Transported Vector** | Applies a transported style/vector direction to a target latent. |

### Metric Nodes (`lrw/metric`)

| Node | Description |
|---|---|
| **Pullback Metric** | Computes pullback metric experiments for compatible decoders. |
| **Latent Curvature Map** | Visualizes local metric/volume behavior where supported. |

### Geodesic Nodes (`lrw/geodesic`)

| Node | Description |
|---|---|
| **Geodesic Interpolate** | Interpolates between latents using LRW geodesic tools where supported. |
| **Geodesic Distance** | Estimates geodesic distance between latent states. |
| **SLERP Interpolate** | Baseline spherical interpolation for comparison. |

### Transport Nodes (`lrw/transport`)

| Node | Description |
|---|---|
| **Parallel Transport** | Transports a latent/vector direction along a path using LRW transport tools. |

### World Model Nodes (`lrw/world`)

| Node | Description |
|---|---|
| **Latent Trajectory** | Generates latent-space trajectories for experimental world-model workflows. |

---

## Example Workflows

### 1. WAN2.2 Direct FLF + LRW-Guided Latent Blend

```text
examples/workflows/wan22_practical_direct_flf_plus_lrw_guided_blend.json
```

Recommended practical WAN2.2 workflow.

It keeps WAN2.2 First-Last Frame as the main continuity path, then injects LRW geodesic guidance into the video latent before KSampler.

```text
LoadImage(first) ─┐
                  ├─ WanFirstLastFrameToVideo ─┐
LoadImage(last)  ─┘                             │
                                                ├─ LRW_WanLatentGuideBlend → KSampler → VAEDecode → VHS_VideoCombine
VAEEncode(first) ─┐                             │
                  ├─ LRW_WanGeodesicKeyframes ─┘
VAEEncode(last)  ─┘
```

### 2. WAN2.2 Direct FLF Baseline

```text
examples/workflows/wan22_direct_flf_baseline.json
```

Baseline workflow for comparison.

### 3. Geodesic Interpolation for Images

```text
examples/workflows/geodesic_interpolation_workflow.json
```

Image-to-image latent geodesic interpolation for compatible image models.

### 4. Style Transfer via Parallel Transport

```text
examples/workflows/style_transfer_workflow.json
```

Transport style direction vectors along latent paths.

---

## Recommended repository layout

```text
comfyui-lrw-nodes/
├─ examples/
│  ├─ workflows/
│  │  ├─ wan22_direct_flf_baseline.json
│  │  └─ wan22_practical_direct_flf_plus_lrw_guided_blend.json
│  │
│  └─ media/
│     ├─ output_compare_side_by_side.mp4
│     └─ output_compare_side_by_side01.mp4
│
├─ nodes/
│  ├─ wan_nodes.py
│  ├─ latent_keyframe_picker.py
│  └─ lrw_wan_latent_guide_blend.py
│
├─ README.md
└─ requirements.txt
```

---

## Installation

### Via ComfyUI Manager

Search for:

```text
comfyui-lrw-nodes
```

### Manual installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lajjadred/comfyui-lrw-nodes
cd comfyui-lrw-nodes
pip install -r requirements.txt
```

Restart ComfyUI after installation.

---

## Required custom nodes for WAN2.2 workflows

The WAN2.2 demo workflows may require:

- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — `UnetLoaderGGUF`
- WAN2.2 First-Last Frame support — `WanFirstLastFrameToVideo`
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) — `VHS_VideoCombine`
- Optional: [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) for image resizing and utility nodes

---

## VRAM Guide

| VRAM | Recommended WAN2.2 Setting |
|---|---|
| 12GB | GGUF Q3/Q4, `blend_strength=0.05–0.10`, lower frame count |
| 16GB | GGUF Q3/Q4, `blend_strength=0.10–0.20`, 81 frames practical |
| 24GB+ | Larger models or higher steps, stronger experimentation possible |

For the practical LRW-guided WAN workflow:

```text
LRW_WanLatentGuideBlend
blend_strength: 0.10–0.20
time_schedule: middle_focus
keyframe_index: middle keyframe
```

Suggested starting point:

```text
blend_strength = 0.15
time_schedule = middle_focus
normalize_guide = true
```

---

## Why use LRW guidance with WAN2.2?

Direct WAN2.2 First-Last Frame generation is usually the best source of first-to-last continuity.  
However, with large changes in pose, gaze, framing, or camera distance, the intermediate motion can become abrupt or unstable.

LRW-guided latent blending adds a soft latent-space guide:

| Method | Role | Expected effect |
|---|---|---|
| Direct FLF | Main continuity path | Strong first-to-last arrival |
| LRW Geodesic Keyframes | Latent-space guide | Provides intermediate direction |
| LRW Latent Guide Blend | Practical injection point | Lets the geodesic guide influence KSampler input |
| Strong midpoint replacement | Not recommended as default | Can weaken arrival at the real last frame |

The recommended workflow does **not** replace the last frame with a midpoint.  
Instead, it keeps the actual last frame in WAN FLF and blends LRW guidance into the video latent before sampling.

---

## Mathematical note

For general compatible decoders, LRW can be used to explore pullback metrics, geodesic interpolation, transport, and latent-space trajectories.

For WAN2.2, the integration is intentionally practical and VRAM-safe:

- WAN VAE decoder Jacobian / dense pullback metric paths can be too expensive or incompatible with some `vmap`/in-place operations.
- The WAN workflow therefore uses LRW geodesic keyframes as a latent guide.
- The guide is softly blended into the WAN video latent rather than replacing WAN's first-last conditioning.

This keeps the workflow usable on consumer GPUs while still allowing LRW geometry to influence video generation.

---

## Requirements

- ComfyUI
- latent-riemannian-world >= 0.3.0
- torch >= 2.4
- Python compatible with your ComfyUI installation

---

## License

BSL-1.1 — (c) 2025 lajjadred
