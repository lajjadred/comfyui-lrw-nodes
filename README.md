# comfyui-lrw-nodes

[![License: BSL-1.1](https://img.shields.io/badge/License-BSL%201.1-yellow.svg)](LICENSE)

ComfyUI custom nodes for **Riemannian geometry and Bayesian latent space manipulation** powered by [latent-riemannian-world](https://github.com/lajjadred/latent-riemannian-world) (v0.3.0+).

Instead of treating latent space as flat Euclidean, these nodes compute **true geodesics via BVP solver**, **parallel transport** for style transfer, **curvature-guided keyframe injection** for WAN2.2 video generation, and **world model trajectories**.

---

## Node List

### WAN2.2 Video Nodes (lrw/wan) ★ Core Feature

Designed specifically for WAN2.2 First-Last Frame video generation.
Replaces WAN's default Euclidean interpolation with true Riemannian geodesics,
reducing flickering and producing semantically consistent motion.

| Node | Description |
|---|---|
| WAN Temporal Metric | Computes Riemannian metric G(z) = J^T J from WAN VAE decoder. VRAM-aware (fp16/fp8, chunked Jacobian). |
| WAN Curvature Guide | Measures latent manifold curvature → recommends optimal keyframe count automatically |
| WAN Geodesic Keyframes | Generates true geodesic keyframes via **BVP Solver** — guarantees arrival at target latent |

**Required custom nodes for WAN workflows:**
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — `UnetLoaderGGUF`
- [Wan22FirstLastFrameToVideoLatent](https://github.com/stduhpf/ComfyUI--Wan22FirstLastFrameToVideoLatent) — FLF node
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) — `VHS_VideoCombine`

### Core Bridge Nodes (lrw/core)

| Node | Description |
|---|---|
| VAE Decoder Bridge | Wraps any VAE into lrw metric — general purpose |
| Latent Blend / Frame Select | Pick one frame from geodesic path for KSampler |
| Latent Vector from Diff | Create style direction vector from two latents |
| Apply Transported Vector | Add transported style vector to target latent |

### Metric Nodes (lrw/metric)

| Node | Description |
|---|---|
| Pullback Metric | Compute G(z) = J^T J from decoder Jacobian |
| Latent Curvature Map | Visualize local volume element sqrt(det G) |

### Geodesic Nodes (lrw/geodesic)

| Node | Description |
|---|---|
| Geodesic Interpolate | Interpolate along geodesic (Euler integration) |
| Geodesic Distance | Compute geodesic distance between latents |
| SLERP Interpolate | Baseline spherical interpolation for comparison |

### Transport Nodes (lrw/transport)

| Node | Description |
|---|---|
| Parallel Transport | Transport style vector along geodesic (Pole Ladder or Schild's Ladder) |

### World Model Nodes (lrw/world)

| Node | Description |
|---|---|
| Latent Trajectory | Generate temporal trajectory via Riemannian Langevin dynamics |

---

## Example Workflows

### 1. WAN2.2 FLF + BVP Geodesic + GGUF (Recommended)
`examples/wan22_lrw_bvp_geodesic_gguf_workflow.json`

WAN2.2 First-Last Frame with true Riemannian geodesic keyframe injection.
Optimized for 16GB VRAM using GGUF quantization.

```
UnetLoaderGGUF (wan2.2_i2v_5B_Q8_0.gguf)
      |
LoadImage(first) → VAEEncode → LRW_WanTemporalMetric → LRW_WanCurvatureGuide
LoadImage(last)  → VAEEncode ↗                       ↘ recommended_keyframes
                                                LRW_WanGeodesicKeyframes (BVP)
                                                           ↓
                              Wan22FirstLastFrameToVideoLatent → KSampler → VAEDecode → VHS_VideoCombine
```

### 2. WAN2.2 FLF + LRW Geodesic (fp16)
`examples/wan22_lrw_geodesic_flf_workflow.json`

Same pipeline without GGUF — for 24GB+ VRAM.

### 3. Geodesic Interpolation (Image)
`examples/geodesic_interpolation_workflow.json`

Image-to-image geodesic interpolation for SD-based models.

### 4. Style Transfer via Parallel Transport
`examples/style_transfer_workflow.json`

Transport style direction vector along geodesic to target latent.

---

## Installation

### Via ComfyUI Manager
Search for `comfyui-lrw-nodes`.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lajjadred/comfyui-lrw-nodes
cd comfyui-lrw-nodes
pip install -r requirements.txt
```

---

## VRAM Guide

| VRAM | Recommended Setting |
|---|---|
| 12GB | precision=fp16, chunk_frames=1, GGUF Q4_K_M |
| 16GB | precision=fp16, chunk_frames=1, GGUF Q8_0 |
| 24GB+ | precision=fp32, chunk_frames=2, fp16 safetensors |

---

## Why Geodesic vs Linear?

| Method | Geometry | Result |
|---|---|---|
| Linear (Lerp) | Euclidean straight line | Blurry, unnatural motion |
| SLERP | Fixed spherical curvature | Better, but approximate |
| Geodesic (Euler) | Riemannian manifold curvature | Semantically consistent |
| **BVP Geodesic (LRW)** | **True geodesic — guaranteed arrival** | **Best quality, no drift** |

---

## Requirements
- ComfyUI (latest)
- latent-riemannian-world >= 0.3.0
- torch >= 2.4

## License
BSL-1.1 — (c) 2025 lajjadred
