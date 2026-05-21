# comfyui-lrw-nodes

[![License: BSL-1.1](https://img.shields.io/badge/License-BSL%201.1-yellow.svg)](LICENSE)

ComfyUI custom nodes for **Riemannian geometry and Bayesian latent space manipulation** powered by [latent-riemannian-world](https://github.com/lajjadred/latent-riemannian-world).

Instead of treating latent space as flat Euclidean, these nodes compute **true geodesics**, **parallel transport** for style transfer, and **world model trajectories** — all connected directly to ComfyUI core nodes.

---

## Node List

### Core Bridge Nodes (lrw/core)

| Node | Description |
|---|---|
| VAE Decoder Bridge | Wraps VAE into lrw metric — start here |
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
| Geodesic Interpolate | Interpolate along true geodesic |
| Geodesic Distance | Compute geodesic distance between latents |
| SLERP Interpolate | Baseline spherical interpolation |

### Transport Nodes (lrw/transport)

| Node | Description |
|---|---|
| Parallel Transport | Transport style vector along geodesic (Pole Ladder or Schild's Ladder) |

### World Model Nodes (lrw/world)

| Node | Description |
|---|---|
| Latent Trajectory | Generate temporal trajectory via Riemannian dynamics |

---

## Example Workflows

### 1. Geodesic Interpolation
`examples/geodesic_interpolation_workflow.json`

Drag this JSON into ComfyUI to load the full workflow.

Pipeline: VAEEncode x2 → LRW_VAEDecoderBridge → LRW_GeodesicInterpolate → LRW_LatentBlend → KSampler → VAEDecode → SaveImage

### 2. Style Transfer via Parallel Transport
`examples/style_transfer_workflow.json`

Drag this JSON into ComfyUI to load the full workflow.

Pipeline: VAEEncode x3 → LRW_VAEDecoderBridge → LRW_LatentVectorFromDiff → LRW_ParallelTransport → LRW_ApplyTransportedVector → KSampler → VAEDecode → SaveImage

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

## Why Geodesic vs Linear?

| Method | Geometry | Quality |
|---|---|---|
| Linear (Lerp) | Euclidean straight line | Blurry midpoints |
| SLERP | Fixed spherical curvature | Better, approximate |
| Geodesic (LRW) | True manifold curvature | Semantically consistent |

---

## Requirements
- ComfyUI
- latent-riemannian-world >= 0.2.0
- torch >= 2.4

## License
BSL-1.1 — (c) 2025 lajjadred
