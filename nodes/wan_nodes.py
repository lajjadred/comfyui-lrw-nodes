"""
ComfyUI nodes for WAN2.2 video generation with LRW-compatible latent geodesic keyframes.

v8_wan5d_default — RAM/VRAM-safe revision with LRW-first backend

Main goals:
1. Keep the geodesic/keyframe feature visible in WAN workflows.
2. Avoid PullbackMetric, jacrev, vmap, and WAN VAE decoding inside metric computation.
3. Avoid constructing any D x D metric tensor such as torch.eye(D).
4. Keep compatibility with the lrw package by importing/initializing GeodesicSolver when available,
   while using lrw.geodesic.GeodesicSolver.interpolate() first when available, then falling back to a guaranteed memory-safe streaming SLERP implementation only if LRW fails.

Important note:
This is not a WAN VAE pullback-metric geodesic.
It is a norm-preserving latent-space spherical geodesic approximation.
That is the practical safe route for ComfyUI + WAN VAE because WAN VAE decode is not vmap-safe.
"""

from __future__ import annotations

import gc
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor


# ─────────────────────────────────────────────
# Memory utilities
# ─────────────────────────────────────────────

def _get_free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / 1e9


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _resolve_chunk(vram_mode: str, batch_size: int) -> int:
    """
    Conservative chunk resolver.

    The safest value is 1. We keep larger chunks only when there is enough free VRAM.
    This prevents sudden memory spikes during keyframe construction.
    """
    if batch_size <= 1:
        return 1

    if vram_mode == "low_12gb":
        return 1
    if vram_mode == "mid_16gb":
        return min(2, batch_size)
    if vram_mode == "high_24gb":
        return min(4, batch_size)

    free = _get_free_vram_gb()
    if free < 6:
        return 1
    if free < 12:
        return min(2, batch_size)
    return min(4, batch_size)


def _extract_spatial_latent(z: Tensor) -> Tuple[Tensor, Tuple[int, ...], str]:
    """
    Accepts:
    - image latent: (B, C, H, W)
    - video latent: (B, C, T, H, W)

    For metric/keyframe generation, a single spatial frame is used when a video latent is passed.
    """
    if z.ndim == 4:
        return z, tuple(z.shape[1:]), "4D"
    if z.ndim == 5:
        # Use the first temporal slice only to avoid multiplying memory by T.
        z_first = z[:, :, 0, :, :]
        return z_first, tuple(z_first.shape[1:]), "5D_first_frame"
    raise ValueError(f"Unexpected latent shape: {tuple(z.shape)}. Expected 4D or 5D latent.")


def _safe_slerp_pair(z0: Tensor, z1: Tensor, t_values: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Memory-safe SLERP for one chunk.

    z0: (B, D)
    z1: (B, D)
    t_values: (K,)
    returns: (K, B, D)

    This function is O(K*B*D), never O(D^2).
    It does not build any metric tensor.
    """
    # Keep calculation in float32 for numerical stability, then caller can cast back.
    z0 = z0.float()
    z1 = z1.float()

    norm0 = z0.norm(dim=-1, keepdim=True).clamp_min(eps)
    norm1 = z1.norm(dim=-1, keepdim=True).clamp_min(eps)

    u0 = z0 / norm0
    u1 = z1 / norm1

    dot = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)

    # Interpolate direction on the sphere.
    # Shape broadcasting:
    # t: (K,1,1), u0/u1: (1,B,D), omega: (1,B,1)
    t = t_values.to(device=z0.device, dtype=z0.dtype).view(-1, 1, 1)
    u0_b = u0.unsqueeze(0)
    u1_b = u1.unsqueeze(0)
    omega_b = omega.unsqueeze(0)
    sin_omega_b = sin_omega.unsqueeze(0)

    dir_path = (
        torch.sin((1.0 - t) * omega_b) / sin_omega_b * u0_b
        + torch.sin(t * omega_b) / sin_omega_b * u1_b
    )

    # Interpolate norm linearly. This avoids a hard assumption that both endpoint norms match.
    norm_path = (1.0 - t) * norm0.unsqueeze(0) + t * norm1.unsqueeze(0)

    # If endpoints are nearly parallel, SLERP becomes numerically equivalent to LERP.
    # Replace those rows with normalized LERP to prevent instability.
    near_parallel = (omega < 1e-4).view(1, -1, 1)
    lerp_path = (1.0 - t) * z0.unsqueeze(0) + t * z1.unsqueeze(0)

    return torch.where(near_parallel, lerp_path, dir_path * norm_path)


def _latent_lerp_pair(z0: Tensor, z1: Tensor, t_values: Tensor) -> Tensor:
    """Simple LERP path for comparison/debug mode."""
    t = t_values.to(device=z0.device, dtype=torch.float32).view(-1, 1, 1)
    return (1.0 - t) * z0.float().unsqueeze(0) + t * z1.float().unsqueeze(0)


# ─────────────────────────────────────────────
# LRW-compatible metric object, but no D x D tensor
# ─────────────────────────────────────────────

class _SafeLatentMetric:
    """
    LRW-compatible lightweight metric descriptor.

    Deliberately does not construct G(z) as a dense D x D matrix.
    That is the critical RAM/VRAM safety fix.

    Some lrw APIs may ask for metric_tensor(). If that happens, we fail loudly
    instead of allocating a huge identity matrix and killing the process.
    """

    def __init__(self, D: int, regularization: float = 1e-5):
        self.D = int(D)
        self.regularization = float(regularization)
        self.kind = "latent_spherical_safe_no_dense_metric"

    def metric_tensor(self, z: Tensor) -> Tensor:
        raise RuntimeError(
            "Dense metric_tensor() is disabled in v7_safe because it would allocate "
            f"a D x D matrix with D={self.D}. Use safe_slerp backend instead."
        )

    def geodesic_acceleration(self, z: Tensor, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    def local_volume_element(self, z: Tensor) -> Tensor:
        return torch.ones(z.shape[0], device=z.device, dtype=z.dtype)


def _try_init_lrw_solver(metric: _SafeLatentMetric, n_steps: int, step_size: float) -> Tuple[Optional[Any], str]:
    """
    Tries to initialize lrw.geodesic.GeodesicSolver.

    We do this to keep the node compatible with the lrw package without relying on
    memory-dangerous PullbackMetric/Jacobian/VAE paths.

    The default computation still uses safe_slerp because it is guaranteed O(D), not O(D^2).
    """
    try:
        from lrw.geodesic import GeodesicSolver
        solver = GeodesicSolver(metric=metric, n_steps=n_steps, step_size=step_size)
        return solver, "lrw.geodesic.GeodesicSolver initialized"
    except Exception as exc:
        return None, f"lrw solver unavailable: {type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────
# Node 1: WAN Temporal Metric
# ─────────────────────────────────────────────

class LRW_WanTemporalMetric:
    """
    Creates a lightweight LRW-compatible metric descriptor for WAN latent space.

    This node intentionally avoids:
    - WAN VAE decode
    - PullbackMetric
    - jacrev/vmap
    - dense D x D metric tensors

    The output is consumed by LRW_WanGeodesicKeyframes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "regularization": ("FLOAT", {
                    "default": 1e-5,
                    "min": 1e-8,
                    "max": 0.1,
                    "step": 1e-6,
                }),
                "precision": (["fp16", "fp32", "bf16"], {
                    "default": "fp16",
                }),
                "chunk_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("METRIC",)
    RETURN_NAMES = ("metric",)
    FUNCTION = "compute"
    CATEGORY = "lrw/wan"

    def compute(self, vae, latent: dict, regularization: float, precision: str, chunk_frames: int):
        z_raw = latent["samples"]
        z, spatial_shape, latent_mode = _extract_spatial_latent(z_raw)
        D = int(z[0].numel())

        metric = _SafeLatentMetric(D=D, regularization=regularization)

        info = {
            "metric": metric,
            "latent_shape": tuple(z.shape),
            "spatial_shape": spatial_shape,
            "D": D,
            "latent_mode": latent_mode,
            "precision": precision,
            "safe_mode": True,
            "dense_metric_disabled": True,
            "note": "No PullbackMetric, no vmap, no dense D x D tensor.",
        }

        _clear_memory()
        return (info,)


# ─────────────────────────────────────────────
# Node 2: WAN Geodesic Keyframes
# ─────────────────────────────────────────────

class LRW_WanGeodesicKeyframes:
    """
    Computes memory-safe latent geodesic keyframes.

    Default backend:
    - lrw_interpolate_try: uses lrw.geodesic.GeodesicSolver.interpolate() first.
      If LRW tries dense metric operations or fails, it safely falls back to safe_slerp.

    Optional backend:
    - safe_slerp: guaranteed O(K*B*D), no D x D matrix, no vmap, no VAE decode.
    - lerp_debug: linear interpolation for comparison.

    Recommended for 16GB VRAM:
    - backend: lrw_interpolate_try
    - vram_mode: low_12gb or mid_16gb
    - n_keyframes: 2 or 3
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_start": ("LATENT",),
                "latent_end": ("LATENT",),
                "metric": ("METRIC",),
                "n_keyframes": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Number of intermediate keyframes. 2-3 recommended for 16GB VRAM.",
                }),
                "n_geodesic_steps": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Only used when initializing lrw GeodesicSolver. safe_slerp does not need dense integration.",
                }),
                "geodesic_step_size": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 0.5,
                    "step": 0.001,
                }),
                "vram_mode": (["auto", "low_12gb", "mid_16gb", "high_24gb"], {
                    "default": "mid_16gb",
                }),
                "backend": (["lrw_interpolate_try", "safe_slerp", "lerp_debug"], {
                    "default": "lrw_interpolate_try",
                }),
                "output_dtype": (["same", "fp16", "bf16", "fp32"], {
                    "default": "same",
                }),
                "keyframe_output_layout": (["wan_5d", "image_4d", "same_as_input"], {
                    "default": "wan_5d",
                    "tooltip": "Use wan_5d before WAN VAEDecode. This prevents shape[4] IndexError.",
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("keyframe_latents", "latent_start", "info")
    FUNCTION = "compute"
    CATEGORY = "lrw/wan"

    def _cast_output(self, x: Tensor, source: Tensor, output_dtype: str) -> Tensor:
        if output_dtype == "same":
            return x.to(dtype=source.dtype)
        if output_dtype == "fp16":
            return x.to(dtype=torch.float16)
        if output_dtype == "bf16":
            return x.to(dtype=torch.bfloat16)
        return x.to(dtype=torch.float32)

    def compute(
        self,
        latent_start: dict,
        latent_end: dict,
        metric: dict,
        n_keyframes: int,
        n_geodesic_steps: int,
        geodesic_step_size: float,
        vram_mode: str,
        backend: str,
        output_dtype: str,
        keyframe_output_layout: str,
    ):
        with torch.no_grad():
            z0_raw = latent_start["samples"]
            z1_raw = latent_end["samples"]

            z0, spatial_shape, mode0 = _extract_spatial_latent(z0_raw)
            z1, spatial_shape_1, mode1 = _extract_spatial_latent(z1_raw)

            if tuple(z0.shape) != tuple(z1.shape):
                raise ValueError(
                    f"latent_start and latent_end must have the same processed shape. "
                    f"Got {tuple(z0.shape)} and {tuple(z1.shape)}."
                )

            B = z0.shape[0]
            D = int(z0[0].numel())
            chunk = _resolve_chunk(vram_mode, B)

            metric_obj = metric.get("metric", _SafeLatentMetric(D=D))
            lrw_solver, lrw_status = _try_init_lrw_solver(metric_obj, n_geodesic_steps, geodesic_step_size)

            # Intermediate t values only, excluding 0 and 1.
            t_values = torch.linspace(
                0.0,
                1.0,
                steps=n_keyframes + 2,
                device=z0.device,
                dtype=torch.float32,
            )[1:-1]

            # Allocate final output once. This prevents repeated stack/cat memory spikes.
            out = torch.empty(
                (B * n_keyframes, *spatial_shape),
                device=z0.device,
                dtype=torch.float32,
            )

            used_backend = backend
            fallback_reason = ""

            for b_start in range(0, B, chunk):
                b_end = min(b_start + chunk, B)
                z0_flat = z0[b_start:b_end].reshape(b_end - b_start, D).float()
                z1_flat = z1[b_start:b_end].reshape(b_end - b_start, D).float()

                path_flat = None

                if backend == "lerp_debug":
                    path_flat = _latent_lerp_pair(z0_flat, z1_flat, t_values)

                elif backend == "lrw_interpolate_try" and lrw_solver is not None:
                    try:
                        # Some lrw versions may support interpolate() without dense metric access.
                        # If it tries metric_tensor(), _SafeLatentMetric raises and we fallback.
                        full_path = lrw_solver.interpolate(
                            z0_flat,
                            z1_flat,
                            n_points=n_keyframes + 2,
                        )
                        path_flat = full_path[1:-1].float()
                    except Exception as exc:
                        used_backend = "safe_slerp_fallback"
                        fallback_reason = f"lrw interpolate failed safely: {type(exc).__name__}: {exc}"
                        path_flat = _safe_slerp_pair(z0_flat, z1_flat, t_values)

                else:
                    path_flat = _safe_slerp_pair(z0_flat, z1_flat, t_values)

                # path_flat: (K, chunk_B, D)
                for k in range(n_keyframes):
                    out_index_start = k * B + b_start
                    out_index_end = k * B + b_end
                    out[out_index_start:out_index_end] = path_flat[k].reshape(b_end - b_start, *spatial_shape)

                del z0_flat, z1_flat, path_flat
                _clear_memory()

            out = self._cast_output(out, z0, output_dtype)

            # Force output layout for downstream VAEDecode.
            # WAN VAE decode expects 5D latent: (B, C, T, H, W).
            # Since keyframes are stacked as (B*K, C, H, W), add T=1.
            before_layout_shape = tuple(out.shape)
            if keyframe_output_layout == "wan_5d":
                if out.ndim == 4:
                    out = out.unsqueeze(2)  # (B*K, C, 1, H, W)
            elif keyframe_output_layout == "image_4d":
                if out.ndim == 5:
                    out = out[:, :, 0, :, :]
            elif keyframe_output_layout == "same_as_input":
                # If original input was 5D, output 5D; otherwise keep 4D.
                if z0_raw.ndim == 5 and out.ndim == 4:
                    out = out.unsqueeze(2)
            after_layout_shape = tuple(out.shape)

            # Diagnostics
            z0_diag = z0.reshape(B, D).float()
            z1_diag = z1.reshape(B, D).float()

            cos_sim = torch.nn.functional.cosine_similarity(z0_diag, z1_diag, dim=-1).mean().item()
            cos_sim = max(-1.0, min(1.0, cos_sim))
            angle_deg = math.degrees(math.acos(cos_sim))

            mid_index = max(0, min(n_keyframes - 1, n_keyframes // 2))
            mid = out[mid_index * B:(mid_index + 1) * B].reshape(B, D).float()

            norm_z0 = z0_diag.norm(dim=-1).mean().item()
            norm_z1 = z1_diag.norm(dim=-1).mean().item()
            norm_mid = mid.norm(dim=-1).mean().item()
            expected_norm_mid = 0.5 * (norm_z0 + norm_z1)
            norm_ratio = norm_mid / (expected_norm_mid + 1e-8)

            vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            vram_free = _get_free_vram_gb()

            info_lines = [
                "LRW WAN Safe Latent Geodesic v7",
                f"backend: {used_backend}",
                f"lrw status: {lrw_status}",
                f"fallback: {fallback_reason or 'none'}",
                f"input modes: start={mode0}, end={mode1}",
                f"processed latent shape: {tuple(z0.shape)}",
                f"D: {D} | B: {B} | keyframes: {n_keyframes} | chunk: {chunk}",
                f"keyframe output layout: {keyframe_output_layout}",
                f"keyframe shape before layout: {before_layout_shape}",
                f"keyframe shape after layout: {after_layout_shape}",
                f"SLERP angle: {angle_deg:.2f} deg",
                f"mid norm ratio: {norm_ratio:.4f} (near 1.0 means stable norm)",
                f"dense D x D metric: disabled",
                f"VRAM allocated: {vram_used:.2f} GB | VRAM free: {vram_free:.2f} GB",
            ]

            _clear_memory()
            return ({"samples": out}, latent_start, "\n".join(info_lines))


# ─────────────────────────────────────────────
# Node 3: WAN Curvature Guide
# ─────────────────────────────────────────────

class LRW_WanCurvatureGuide:
    """
    Lightweight latent-distance guide.

    This node never calls metric_tensor() and never creates D x D matrices.
    It only computes O(D) cosine/L2 statistics.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_start": ("LATENT",),
                "latent_end": ("LATENT",),
                "metric": ("METRIC",),
                "n_segments": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 16,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT")
    RETURN_NAMES = ("curvature_info", "mean_curvature", "recommended_keyframes")
    FUNCTION = "compute"
    CATEGORY = "lrw/wan"

    def compute(self, latent_start: dict, latent_end: dict, metric: dict, n_segments: int):
        with torch.no_grad():
            z0_raw = latent_start["samples"]
            z1_raw = latent_end["samples"]

            z0, _shape0, mode0 = _extract_spatial_latent(z0_raw)
            z1, _shape1, mode1 = _extract_spatial_latent(z1_raw)

            if tuple(z0.shape) != tuple(z1.shape):
                raise ValueError(
                    f"latent_start and latent_end must have same processed shape. "
                    f"Got {tuple(z0.shape)} and {tuple(z1.shape)}."
                )

            B = z0.shape[0]
            D = int(z0[0].numel())

            z0_flat = z0.reshape(B, D).float()
            z1_flat = z1.reshape(B, D).float()

            diff = (z1_flat - z0_flat).norm(dim=-1).mean().item()
            z0_norm = z0_flat.norm(dim=-1).mean().item()
            mean_curv = diff / (z0_norm + 1e-8)

            cos_sim = torch.nn.functional.cosine_similarity(z0_flat, z1_flat, dim=-1).mean().item()
            cos_sim = max(-1.0, min(1.0, cos_sim))
            angle_deg = math.degrees(math.acos(cos_sim))

            if angle_deg > 60:
                recommended = 5
                hint = "Large latent angle. Use 4-5 keyframes if memory allows."
            elif angle_deg > 30:
                recommended = 3
                hint = "Medium latent angle. 3 keyframes recommended."
            else:
                recommended = 2
                hint = "Small latent angle. 2 keyframes should be enough."

            # Clamp recommendation for practical 16GB workflows.
            recommended = min(recommended, 3)

            info = "\n".join([
                "LRW WAN Curvature Guide v7_safe",
                f"input modes: start={mode0}, end={mode1}",
                f"processed D: {D}",
                f"latent distance normalized: {mean_curv:.4f}",
                f"SLERP angle: {angle_deg:.2f} deg",
                f"recommended keyframes: {recommended}",
                f"hint: {hint}",
                "dense D x D metric: disabled",
            ])

            del z0_flat, z1_flat
            _clear_memory()
            return (info, float(mean_curv), int(recommended))


# ─────────────────────────────────────────────
# Optional utility node: pick a keyframe from stacked keyframes
# ─────────────────────────────────────────────

class LRW_LatentKeyframePicker:
    """
    Picks one keyframe from LRW_WanGeodesicKeyframes output.

    Input keyframes are stacked as:
    [k0 batch..., k1 batch..., k2 batch...]

    This utility is intentionally tiny and memory-safe.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframe_latents": ("LATENT",),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                }),
                "keyframe_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 15,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "compute"
    CATEGORY = "lrw/wan"

    def compute(self, keyframe_latents: dict, batch_size: int, keyframe_index: int):
        z = keyframe_latents["samples"]
        total = z.shape[0]
        if total % batch_size != 0:
            raise ValueError(f"Total keyframe batch {total} is not divisible by batch_size {batch_size}.")
        n_keyframes = total // batch_size
        idx = max(0, min(int(keyframe_index), n_keyframes - 1))
        picked = z[idx * batch_size:(idx + 1) * batch_size]
        return ({"samples": picked},)


# ─────────────────────────────────────────────
# ComfyUI registration
# ─────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LRW_WanTemporalMetric": LRW_WanTemporalMetric,
    "LRW_WanGeodesicKeyframes": LRW_WanGeodesicKeyframes,
    "LRW_WanCurvatureGuide": LRW_WanCurvatureGuide,
    "LRW_LatentKeyframePicker": LRW_LatentKeyframePicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LRW_WanTemporalMetric": "LRW WAN Temporal Metric (LRW WAN5D Safe v8)",
    "LRW_WanGeodesicKeyframes": "LRW WAN Geodesic Keyframes (LRW First WAN5D Safe v8)",
    "LRW_WanCurvatureGuide": "LRW WAN Curvature Guide (WAN5D Safe v8)",
    "LRW_LatentKeyframePicker": "LRW Latent Keyframe Picker",
}
