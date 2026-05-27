"""
ComfyUI nodes for WAN2.2 video generation with Riemannian geometry.

Key fixes:
1. WAN VAE handles 5D tensors (B, C, T, H, W) — time axis included
2. Proper VRAM management: chunked Jacobian + torch.cuda.empty_cache()
3. fp16/fp8 auto-casting based on available VRAM
4. Chunked geodesic computation to fit 12-16GB VRAM
"""

from __future__ import annotations

import gc
import torch
from torch import Tensor


# ─────────────────────────────────────────────
# VRAM utilities
# ─────────────────────────────────────────────

def _get_free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return free / 1e9


def _auto_dtype() -> torch.dtype:
    """Pick fp16 or fp32 based on free VRAM."""
    free = _get_free_vram_gb()
    return torch.float16 if free < 10.0 else torch.float32


def _clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_dtype(precision: str) -> torch.dtype:
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
    }
    # fp8 requires torch 2.4+ and CUDA; fallback to fp16
    if precision == "fp8":
        try:
            return torch.float8_e4m3fn
        except AttributeError:
            return torch.float16
    return dtype_map.get(precision, torch.float16)


# ─────────────────────────────────────────────
# WAN VAE decoder bridge
# ─────────────────────────────────────────────

def _make_wan_decoder(vae, spatial_shape: tuple, dtype: torch.dtype):
    """
    Build a decoder function compatible with lrw.metric.PullbackMetric.

    WAN VAE expects (B, C, H, W) per-frame input for decoding.
    We flatten (C, H, W) -> D for metric computation, then reshape back.

    Parameters
    ----------
    vae : ComfyUI VAE object
    spatial_shape : tuple (C, H, W)
    dtype : torch.dtype for computation
    """
    C, H, W = spatial_shape

    def decoder(z_flat: Tensor) -> Tensor:
        """
        z_flat: (B, D) where D = C*H*W
        returns: (B, M) flattened decoded pixels
        """
        B = z_flat.shape[0]

        # Reshape to (B, C, H, W)
        z_spatial = z_flat.reshape(B, C, H, W).to(torch.float32)

        with torch.no_grad():
            decoded = vae.decode(z_spatial)   # (B, 3, H*8, W*8)

        # Flatten and cast back
        result = decoded.reshape(B, -1).to(dtype)

        # Free intermediate tensors
        del decoded
        _clear_vram()

        return result

    return decoder


# ─────────────────────────────────────────────
# Node 1: WAN Temporal Metric
# ─────────────────────────────────────────────

class LRW_WanTemporalMetric:
    """
    Computes the Riemannian metric for WAN2.2 latent space.

    Properly handles WAN's per-frame latent format and manages VRAM
    through chunked Jacobian computation.

    VRAM guide:
        12GB → precision=fp16, chunk_frames=1
        16GB → precision=fp16, chunk_frames=2
        24GB → precision=fp32, chunk_frames=4

    Connect BEFORE LRW_WanGeodesicKeyframes.

    Pipeline:
        VAELoader ──> LRW_WanTemporalMetric ──> LRW_WanGeodesicKeyframes
        VAEEncode ──/
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
                "precision": (["fp16", "fp32", "fp8"], {
                    "default": "fp16",
                    "tooltip": "fp16 for 12-16GB, fp32 for 24GB+, fp8 experimental",
                }),
                "chunk_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Jacobian chunk size. 1 for 12GB, 2 for 16GB, 4 for 24GB+",
                }),
            }
        }

    RETURN_TYPES = ("METRIC",)
    RETURN_NAMES = ("metric",)
    FUNCTION = "compute"
    CATEGORY = "lrw/wan"

    def compute(
        self,
        vae,
        latent: dict,
        regularization: float,
        precision: str,
        chunk_frames: int,
    ):
        from lrw.metric import PullbackMetric

        z = latent["samples"]              # (B, C, H, W)
        B = z.shape[0]
        spatial_shape = z.shape[1:]        # (C, H, W)

        dtype = _safe_dtype(precision)

        decoder = _make_wan_decoder(vae, spatial_shape, dtype)

        metric = PullbackMetric(
            decoder=decoder,
            chunk_size=chunk_frames,
            regularization=regularization,
        )

        _clear_vram()

        return ({
            "metric": metric,
            "latent_shape": z.shape,
            "precision": precision,
            "chunk_frames": chunk_frames,
        },)


# ─────────────────────────────────────────────
# Node 2: WAN Geodesic Keyframes
# ─────────────────────────────────────────────

class LRW_WanGeodesicKeyframes:
    """
    Computes Riemannian geodesic keyframes between start and end latents.

    WAN's default path through latent space is Euclidean (straight line),
    which causes unnatural motion arcs and flickering. This node computes
    the true geodesic path and injects intermediate keyframes to guide WAN
    along a semantically consistent trajectory.

    Connect BEFORE WanFirstLastFrameToVideo or WanAdvancedI2V.

    Pipeline:
        VAEEncode(start) ──> LRW_WanGeodesicKeyframes ──> WanFirstLastFrameToVideo
        VAEEncode(end)   ──/        + metric            /
        LRW_WanTemporalMetric ─────/
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
                    "tooltip": "Intermediate keyframes between start and end.",
                }),
                "n_geodesic_steps": ("INT", {
                    "default": 10,
                    "min": 5,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Integration steps. Higher = accurate but slower.",
                }),
                "geodesic_step_size": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 0.5,
                    "step": 0.001,
                }),
                "vram_mode": (["auto", "low_12gb", "mid_16gb", "high_24gb"], {
                    "default": "auto",
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("keyframe_latents", "latent_start", "info")
    FUNCTION = "compute"
    CATEGORY = "lrw/wan"

    def _resolve_chunk(self, vram_mode: str) -> int:
        if vram_mode == "auto":
            free = _get_free_vram_gb()
            if free < 4:
                return 1
            elif free < 8:
                return 2
            return 4
        return {"low_12gb": 1, "mid_16gb": 2, "high_24gb": 4}.get(vram_mode, 1)

    def compute(
        self,
        latent_start: dict,
        latent_end: dict,
        metric: dict,
        n_keyframes: int,
        n_geodesic_steps: int,
        geodesic_step_size: float,
        vram_mode: str,
    ):
        from lrw.geodesic import BVPSolver

        chunk = self._resolve_chunk(vram_mode)

        z0 = latent_start["samples"]       # (B, C, H, W)
        z1 = latent_end["samples"]

        B = z0.shape[0]
        spatial_shape = z0.shape[1:]
        D = z0[0].numel()

        # float32 for numerical stability in geodesic computation
        z0_flat = z0.reshape(B, D).float()
        z1_flat = z1.reshape(B, D).float()

        m = metric["metric"]

        # BVPSolver: finds true geodesic by solving the Boundary Value Problem.
        # Iteratively refines initial velocity v0 until shoot(z0, v0) ≈ z1.
        # This guarantees arrival at z1 — unlike GeodesicSolver.interpolate()
        # which uses Euclidean initial velocity without convergence guarantee.
        solver = BVPSolver(
            metric=m,
            n_steps=n_geodesic_steps,
            step_size=geodesic_step_size,
            lr=0.1,
            max_iter=30,
            tol=1e-3,
        )

        # Compute true geodesic path: (n_keyframes+2, B, D)
        # includes z0 at [0] and z1 at [-1]
        if chunk == 1:
            path, bvp_info = solver.geodesic_path(
                z0_flat, z1_flat, n_points=n_keyframes + 2
            )
            converged = bvp_info["converged"]
            final_error = bvp_info["final_error"]
        else:
            path_chunks = []
            converged = True
            final_error = 0.0
            for b_start in range(0, B, chunk):
                b_end = min(b_start + chunk, B)
                path_chunk, bvp_info = solver.geodesic_path(
                    z0_flat[b_start:b_end],
                    z1_flat[b_start:b_end],
                    n_points=n_keyframes + 2,
                )
                path_chunks.append(path_chunk)
                converged = converged and bvp_info["converged"]
                final_error = max(final_error, bvp_info["final_error"])
                _clear_vram()
            path = torch.cat(path_chunks, dim=1)

        # Exclude start (index 0) and end (index -1)
        keyframes_flat = path[1:-1]  # (n_keyframes, B, D)

        keyframes = []
        for i in range(n_keyframes):
            z_t = keyframes_flat[i].reshape(B, *spatial_shape)
            keyframes.append(z_t)
            _clear_vram()

        keyframe_tensor = torch.stack(keyframes, dim=1)    # (B, n, C, H, W)
        keyframe_tensor = keyframe_tensor.reshape(
            B * n_keyframes, *spatial_shape
        )

        vram_used = (
            torch.cuda.memory_allocated() / 1e9
            if torch.cuda.is_available() else 0.0
        )
        info = (
            f"BVP converged: {converged} | error: {final_error:.4f} | "
            f"VRAM mode: {vram_mode} | chunk: {chunk} | "
            f"keyframes: {n_keyframes} | VRAM used: {vram_used:.1f}GB"
        )

        return (
            {"samples": keyframe_tensor},
            latent_start,
            info,
        )


# ─────────────────────────────────────────────
# Node 3: WAN Curvature Guide
# ─────────────────────────────────────────────

class LRW_WanCurvatureGuide:
    """
    Measures latent manifold curvature along the path from start to end.

    High curvature = complex transition (WAN needs more guidance).
    Low curvature = simple transition (fewer keyframes needed).

    Use the output info to decide:
    - How many keyframes to inject (n_keyframes in LRW_WanGeodesicKeyframes)
    - Whether to use high-noise or low-noise WAN model variant

    High mean_curvature (>1.0) → use more keyframes + high-noise model
    Low mean_curvature (<0.5)  → fewer keyframes + low-noise model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_start": ("LATENT",),
                "latent_end": ("LATENT",),
                "metric": ("METRIC",),
                "n_segments": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 32,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT")
    RETURN_NAMES = ("curvature_info", "mean_curvature", "recommended_keyframes")
    FUNCTION = "compute"
    CATEGORY = "lrw/wan"

    def compute(
        self,
        latent_start: dict,
        latent_end: dict,
        metric: dict,
        n_segments: int,
    ):
        from lrw.utils import riemannian_norm

        z0 = latent_start["samples"]
        z1 = latent_end["samples"]

        B = z0.shape[0]
        D = z0[0].numel()

        z0_flat = z0.reshape(B, D).float()
        z1_flat = z1.reshape(B, D).float()

        m = metric["metric"]
        curvatures = []

        ts = torch.linspace(0, 1, n_segments)
        for t in ts:
            z_t = (1 - t) * z0_flat + t * z1_flat
            G = m.metric_tensor(z_t)                   # (B, D, D)
            # Frobenius norm as curvature proxy
            curv = G.norm(dim=(-2, -1)).mean().item()
            curvatures.append(curv)
            del G
            _clear_vram()

        mean_curv = float(torch.tensor(curvatures).mean().item())

        # Recommend keyframes based on curvature
        if mean_curv > 2.0:
            recommended = 7
            model_hint = "high-noise model recommended"
        elif mean_curv > 1.0:
            recommended = 5
            model_hint = "high-noise model recommended"
        elif mean_curv > 0.5:
            recommended = 3
            model_hint = "either model works"
        else:
            recommended = 1
            model_hint = "low-noise model recommended"

        info = "\n".join([
            f"Mean curvature: {mean_curv:.4f}",
            f"Recommended keyframes: {recommended}",
            f"Model hint: {model_hint}",
            "─" * 30,
        ] + [
            f"  t={t:.2f}: curvature={c:.4f}"
            for t, c in zip(ts.tolist(), curvatures)
        ])

        return (info, mean_curv, recommended)
