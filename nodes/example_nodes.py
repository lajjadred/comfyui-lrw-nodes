"""
Helper nodes that bridge ComfyUI core nodes with lrw metric/geodesic nodes.

These nodes are designed to plug directly into standard ComfyUI pipelines:
    CheckpointLoader -> CLIPTextEncode -> KSampler -> VAEDecode
                                              |
                              LRW_VAEDecoder (bridges VAE to lrw metric)
"""

from __future__ import annotations

import torch


class LRW_VAEDecoderBridge:
    """
    Wraps a ComfyUI VAE into an lrw-compatible decoder function
    and computes the Pullback Metric.

    Connect this after CheckpointLoader to enable all lrw geometry nodes.

    Pipeline:
        CheckpointLoader --> LRW_VAEDecoderBridge --> LRW_GeodesicInterpolate
                                                  --> LRW_ParallelTransport
                                                  --> LRW_LatentTrajectory
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
                    "display": "number",
                }),
            }
        }

    RETURN_TYPES = ("METRIC", "LATENT")
    RETURN_NAMES = ("metric", "latent")
    FUNCTION = "build_metric"
    CATEGORY = "lrw/core"

    def build_metric(self, vae, latent: dict, regularization: float):
        from lrw.metric import PullbackMetric

        z = latent["samples"]               # (B, C, H, W)
        B = z.shape[0]
        spatial_shape = z.shape[1:]
        D = z[0].numel()

        def decoder(z_flat: torch.Tensor) -> torch.Tensor:
            # z_flat: (B, D) -> decode via VAE -> (B, C, H, W) -> flatten
            z_spatial = z_flat.reshape(B, *spatial_shape)
            decoded = vae.decode(z_spatial)           # (B, C, H, W) pixel space
            return decoded.reshape(B, -1)             # (B, C*H*W)

        metric = PullbackMetric(decoder=decoder, regularization=regularization)
        return ({"metric": metric, "latent_shape": z.shape}, latent)


class LRW_LatentBlend:
    """
    Blends two latent tensors after geodesic interpolation for use with KSampler.

    Takes the geodesic path output and selects a single frame for downstream use.

    Pipeline:
        LRW_GeodesicInterpolate --> LRW_LatentBlend --> KSampler
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("LATENT",),
                "frame_index": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 63,
                    "step": 1,
                }),
                "total_frames": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 64,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "select_frame"
    CATEGORY = "lrw/core"

    def select_frame(self, latent_path: dict, frame_index: int, total_frames: int):
        samples = latent_path["samples"]    # (n_points * B, C, H, W)
        B = samples.shape[0] // total_frames
        spatial_shape = samples.shape[1:]

        idx = min(frame_index, total_frames - 1)
        frame = samples[idx * B:(idx + 1) * B]  # (B, C, H, W)

        return ({"samples": frame},)


class LRW_LatentVectorFromDiff:
    """
    Creates a tangent vector (style direction) from two latent tensors.

    Use this to define a style direction in latent space, then transport
    it to a new location using LRW_ParallelTransport.

    Pipeline:
        VAEEncode(style_A) --> LRW_LatentVectorFromDiff --> LRW_ParallelTransport
        VAEEncode(style_B) -/
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_from": ("LATENT",),
                "latent_to": ("LATENT",),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("vector",)
    FUNCTION = "compute"
    CATEGORY = "lrw/core"

    def compute(self, latent_from: dict, latent_to: dict, scale: float):
        z0 = latent_from["samples"]
        z1 = latent_to["samples"]
        vector = (z1 - z0) * scale
        return ({"samples": vector},)


class LRW_ApplyTransportedVector:
    """
    Applies a transported tangent vector to a latent point.

    After parallel transport, use this to add the transported
    style direction to a target latent.

    Pipeline:
        LRW_ParallelTransport --> LRW_ApplyTransportedVector --> KSampler
        VAEEncode(target)     -/
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vector": ("LATENT",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply"
    CATEGORY = "lrw/core"

    def apply(self, latent: dict, vector: dict, strength: float):
        z = latent["samples"]
        v = vector["samples"]
        result = z + strength * v
        return ({"samples": result},)
