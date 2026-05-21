"""ComfyUI nodes for Riemannian metric computation."""

from __future__ import annotations

import torch


class PullbackMetricNode:
    """
    Computes the Pullback Riemannian metric G(z) = J^T J
    from a VAE decoder and a latent tensor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "regularization": ("FLOAT", {
                    "default": 1e-5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 1e-6,
                }),
            }
        }

    RETURN_TYPES = ("METRIC", "LATENT")
    RETURN_NAMES = ("metric", "latent")
    FUNCTION = "compute"
    CATEGORY = "lrw/metric"

    def compute(self, latent: dict, vae, regularization: float):
        from lrw.metric import PullbackMetric

        z = latent["samples"]  # (B, C, H, W) in ComfyUI latent format

        # Flatten spatial dims for metric computation
        B = z.shape[0]
        z_flat = z.reshape(B, -1)  # (B, D)

        def decoder(z_in: torch.Tensor) -> torch.Tensor:
            # Reshape back to latent spatial format
            z_spatial = z_in.reshape(B, *z.shape[1:])
            # Decode through VAE
            return vae.decode(z_spatial)

        metric = PullbackMetric(decoder=decoder, regularization=regularization)

        return ({"metric": metric, "latent_shape": z.shape}, latent)


class LatentCurvatureMapNode:
    """
    Computes and visualizes the local volume element sqrt(det G(z))
    as a proxy for latent space curvature.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "metric": ("METRIC",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("curvature_map",)
    FUNCTION = "compute"
    CATEGORY = "lrw/metric"

    def compute(self, latent: dict, metric: dict):
        z = latent["samples"]
        B = z.shape[0]
        z_flat = z.reshape(B, -1)

        m = metric["metric"]
        vol = m.local_volume_element(z_flat)  # (B,)

        # Normalize to [0, 1] for visualization
        vol_min = vol.min()
        vol_max = vol.max()
        vol_norm = (vol - vol_min) / (vol_max - vol_min + 1e-8)

        # Create grayscale image (B, H, W, C)
        img = vol_norm.view(B, 1, 1, 1).expand(B, 64, 64, 3)

        return (img,)
