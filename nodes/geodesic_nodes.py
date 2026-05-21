"""ComfyUI nodes for geodesic computation and interpolation."""

from __future__ import annotations

import torch


class GeodesicInterpolateNode:
    """
    Interpolates between two latent points along the geodesic
    on the Riemannian manifold defined by the decoder.

    Unlike linear or SLERP interpolation, geodesic interpolation
    follows the true shortest path on the curved latent manifold.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_from": ("LATENT",),
                "latent_to": ("LATENT",),
                "metric": ("METRIC",),
                "n_points": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 64,
                    "step": 1,
                }),
                "n_steps": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 200,
                    "step": 1,
                }),
                "step_size": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_path",)
    FUNCTION = "interpolate"
    CATEGORY = "lrw/geodesic"

    def interpolate(
        self,
        latent_from: dict,
        latent_to: dict,
        metric: dict,
        n_points: int,
        n_steps: int,
        step_size: float,
    ):
        from lrw.geodesic import GeodesicSolver

        z0 = latent_from["samples"]  # (B, C, H, W)
        z1 = latent_to["samples"]

        B = z0.shape[0]
        spatial_shape = z0.shape[1:]

        z0_flat = z0.reshape(B, -1)  # (B, D)
        z1_flat = z1.reshape(B, -1)

        solver = GeodesicSolver(
            metric=metric["metric"],
            n_steps=n_steps,
            step_size=step_size,
        )

        path = solver.interpolate(z0_flat, z1_flat, n_points=n_points)
        # path: (n_points, B, D)

        # Reshape back to spatial latent format and stack as batch
        path_spatial = path.reshape(n_points * B, *spatial_shape)

        return ({"samples": path_spatial},)


class GeodesicDistanceNode:
    """
    Computes the geodesic distance between two latent points.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_from": ("LATENT",),
                "latent_to": ("LATENT",),
                "metric": ("METRIC",),
                "n_steps": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 200,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("distance",)
    FUNCTION = "compute"
    CATEGORY = "lrw/geodesic"

    def compute(self, latent_from: dict, latent_to: dict, metric: dict, n_steps: int):
        from lrw.geodesic import GeodesicSolver

        z0 = latent_from["samples"]
        z1 = latent_to["samples"]

        B = z0.shape[0]
        z0_flat = z0.reshape(B, -1)
        z1_flat = z1.reshape(B, -1)

        solver = GeodesicSolver(metric=metric["metric"], n_steps=n_steps)
        dist = solver.geodesic_distance(z0_flat, z1_flat)  # (B,)

        return (dist.mean().item(),)


class SlerpInterpolateNode:
    """
    Interpolates between two latent points using SLERP.
    Provided as a baseline comparison to geodesic interpolation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_from": ("LATENT",),
                "latent_to": ("LATENT",),
                "n_points": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 64,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_path",)
    FUNCTION = "interpolate"
    CATEGORY = "lrw/geodesic"

    def interpolate(self, latent_from: dict, latent_to: dict, n_points: int):
        from lrw.geodesic import slerp_path

        z0 = latent_from["samples"]
        z1 = latent_to["samples"]

        B = z0.shape[0]
        spatial_shape = z0.shape[1:]

        z0_flat = z0.reshape(B, -1)
        z1_flat = z1.reshape(B, -1)

        path = slerp_path(z0_flat, z1_flat, n_points=n_points)
        # path: (n_points, B, D)

        path_spatial = path.reshape(n_points * B, *spatial_shape)

        return ({"samples": path_spatial},)
