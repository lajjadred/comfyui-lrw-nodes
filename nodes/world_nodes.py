"""ComfyUI nodes for world model temporal transitions."""

from __future__ import annotations

import torch


class LatentTrajectoryNode:
    """
    Generates a temporal trajectory in latent space using
    the Riemannian world model (LatentStateSpace).

    Models motion in latent space as geodesic flow rather
    than simple linear extrapolation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_start": ("LATENT",),
                "metric": ("METRIC",),
                "n_steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                }),
                "dt": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "velocity_scale": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "noise_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_trajectory",)
    FUNCTION = "generate"
    CATEGORY = "lrw/world"

    def generate(
        self,
        latent_start: dict,
        metric: dict,
        n_steps: int,
        dt: float,
        velocity_scale: float,
        noise_scale: float,
        seed: int,
    ):
        from lrw.world import LatentStateSpace

        torch.manual_seed(seed)

        z0 = latent_start["samples"]
        B = z0.shape[0]
        spatial_shape = z0.shape[1:]

        z0_flat = z0.reshape(B, -1)
        v0_flat = torch.randn_like(z0_flat) * velocity_scale

        state_space = LatentStateSpace(
            metric=metric["metric"],
            dt=dt,
            noise_scale=noise_scale,
        )

        states, _ = state_space.rollout(z0_flat, v0_flat, n_steps=n_steps)
        # states: (n_steps+1, B, D)

        states_spatial = states.reshape((n_steps + 1) * B, *spatial_shape)

        return ({"samples": states_spatial},)
