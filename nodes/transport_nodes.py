"""ComfyUI nodes for parallel transport on Riemannian manifolds."""

from __future__ import annotations

import torch


class ParallelTransportNode:
    """
    Parallel transports a tangent vector (style vector) from one
    latent point to another along the geodesic.

    Use this for style transfer: transport a style direction
    from the source latent to the target latent while preserving
    the geometric meaning of the vector.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_from": ("LATENT",),
                "latent_to": ("LATENT",),
                "latent_vector": ("LATENT",),
                "metric": ("METRIC",),
                "method": (["PoleLadder", "SchildsLadder"],),
                "n_rungs": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("transported_vector",)
    FUNCTION = "transport"
    CATEGORY = "lrw/transport"

    def transport(
        self,
        latent_from: dict,
        latent_to: dict,
        latent_vector: dict,
        metric: dict,
        method: str,
        n_rungs: int,
    ):
        from lrw.transport import SchildsLadder, PoleLadder

        z0 = latent_from["samples"]
        z1 = latent_to["samples"]
        v = latent_vector["samples"]

        B = z0.shape[0]
        spatial_shape = z0.shape[1:]

        z0_flat = z0.reshape(B, -1)
        z1_flat = z1.reshape(B, -1)
        v_flat = v.reshape(B, -1)

        m = metric["metric"]

        if method == "PoleLadder":
            ladder = PoleLadder(metric=m, n_rungs=n_rungs)
        else:
            ladder = SchildsLadder(metric=m, n_rungs=n_rungs)

        v_transported = ladder.transport(z0_flat, z1_flat, v_flat)
        v_spatial = v_transported.reshape(B, *spatial_shape)

        return ({"samples": v_spatial},)
