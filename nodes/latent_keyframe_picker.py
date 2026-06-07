"""
Standalone ComfyUI utility node for selecting one latent keyframe.

v2_wan_safe

Fix:
- WAN VAE decode expects a 5D latent: (B, C, T, H, W)
- The previous picker returned 4D latent: (B, C, H, W)
- That caused VAEDecode IndexError: shape[4] out of range

This version can output:
- wan_5d: (B, C, 1, H, W) for WAN VAE decode
- keep: original selected latent shape
- image_4d: forced (B, C, H, W)
"""

from __future__ import annotations

import torch


class LRW_LatentKeyframePicker:
    """
    Pick one keyframe from stacked latent keyframes.

    Expected input from LRW_WanGeodesicKeyframes:
        keyframe_latents["samples"] shape:
            (B * K, C, H, W)
        where:
            B = original batch size
            K = number of keyframes

    For WAN VAEDecode, output_layout should be:
        wan_5d

    This converts:
        (B, C, H, W)
    into:
        (B, C, 1, H, W)
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
                    "tooltip": "Original latent batch size. Usually 1.",
                }),
                "keyframe_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 63,
                    "step": 1,
                    "tooltip": "Which keyframe to pick. 0 = first intermediate keyframe.",
                }),
                "output_layout": (["wan_5d", "keep", "image_4d"], {
                    "default": "wan_5d",
                    "tooltip": "Use wan_5d before WAN VAEDecode.",
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "compute"
    CATEGORY = "lrw/wan"

    def compute(self, keyframe_latents: dict, batch_size: int, keyframe_index: int, output_layout: str):
        if not isinstance(keyframe_latents, dict) or "samples" not in keyframe_latents:
            raise ValueError("keyframe_latents must be a LATENT dict with a 'samples' tensor.")

        z = keyframe_latents["samples"]

        if not torch.is_tensor(z):
            raise ValueError("keyframe_latents['samples'] must be a torch.Tensor.")

        total = z.shape[0]

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        if total % batch_size != 0:
            raise ValueError(
                f"Invalid shape: total latent batch {total} is not divisible by batch_size {batch_size}."
            )

        n_keyframes = total // batch_size

        if n_keyframes < 1:
            raise ValueError("No keyframes found in keyframe_latents.")

        idx = max(0, min(int(keyframe_index), n_keyframes - 1))

        start = idx * batch_size
        end = start + batch_size

        picked = z[start:end].clone()
        before_shape = tuple(picked.shape)

        if output_layout == "wan_5d":
            if picked.ndim == 4:
                # (B, C, H, W) -> (B, C, 1, H, W)
                picked = picked.unsqueeze(2)
            elif picked.ndim == 5:
                pass
            else:
                raise ValueError(
                    f"Cannot convert selected latent to WAN 5D. Got shape {before_shape}."
                )

        elif output_layout == "image_4d":
            if picked.ndim == 5:
                # (B, C, T, H, W) -> first frame only
                picked = picked[:, :, 0, :, :]
            elif picked.ndim == 4:
                pass
            else:
                raise ValueError(
                    f"Cannot convert selected latent to image 4D. Got shape {before_shape}."
                )

        elif output_layout == "keep":
            pass

        else:
            raise ValueError(f"Unknown output_layout: {output_layout}")

        after_shape = tuple(picked.shape)

        info = (
            f"LRW_LatentKeyframePicker v2_wan_safe\n"
            f"total latent batch: {total}\n"
            f"batch_size: {batch_size}\n"
            f"detected keyframes: {n_keyframes}\n"
            f"requested index: {keyframe_index}\n"
            f"selected index: {idx}\n"
            f"output_layout: {output_layout}\n"
            f"before shape: {before_shape}\n"
            f"after shape: {after_shape}\n"
            f"note: use output_layout=wan_5d before WAN VAEDecode"
        )

        return ({"samples": picked}, info)


NODE_CLASS_MAPPINGS = {
    "LRW_LatentKeyframePicker": LRW_LatentKeyframePicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LRW_LatentKeyframePicker": "LRW Latent Keyframe Picker (WAN Safe)",
}
