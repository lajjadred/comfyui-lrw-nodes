"""
Practical bridge node for WAN + LRW workflows.

Goal:
- Keep direct first->last continuity from WanFirstLastFrameToVideo
- Inject a controlled amount of LRW geodesic influence into the actual video latent path
- Stay RAM-safe and simple

Behavior:
1. Takes the FLF video latent: (B, C, T, H, W)
2. Takes stacked geodesic keyframe latents from LRW_WanGeodesicKeyframes:
   typically (B*K, C, 1, H, W) or (B*K, C, H, W)
3. Selects one keyframe by index
4. Expands that keyframe over the video time axis T
5. Blends it into the FLF latent with a time weighting schedule

This means the LRW result has a real effect on the latent that goes into KSampler.
"""

from __future__ import annotations
import torch


class LRW_WanLatentGuideBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "keyframe_latents": ("LATENT",),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Original batch size used by LRW_WanGeodesicKeyframes."
                }),
                "keyframe_index": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 63,
                    "step": 1,
                    "tooltip": "Which LRW keyframe to use. For 5 keyframes, 2 is the middle."
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How strongly the selected geodesic keyframe influences the video latent."
                }),
                "time_schedule": (["all_frames", "middle_focus", "late_focus"], {
                    "default": "middle_focus",
                    "tooltip": "How the guide weight is distributed over time."
                }),
                "normalize_guide": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Match guide latent norm roughly to video latent norm."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "compute"
    CATEGORY = "lrw/wan"

    def _to_5d(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 5:
            return z
        if z.ndim == 4:
            return z.unsqueeze(2)
        raise ValueError(f"Expected 4D or 5D latent, got shape {tuple(z.shape)}")

    def _time_weights(self, T: int, device, dtype, mode: str) -> torch.Tensor:
        if T <= 1:
            return torch.ones(1, device=device, dtype=dtype)

        t = torch.linspace(0.0, 1.0, steps=T, device=device, dtype=dtype)

        if mode == "all_frames":
            w = torch.ones_like(t)
        elif mode == "middle_focus":
            # triangular hump peaking in the middle
            w = 1.0 - torch.abs(t * 2.0 - 1.0)
            w = 0.25 + 0.75 * w
        elif mode == "late_focus":
            # starts lighter, increases toward the end
            w = 0.25 + 0.75 * t
        else:
            w = torch.ones_like(t)

        return w

    def compute(
        self,
        video_latent: dict,
        keyframe_latents: dict,
        batch_size: int,
        keyframe_index: int,
        blend_strength: float,
        time_schedule: str,
        normalize_guide: bool,
    ):
        if "samples" not in video_latent or "samples" not in keyframe_latents:
            raise ValueError("Both inputs must be LATENT dictionaries containing 'samples'.")

        z_video = self._to_5d(video_latent["samples"]).clone()
        z_keys = self._to_5d(keyframe_latents["samples"])

        if not torch.is_tensor(z_video) or not torch.is_tensor(z_keys):
            raise ValueError("LATENT 'samples' values must be tensors.")

        total = z_keys.shape[0]
        if batch_size < 1 or total % batch_size != 0:
            raise ValueError(
                f"Invalid stacked keyframe batch: total={total}, batch_size={batch_size}"
            )

        n_keyframes = total // batch_size
        idx = max(0, min(int(keyframe_index), n_keyframes - 1))

        start = idx * batch_size
        end = start + batch_size
        z_guide = z_keys[start:end]  # (B, C, 1, H, W) expected

        Bv, Cv, Tv, Hv, Wv = z_video.shape
        Bg, Cg, Tg, Hg, Wg = z_guide.shape

        if Bg != Bv:
            raise ValueError(f"Batch mismatch: video={Bv}, guide={Bg}")
        if Cg != Cv or Hg != Hv or Wg != Wv:
            raise ValueError(
                f"Shape mismatch: video={tuple(z_video.shape)}, guide={tuple(z_guide.shape)}"
            )

        # Expand guide over time
        if Tg == 1:
            z_guide = z_guide.expand(Bv, Cv, Tv, Hv, Wv)
        elif Tg == Tv:
            pass
        else:
            # repeat or slice conservatively
            if Tg < Tv:
                reps = (Tv + Tg - 1) // Tg
                z_guide = z_guide.repeat(1, 1, reps, 1, 1)[:, :, :Tv, :, :]
            else:
                z_guide = z_guide[:, :, :Tv, :, :]

        z_video = z_video.float()
        z_guide = z_guide.float()

        if normalize_guide:
            vid_norm = z_video.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-6)
            gui_norm = z_guide.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-6)
            scale = (vid_norm / gui_norm).view(Bv, 1, 1, 1, 1)
            z_guide = z_guide * scale

        wt = self._time_weights(Tv, z_video.device, z_video.dtype, time_schedule).view(1, 1, Tv, 1, 1)
        alpha = float(blend_strength)

        z_out = (1.0 - alpha * wt) * z_video + (alpha * wt) * z_guide

        info = (
            f"LRW_WanLatentGuideBlend\n"
            f"video latent shape: {tuple(z_video.shape)}\n"
            f"guide latent shape: {tuple(z_guide.shape)}\n"
            f"detected keyframes: {n_keyframes}\n"
            f"selected keyframe index: {idx}\n"
            f"blend_strength: {alpha:.3f}\n"
            f"time_schedule: {time_schedule}\n"
            f"normalize_guide: {normalize_guide}\n"
            f"note: output is a real latent_image replacement for KSampler"
        )

        return ({"samples": z_out.to(dtype=video_latent["samples"].dtype)}, info)


NODE_CLASS_MAPPINGS = {
    "LRW_WanLatentGuideBlend": LRW_WanLatentGuideBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LRW_WanLatentGuideBlend": "LRW WAN Latent Guide Blend",
}
