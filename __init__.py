"""ComfyUI LRW Nodes — Riemannian geometry nodes for ComfyUI."""

from .nodes.metric_nodes import PullbackMetricNode, LatentCurvatureMapNode
from .nodes.geodesic_nodes import (
    GeodesicInterpolateNode,
    GeodesicDistanceNode,
    SlerpInterpolateNode,
)
from .nodes.transport_nodes import ParallelTransportNode
from .nodes.world_nodes import LatentTrajectoryNode
from .nodes.example_nodes import (
    LRW_VAEDecoderBridge,
    LRW_LatentBlend,
    LRW_LatentVectorFromDiff,
    LRW_ApplyTransportedVector,
)

NODE_CLASS_MAPPINGS = {
    "LRW_VAEDecoderBridge": LRW_VAEDecoderBridge,
    "LRW_LatentBlend": LRW_LatentBlend,
    "LRW_LatentVectorFromDiff": LRW_LatentVectorFromDiff,
    "LRW_ApplyTransportedVector": LRW_ApplyTransportedVector,
    "LRW_PullbackMetric": PullbackMetricNode,
    "LRW_LatentCurvatureMap": LatentCurvatureMapNode,
    "LRW_GeodesicInterpolate": GeodesicInterpolateNode,
    "LRW_GeodesicDistance": GeodesicDistanceNode,
    "LRW_SlerpInterpolate": SlerpInterpolateNode,
    "LRW_ParallelTransport": ParallelTransportNode,
    "LRW_LatentTrajectory": LatentTrajectoryNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LRW_VAEDecoderBridge": "VAE Decoder Bridge (LRW)",
    "LRW_LatentBlend": "Latent Blend / Frame Select (LRW)",
    "LRW_LatentVectorFromDiff": "Latent Vector from Diff (LRW)",
    "LRW_ApplyTransportedVector": "Apply Transported Vector (LRW)",
    "LRW_PullbackMetric": "Pullback Metric (LRW)",
    "LRW_LatentCurvatureMap": "Latent Curvature Map (LRW)",
    "LRW_GeodesicInterpolate": "Geodesic Interpolate (LRW)",
    "LRW_GeodesicDistance": "Geodesic Distance (LRW)",
    "LRW_SlerpInterpolate": "SLERP Interpolate (LRW)",
    "LRW_ParallelTransport": "Parallel Transport (LRW)",
    "LRW_LatentTrajectory": "Latent Trajectory (LRW)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
