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
from .nodes.wan_nodes import (
    LRW_WanGeodesicKeyframes,
    LRW_WanTemporalMetric,
    LRW_WanCurvatureGuide,
    LRW_LatentKeyframePicker,
)
from .nodes.lrw_wan_latent_guide_blend import LRW_WanLatentGuideBlend

NODE_CLASS_MAPPINGS = {
    # WAN2.2 전용 노드 (핵심)
    "LRW_WanGeodesicKeyframes": LRW_WanGeodesicKeyframes,
    "LRW_WanTemporalMetric": LRW_WanTemporalMetric,
    "LRW_WanCurvatureGuide": LRW_WanCurvatureGuide,
    "LRW_LatentKeyframePicker": LRW_LatentKeyframePicker,
    "LRW_WanLatentGuideBlend": LRW_WanLatentGuideBlend,
    # Core bridge nodes
    "LRW_VAEDecoderBridge": LRW_VAEDecoderBridge,
    "LRW_LatentBlend": LRW_LatentBlend,
    "LRW_LatentVectorFromDiff": LRW_LatentVectorFromDiff,
    "LRW_ApplyTransportedVector": LRW_ApplyTransportedVector,
    # Metric nodes
    "LRW_PullbackMetric": PullbackMetricNode,
    "LRW_LatentCurvatureMap": LatentCurvatureMapNode,
    # Geodesic nodes
    "LRW_GeodesicInterpolate": GeodesicInterpolateNode,
    "LRW_GeodesicDistance": GeodesicDistanceNode,
    "LRW_SlerpInterpolate": SlerpInterpolateNode,
    # Transport nodes
    "LRW_ParallelTransport": ParallelTransportNode,
    # World model nodes
    "LRW_LatentTrajectory": LatentTrajectoryNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # WAN2.2 전용 노드
    "LRW_WanGeodesicKeyframes": "WAN Geodesic Keyframes (LRW v7)",
    "LRW_WanTemporalMetric": "WAN Temporal Metric (LRW v7)",
    "LRW_WanCurvatureGuide": "WAN Curvature Guide (LRW v7)",
    "LRW_LatentKeyframePicker": "LRW Latent Keyframe Picker",
    "LRW_WanLatentGuideBlend": "LRW WAN Latent Guide Blend",
    # Core bridge nodes
    "LRW_VAEDecoderBridge": "VAE Decoder Bridge (LRW)",
    "LRW_LatentBlend": "Latent Blend / Frame Select (LRW)",
    "LRW_LatentVectorFromDiff": "Latent Vector from Diff (LRW)",
    "LRW_ApplyTransportedVector": "Apply Transported Vector (LRW)",
    # Metric nodes
    "LRW_PullbackMetric": "Pullback Metric (LRW)",
    "LRW_LatentCurvatureMap": "Latent Curvature Map (LRW)",
    # Geodesic nodes
    "LRW_GeodesicInterpolate": "Geodesic Interpolate (LRW)",
    "LRW_GeodesicDistance": "Geodesic Distance (LRW)",
    "LRW_SlerpInterpolate": "SLERP Interpolate (LRW)",
    # Transport nodes
    "LRW_ParallelTransport": "Parallel Transport (LRW)",
    # World model nodes
    "LRW_LatentTrajectory": "Latent Trajectory (LRW)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
