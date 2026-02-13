"""
ORAM – AI Module
=================
TGV undercarriage anomaly detection, segmentation, and training.
"""

try:
    from ai.detection import (
        InspectionPipeline,
        UndercarriageInspectionModel,
        CrackDetector,
        LeakDetector,
        CorrosionDetector,
        AnomalyType,
        Severity,
        Detection,
        AnalysisResult,
        get_inspection_pipeline,
    )
except ImportError:
    pass

try:
    from ai.training import (
        AgentConfig,
        AgentTrainer,
        MultiAgentOrchestrator,
        FocalLoss,
        get_orchestrator,
        simulate_training,
    )
except ImportError:
    pass

try:
    from ai.preprocessing import (
        LightingNormalizer,
        get_lighting_normalizer,
        build_lighting_augmentations,
    )
except ImportError:
    pass

try:
    from ai.segmentation import (
        SAM2Segmentor,
        SegmentationMask,
        get_segmentor,
    )
except ImportError:
    pass

__all__ = [
    "InspectionPipeline",
    "UndercarriageInspectionModel",
    "CrackDetector",
    "LeakDetector",
    "CorrosionDetector",
    "AnomalyType",
    "Severity",
    "Detection",
    "AnalysisResult",
    "get_inspection_pipeline",
    "AgentConfig",
    "AgentTrainer",
    "MultiAgentOrchestrator",
    "FocalLoss",
    "get_orchestrator",
    "simulate_training",
    "LightingNormalizer",
    "get_lighting_normalizer",
    "build_lighting_augmentations",
    "SAM2Segmentor",
    "SegmentationMask",
    "get_segmentor",
]
