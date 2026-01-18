"""
Training module for RLMArena.

Provides the recursive distillation flywheel for continuous model improvement.
"""

from .flywheel import (
    ArenaLogConverter,
    DPOExporter,
    FlywheelWorker,
    PreferencePair,
    PreferenceSource,
    ShadowArenaHook,
    TrainingBatch,
    TrainingBuffer,
    TrainingStatus,
    create_flywheel,
)

__all__ = [
    "ArenaLogConverter",
    "DPOExporter",
    "FlywheelWorker",
    "PreferencePair",
    "PreferenceSource",
    "ShadowArenaHook",
    "TrainingBatch",
    "TrainingBuffer",
    "TrainingStatus",
    "create_flywheel",
]
