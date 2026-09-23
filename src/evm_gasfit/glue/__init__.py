"""Glue-opcode estimation, detection, and adjustment."""

from __future__ import annotations

from .adjust import compute_glue_adjustment
from .detect import (
    compute_detection_coverage,
    compute_glue_opcodes_by_test,
    detect_missing_glue,
)
from .estimate import GlueEstimateOutput, compute_driver_support, estimate_glue
from .lane import (
    CAMPAIGN_ROLE_COLUMN,
    ROLE_CALIBRATION,
    ROLE_TARGET,
    calibration_lane,
    split_campaign_lanes,
    target_lane,
)
from .required import (
    CANONICAL_TO_MEMBERS,
    CYCLE_GLUE_OPCODES,
    MEMBER_TO_CANONICAL,
    MIXED_A_GLUE_OPCODES,
    MIXED_B_GLUE_OPCODES,
    PRICED_GLUE_OPCODES,
    PRICED_GLUE_SPECS,
    PURE_GLUE_OPCODES,
    SPEC_BY_NAME,
    GlueOpcodeSpec,
    validate_inputs,
)

__all__ = [
    "CAMPAIGN_ROLE_COLUMN",
    "CANONICAL_TO_MEMBERS",
    "CYCLE_GLUE_OPCODES",
    "MEMBER_TO_CANONICAL",
    "MIXED_A_GLUE_OPCODES",
    "MIXED_B_GLUE_OPCODES",
    "PRICED_GLUE_OPCODES",
    "PRICED_GLUE_SPECS",
    "PURE_GLUE_OPCODES",
    "ROLE_CALIBRATION",
    "ROLE_TARGET",
    "SPEC_BY_NAME",
    "GlueEstimateOutput",
    "GlueOpcodeSpec",
    "calibration_lane",
    "compute_detection_coverage",
    "compute_driver_support",
    "compute_glue_adjustment",
    "compute_glue_opcodes_by_test",
    "detect_missing_glue",
    "estimate_glue",
    "split_campaign_lanes",
    "target_lane",
    "validate_inputs",
]
