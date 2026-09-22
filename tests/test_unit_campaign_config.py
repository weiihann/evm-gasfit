"""Validation boundaries for campaign-analysis configuration."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from _data_synth import base_config
from evm_gasfit.config import Config
from evm_gasfit.provenance import planned_models_payload


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("root", "anchor_rate", math.inf),
        ("qualification", "max_condition_number", math.inf),
        ("qualification", "max_relative_uncertainty", math.nan),
        ("qualification", "max_holdout_error", math.inf),
        ("scenario", "anchor_rate", math.inf),
        ("scenario", "margin_pct", math.nan),
    ],
)
def test_pricing_and_qualification_limits_must_be_finite(
    section: str, field: str, value: float
) -> None:
    config = base_config()
    if section == "root":
        config[field] = value
    elif section == "qualification":
        config["qualification"] = {field: value}
    else:
        config["pricing_scenarios"] = [
            {"name": "calibration", "anchor_rate": 1.0e8, field: value}
        ]

    with pytest.raises(ValidationError, match="finite"):
        Config.model_validate(config)


def test_status_payload_keeps_model_by_identity_and_nulls_nonfinite_evidence() -> None:
    qualification = pd.DataFrame(
        [
            {
                "source_label": "models.custom[0]",
                "test_name": "test_keccak",
                "target_opcode": "KECCAK256",
                "client_name": "geth",
                "operand_size": "4096",
                "status": "inconclusive",
                "reasons": "holdout unavailable",
                "adjusted_estimate_status": "inconclusive",
                "condition_number": math.inf,
                "holdout_point_error": math.nan,
            }
        ]
    )
    spec = SimpleNamespace(source_label="models.custom[0]", model_by=["operand_size"])

    payload = planned_models_payload(qualification, [spec])

    assert payload[0]["group_values"] == {"operand_size": "4096"}
    assert payload[0]["condition_number"] is None
    assert payload[0]["holdout_point_error"] is None
