#!/usr/bin/env python3
"""Local integrity and inference smoke test for the Hidden Bites Skill."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    skill_root = script_dir.parent
    command = [
        sys.executable,
        str(script_dir / "recommend.py"),
        "--query",
        "第一次去澳门，雨天晚上两个人，人均150，想吃地道特色，不想等太久",
        "--top-k",
        "3",
        "--compact",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["model_type"] == "StackedPareto"
    assert len(result["recommendations"]) == 3
    assert result["parsed_context"]["urgent"] == 1
    assert result["parsed_context"]["party_size"] == 2
    assert result["parsed_context"]["period"] == "dinner"
    assert result["parsed_context"]["district_source"] == "default"
    assert result["parsed_context"]["budget_source"] == "explicit"
    assert result["parsed_context"]["assumptions"]
    weights = list(result["four_weights"].values())
    assert len(weights) == 4
    assert all(value > 0 for value in weights)
    assert abs(sum(weights) - 1.0) < 1e-5
    base = list(result["base_priorities"].values())
    residual = list(result["context_residual"].values())
    assert abs(sum(base) - 1.0) < 1e-5
    assert abs(sum(residual)) < 1e-5
    assert all("meal_role" in item for item in result["recommendations"])
    assert all(
        "fictional_avg_wait_min" in item
        for item in result["recommendations"]
    )
    assert (skill_root / "assets" / "model_bundle.json").is_file()
    assert (skill_root / "assets" / "macau_restaurants_50.csv").is_file()
    print("Hidden Bites QwenPaw Skill self-check: PASS")


if __name__ == "__main__":
    main()
