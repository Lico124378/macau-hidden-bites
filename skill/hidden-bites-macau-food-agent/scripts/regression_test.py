#!/usr/bin/env python3
"""Regression checks for parsing, grounding, and UTF-8 JSON output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "recommend.py"


def run_query(query: str, top_k: int = 5) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--query",
            query,
            "--top-k",
            str(top_k),
            "--compact",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    return json.loads(completed.stdout)


def main() -> None:
    rainy_dinner = run_query(
        "第一次去澳门，雨天晚上两个人，人均150，"
        "想吃地道特色，不想等太久"
    )
    context = rainy_dinner["parsed_context"]
    assert context["tourist"] == 1
    assert context["rainy"] == 1
    assert context["urgent"] == 1
    assert context["party_size"] == 2
    assert context["group"] == 0
    assert context["period"] == "dinner"
    assert context["period_source"] == "explicit"
    assert context["district"] == "新马路/大三巴"
    assert context["district_source"] == "default"
    assert context["budget"] == 150.0
    assert context["budget_source"] == "explicit"
    assert any("默认" in item for item in context["assumptions"])

    explicit_location = run_query(
        "今晚在官也街附近，三个人，人均100，不想排队，想吃清淡一点"
    )
    context = explicit_location["parsed_context"]
    assert context["district"] == "氹仔旧城区"
    assert context["district_source"] == "explicit"
    assert context["party_size"] == 3
    assert context["group"] == 1
    assert context["urgent"] == 1
    assert context["budget"] == 100.0

    negation = run_query("两个人晚餐，不吃辣，也不要海鲜，人均200")
    assert negation["ok"] is True
    assert negation["parsed_context"]["party_size"] == 2
    assert {
        (item["preference"], item["direction"])
        for item in negation["parsed_context"]["recognized_preferences"]
    } >= {("辣味", "avoid"), ("海鲜", "avoid")}

    allergy = run_query("海鲜过敏，想在路环吃晚餐")
    assert allergy["parsed_context"]["district"] == "路环"
    assert allergy["parsed_context"]["limitations"]
    assert any(
        "过敏原" in item
        for item in allergy["answer_constraints"]["required_disclosures"]
    )

    for result in (rainy_dinner, explicit_location, negation, allergy):
        assert result["ok"] is True
        assert abs(sum(result["four_weights"].values()) - 1.0) < 1e-5
        assert abs(sum(result["base_priorities"].values()) - 1.0) < 1e-5
        assert abs(sum(result["context_residual"].values())) < 1e-5
        assert result["recommendations"]
        for item in result["recommendations"]:
            assert item["meal_role"] in {
                "正餐候选",
                "特色小食",
                "甜品或加餐",
            }
            assert item["fictional_avg_wait_min"] >= 0

    print("Hidden Bites regression tests: PASS")


if __name__ == "__main__":
    main()
