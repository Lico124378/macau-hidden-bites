#!/usr/bin/env python3
"""Portable Hidden Bites v23 inference for QwenPaw.

This script intentionally uses NumPy instead of PyTorch. It reproduces the
evaluation-time forward pass of the two component models stored in the final
StackedPareto bundle, so QwenPaw does not need the training notebook or CUDA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "Missing dependency: numpy. Install it with: python -m pip install numpy>=1.26"
    ) from error


FEATURE_NAMES = ["taste_match", "quality", "convenience", "value"]
FEATURE_NAMES_ZH = ["口味匹配", "品质稳定", "便利程度", "性价比"]
USER_DIM = 12
CONTEXT_DIM = 10
HIDDEN_DIM = 32
HEAD_COUNT = 4
HEAD_DIM = HIDDEN_DIM // HEAD_COUNT
DEFAULT_DISTRICT = "新马路/大三巴"
DEFAULT_BUDGET_MOP = 150.0
DISTRICT_ALIASES = {
    "新马路/大三巴": ("新马路", "大三巴", "议事亭", "板樟堂"),
    "南湾/西湾": ("南湾", "西湾", "西湾湖"),
    "妈阁/下环": ("妈阁", "下环"),
    "氹仔旧城区": ("氹仔旧城区", "氹仔", "官也街", "龙环葡韵"),
    "路氹": ("路氹", "威尼斯人", "巴黎人", "伦敦人", "永利皇宫"),
    "路环": ("路环", "黑沙海滩"),
    "高士德/红街市": ("高士德", "红街市"),
    "黑沙环/祐汉": ("黑沙环", "祐汉", "佑汉"),
}


def configure_utf8_stdout() -> None:
    """Make JSON output deterministic on Windows and redirected consoles."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="strict")


def stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def linear(value: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return value @ weight.T + bias


def layer_norm(
    value: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    mean = np.mean(value, axis=-1, keepdims=True)
    variance = np.mean((value - mean) ** 2, axis=-1, keepdims=True)
    normalized = (value - mean) / np.sqrt(variance + epsilon)
    return normalized * weight + bias


def state_arrays(mapping: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value, dtype=np.float32)
        for name, value in mapping.items()
    }


def encode_token(
    value: np.ndarray,
    state: dict[str, np.ndarray],
    prefix: str,
) -> np.ndarray:
    encoded = linear(
        value,
        state[f"{prefix}.0.weight"],
        state[f"{prefix}.0.bias"],
    )
    encoded = np.maximum(encoded, 0.0)
    return layer_norm(
        encoded,
        state[f"{prefix}.2.weight"],
        state[f"{prefix}.2.bias"],
    )


def transformer_layer(
    tokens: np.ndarray,
    state: dict[str, np.ndarray],
) -> np.ndarray:
    prefix = "transformer.layers.0"
    normalized = layer_norm(
        tokens,
        state[f"{prefix}.norm1.weight"],
        state[f"{prefix}.norm1.bias"],
    )
    qkv = linear(
        normalized,
        state[f"{prefix}.self_attn.in_proj_weight"],
        state[f"{prefix}.self_attn.in_proj_bias"],
    )
    query, key, value = np.split(qkv, 3, axis=-1)

    def split_heads(item: np.ndarray) -> np.ndarray:
        return item.reshape(item.shape[0], HEAD_COUNT, HEAD_DIM).transpose(1, 0, 2)

    query_heads = split_heads(query)
    key_heads = split_heads(key)
    value_heads = split_heads(value)
    attention_logits = (
        query_heads @ key_heads.transpose(0, 2, 1)
    ) / math.sqrt(HEAD_DIM)
    attention = stable_softmax(attention_logits, axis=-1)
    attended = attention @ value_heads
    attended = attended.transpose(1, 0, 2).reshape(tokens.shape[0], HIDDEN_DIM)
    attended = linear(
        attended,
        state[f"{prefix}.self_attn.out_proj.weight"],
        state[f"{prefix}.self_attn.out_proj.bias"],
    )
    tokens = tokens + attended

    normalized = layer_norm(
        tokens,
        state[f"{prefix}.norm2.weight"],
        state[f"{prefix}.norm2.bias"],
    )
    hidden = linear(
        normalized,
        state[f"{prefix}.linear1.weight"],
        state[f"{prefix}.linear1.bias"],
    )
    hidden = np.maximum(hidden, 0.0)
    feed_forward = linear(
        hidden,
        state[f"{prefix}.linear2.weight"],
        state[f"{prefix}.linear2.bias"],
    )
    return tokens + feed_forward


def component_weights(
    user: np.ndarray,
    context: np.ndarray,
    raw_state: dict[str, Any],
) -> np.ndarray:
    state = state_arrays(raw_state)
    priorities = np.maximum(user[-4:], 1e-6)
    priorities = priorities / np.sum(priorities)
    user_token = encode_token(user, state, "user_encoder")
    context_token = encode_token(context, state, "context_encoder")
    tokens = transformer_layer(np.stack([user_token, context_token]), state)
    encoded = np.mean(tokens, axis=0)
    hidden = linear(
        encoded,
        state["weight_head.0.weight"],
        state["weight_head.0.bias"],
    )
    hidden = np.maximum(hidden, 0.0)
    residual = linear(
        hidden,
        state["weight_head.2.weight"],
        state["weight_head.2.bias"],
    )
    residual_scale = 2.0 * float(sigmoid(state["context_residual_scale_raw"]))
    residual = residual_scale * np.tanh(residual)
    return stable_softmax(np.log(priorities) + residual).astype(np.float32)


def load_restaurants(path: Path) -> list[dict[str, Any]]:
    numeric_columns = {
        "avg_price_mop",
        "taste_quality",
        "quality_consistency",
        "avg_wait_min",
        "popularity",
        "tourist_friendly",
        "local_authenticity",
        "spicy",
        "sweet",
        "rich",
        "light",
        "seafood",
        "vegetarian_friendly",
        "open_breakfast",
        "open_late",
        "latitude",
        "longitude",
        "base_value_score",
        "base_convenience_score",
    }
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for name in numeric_columns:
                row[name] = float(row[name])
            rows.append(row)
    if len(rows) != 50:
        raise ValueError(f"Expected 50 restaurants, found {len(rows)}")
    return rows


def parse_budget(query: str) -> tuple[float, bool]:
    patterns = [
        r"人均\s*(\d+(?:\.\d+)?)",
        r"预算\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:元|澳门元|mop)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)), True
    return DEFAULT_BUDGET_MOP, False


def chinese_number_to_int(value: str) -> int | None:
    digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value in digits:
        return digits[value]
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def parse_party_size(query: str) -> tuple[int | None, bool]:
    digit_match = re.search(r"(\d{1,2})\s*(?:个)?(?:人|位)", query)
    if digit_match:
        return int(digit_match.group(1)), True
    chinese_match = re.search(
        r"([一二两三四五六七八九十]{1,3})\s*(?:个)?(?:人|位)",
        query,
    )
    if chinese_match:
        size = chinese_number_to_int(chinese_match.group(1))
        if size is not None:
            return size, True
    return None, False


def match_district(
    query: str,
    districts: list[str],
) -> tuple[str, bool]:
    for district in districts:
        aliases = DISTRICT_ALIASES.get(district, (district,))
        if any(alias in query for alias in aliases):
            return district, True
    return DEFAULT_DISTRICT, False


def meal_role(cuisine: str) -> str:
    if "甜品" in cuisine or "烘焙" in cuisine:
        return "甜品或加餐"
    if "小食" in cuisine:
        return "特色小食"
    return "正餐候选"


def parse_query(
    query: str,
    restaurants: list[dict[str, Any]],
) -> dict[str, Any]:
    districts = list(dict.fromkeys(row["district"] for row in restaurants))
    district, district_explicit = match_district(query, districts)
    if any(word in query for word in ("早餐", "早上", "早晨")):
        period = "breakfast"
        period_explicit = True
    elif any(word in query for word in ("午餐", "中午")):
        period = "lunch"
        period_explicit = True
    elif any(word in query for word in ("宵夜", "夜宵", "深夜", "凌晨")):
        period = "late_night"
        period_explicit = True
    elif any(word in query for word in ("晚餐", "晚饭", "晚上", "今晚", "夜晚")):
        period = "dinner"
        period_explicit = True
    else:
        period = "dinner"
        period_explicit = False

    budget, budget_explicit = parse_budget(query)
    party_size, party_size_explicit = parse_party_size(query)
    tourist = int(
        any(
            word in query
            for word in ("游客", "旅游", "第一次", "首次", "初次", "头一次", "澳门特色")
        )
    )
    urgent = int(
        any(
            word in query
            for word in (
                "赶时间",
                "快一点",
                "快一些",
                "快些",
                "附近",
                "马上",
                "尽快",
                "时间紧",
                "来不及",
                "不想等",
                "不想排队",
                "不要排队",
                "少排队",
                "排队少",
                "等太久",
                "等很久",
            )
        )
    )
    rainy = int(any(word in query for word in ("下雨", "雨天", "雨夜", "落雨")))
    group = int(
        (party_size is not None and party_size >= 3)
        or any(word in query for word in ("多人", "聚餐", "一家", "团体", "朋友们"))
    )
    holiday = int(any(word in query for word in ("假期", "节日", "周末")))
    crowd = 0.75 if any(word in query for word in ("热闹", "高峰", "周末")) else 0.45

    preference = np.full(6, 0.5, dtype=np.float32)
    keyword_preferences = {
        0: ("辣", "川菜", "火锅"),
        1: ("甜", "甜品", "蛋挞"),
        2: ("浓郁", "重口", "烧腊"),
        3: ("清淡", "健康", "粥"),
        4: ("海鲜", "鱼", "蟹"),
        5: ("素食", "不吃肉", "蔬菜"),
    }
    negative_preferences = {
        0: ("不吃辣", "不要辣", "不能吃辣", "不辣", "怕辣"),
        1: ("不吃甜", "不要甜", "不能吃甜", "少甜"),
        2: ("不要油腻", "不油腻", "清淡"),
        4: ("不吃海鲜", "不要海鲜", "海鲜过敏"),
    }
    preference_labels = ("辣味", "甜味", "浓郁", "清淡", "海鲜", "素食")
    recognized_preferences = []
    for index, words in keyword_preferences.items():
        if any(
            word in query
            for word in negative_preferences.get(index, ())
        ):
            preference[index] = 0.1
            recognized_preferences.append(
                {
                    "preference": preference_labels[index],
                    "direction": "avoid",
                }
            )
        elif any(word in query for word in words):
            preference[index] = 0.9
            recognized_preferences.append(
                {
                    "preference": preference_labels[index],
                    "direction": "prefer",
                }
            )

    priorities = np.asarray([0.34, 0.25, 0.20, 0.21], dtype=np.float32)
    if any(word in query for word in ("好吃", "特色", "地道", "正宗")):
        priorities[0] += 0.18
    if any(word in query for word in ("稳定", "高品质", "不踩雷")):
        priorities[1] += 0.15
    if urgent:
        priorities[2] += 0.28
    if any(word in query for word in ("便宜", "性价比", "实惠")) or budget <= 100:
        priorities[3] += 0.22
    priorities = priorities / np.sum(priorities)

    user = np.concatenate(
        [
            preference,
            np.asarray(
                [
                    np.clip(budget / 400.0, 0.0, 1.0),
                    0.90 if tourist else 0.72,
                ],
                dtype=np.float32,
            ),
            priorities,
        ]
    ).astype(np.float32)
    periods = ["breakfast", "lunch", "dinner", "late_night"]
    context = np.asarray(
        [int(period == name) for name in periods]
        + [rainy, urgent, group, tourist, holiday, crowd],
        dtype=np.float32,
    )
    assumptions = []
    if not district_explicit:
        assumptions.append(f"未指定区域，默认按{DEFAULT_DISTRICT}生成候选")
    if not period_explicit:
        assumptions.append("未指定用餐时段，默认按晚餐处理")
    if not budget_explicit:
        assumptions.append(f"未指定人均预算，默认按{DEFAULT_BUDGET_MOP:.0f}澳门元处理")
    limitations = []
    if "过敏" in query:
        limitations.append("虚构餐厅数据不含过敏原验证，不能作食品安全判断")
    if urgent:
        limitations.append("等待时间是虚构数据中的模拟均值，不代表实时排队情况")
    return {
        "district": district,
        "district_source": "explicit" if district_explicit else "default",
        "period": period,
        "period_source": "explicit" if period_explicit else "default",
        "budget": budget,
        "budget_source": "explicit" if budget_explicit else "default",
        "party_size": party_size,
        "party_size_source": "explicit" if party_size_explicit else "not_provided",
        "tourist": tourist,
        "urgent": urgent,
        "rainy": rainy,
        "group": group,
        "holiday": holiday,
        "crowd": crowd,
        "recognized_preferences": recognized_preferences,
        "assumptions": assumptions,
        "limitations": limitations,
        "user": user,
        "context": context,
    }


def haversine(
    latitude_1: float,
    longitude_1: float,
    latitude_2: np.ndarray,
    longitude_2: np.ndarray,
) -> np.ndarray:
    radius = 6371.0
    latitude_1 = np.radians(latitude_1)
    longitude_1 = np.radians(longitude_1)
    latitude_2 = np.radians(latitude_2)
    longitude_2 = np.radians(longitude_2)
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = longitude_2 - longitude_1
    value = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude_1)
        * np.cos(latitude_2)
        * np.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * radius * np.arcsin(np.sqrt(value))


def column(
    restaurants: list[dict[str, Any]],
    name: str,
) -> np.ndarray:
    return np.asarray([row[name] for row in restaurants], dtype=np.float64)


def build_restaurant_features(
    restaurants: list[dict[str, Any]],
    parsed: dict[str, Any],
) -> np.ndarray:
    flavor_columns = [
        "spicy",
        "sweet",
        "rich",
        "light",
        "seafood",
        "vegetarian_friendly",
    ]
    flavors = np.stack(
        [column(restaurants, name) for name in flavor_columns],
        axis=1,
    )
    preference = parsed["user"][:6].astype(np.float64)
    flavor_match = 1.0 - np.mean(np.abs(flavors - preference[None, :]), axis=1)
    authenticity_match = 1.0 - np.abs(
        column(restaurants, "local_authenticity") - parsed["user"][7]
    )
    taste = column(restaurants, "taste_quality")
    taste_match = np.clip(
        0.48 * taste + 0.40 * flavor_match + 0.12 * authenticity_match,
        0.0,
        1.0,
    )
    quality = np.clip(
        0.58 * column(restaurants, "quality_consistency") + 0.42 * taste,
        0.0,
        1.0,
    )

    district_rows = [
        row for row in restaurants if row["district"] == parsed["district"]
    ]
    if not district_rows:
        district_rows = restaurants
    current_latitude = float(
        np.mean([row["latitude"] for row in district_rows])
    )
    current_longitude = float(
        np.mean([row["longitude"] for row in district_rows])
    )
    distance = haversine(
        current_latitude,
        current_longitude,
        column(restaurants, "latitude"),
        column(restaurants, "longitude"),
    )
    distance_score = np.exp(-distance / 2.6)
    wait_score = np.exp(
        -column(restaurants, "avg_wait_min")
        * (1.0 + 0.35 * parsed["context"][9])
        / 34.0
    )
    availability = np.ones(len(restaurants))
    if parsed["period"] == "breakfast":
        availability = np.where(
            column(restaurants, "open_breakfast") > 0,
            1.0,
            0.35,
        )
    elif parsed["period"] == "late_night":
        availability = np.where(
            column(restaurants, "open_late") > 0,
            1.0,
            0.25,
        )
    convenience = np.clip(
        0.57 * distance_score + 0.28 * wait_score + 0.15 * availability,
        0.0,
        1.0,
    )
    budget = parsed["budget"] * (1.35 if parsed["group"] else 1.0)
    scale = 0.32 * budget + 12.0
    affordability = 1.0 / (
        1.0 + np.exp((column(restaurants, "avg_price_mop") - budget) / scale)
    )
    value = np.clip(0.72 * affordability + 0.28 * taste, 0.0, 1.0)
    return np.stack([taste_match, quality, convenience, value], axis=1).astype(
        np.float32
    )


def make_reason(
    restaurant: dict[str, Any],
    features: np.ndarray,
    weights: np.ndarray,
) -> tuple[str, list[dict[str, float | str]]]:
    contribution = features * weights
    order = np.argsort(contribution)[::-1]
    details = [
        {
            "dimension": FEATURE_NAMES[index],
            "dimension_zh": FEATURE_NAMES_ZH[index],
            "feature_value": round(float(features[index]), 6),
            "weight": round(float(weights[index]), 6),
            "contribution": round(float(contribution[index]), 6),
        }
        for index in order
    ]
    first, second = order[:2]
    reason = (
        f"{restaurant['name_zh']}的{FEATURE_NAMES_ZH[first]}和"
        f"{FEATURE_NAMES_ZH[second]}与当前需求最匹配，"
        f"招牌菜是{restaurant['signature_dish']}。"
    )
    return reason, details


def recommend(query: str, top_k: int, skill_root: Path) -> dict[str, Any]:
    assets = skill_root / "assets"
    bundle = json.loads(
        (assets / "model_bundle.json").read_text(encoding="utf-8")
    )
    restaurants = load_restaurants(assets / "macau_restaurants_50.csv")
    parsed = parse_query(query, restaurants)
    features = build_restaurant_features(restaurants, parsed)

    soft_weights = component_weights(
        parsed["user"],
        parsed["context"],
        bundle["component_state_dicts"]["SoftLabel"],
    )
    hybrid_weights = component_weights(
        parsed["user"],
        parsed["context"],
        bundle["component_state_dicts"]["ParetoHybrid"],
    )
    alpha = float(bundle["blend_alpha"])
    weights = (1.0 - alpha) * soft_weights + alpha * hybrid_weights
    weights = weights / np.sum(weights)
    scores = features @ weights
    probability = stable_softmax(scores / float(bundle["temperature"]))
    safety_mix = float(bundle["safety_mix"])
    probability = (
        (1.0 - safety_mix) * probability + safety_mix / len(probability)
    )

    order = np.argsort(probability)[::-1][:top_k]
    recommendations = []
    for rank, index in enumerate(order, 1):
        restaurant = restaurants[int(index)]
        reason, contribution = make_reason(
            restaurant,
            features[index],
            weights,
        )
        recommendations.append(
            {
                "rank": rank,
                "restaurant_id": restaurant["restaurant_id"],
                "name": restaurant["name_zh"],
                "district": restaurant["district"],
                "cuisine": restaurant["cuisine"],
                "signature_dish": restaurant["signature_dish"],
                "price_mop": int(round(restaurant["avg_price_mop"])),
                "meal_role": meal_role(restaurant["cuisine"]),
                "fictional_avg_wait_min": int(
                    round(restaurant["avg_wait_min"])
                ),
                "recommendation_probability": round(
                    float(probability[index]),
                    8,
                ),
                "score": round(float(scores[index]), 8),
                "reason": reason,
                "contribution_breakdown": contribution,
            }
        )

    public_context = {
        key: value
        for key, value in parsed.items()
        if key not in {"user", "context"}
    }
    base_priorities = parsed["user"][-4:].astype(np.float64)
    base_priorities = base_priorities / np.sum(base_priorities)
    context_residual = weights.astype(np.float64) - base_priorities
    required_disclosures = list(parsed["assumptions"])
    required_disclosures.extend(parsed["limitations"])
    required_disclosures.append("食店、等待时间与交互均为比赛用虚构数据")
    return {
        "ok": True,
        "tool": "hidden_bites_recommend",
        "model_version": bundle["version"],
        "model_type": bundle["model_type"],
        "query": query,
        "parsed_context": public_context,
        "base_priorities": {
            name: round(float(value), 6)
            for name, value in zip(FEATURE_NAMES_ZH, base_priorities)
        },
        "context_residual": {
            name: round(float(value), 6)
            for name, value in zip(FEATURE_NAMES_ZH, context_residual)
        },
        "context_residual_definition": (
            "最终权重减去基础优先级；正值表示完整场景综合提高该维度，"
            "负值表示综合降低，四项之和为0。"
        ),
        "four_weights": {
            name: round(float(value), 6)
            for name, value in zip(FEATURE_NAMES_ZH, weights)
        },
        "model_parameters": {
            "blend_alpha": round(float(bundle["blend_alpha"]), 6),
            "temperature": round(float(bundle["temperature"]), 6),
            "safety_mix": round(float(bundle["safety_mix"]), 6),
        },
        "recommendations": recommendations,
        "answer_constraints": {
            "required_disclosures": required_disclosures,
            "allowed_evidence": [
                "parsed_context",
                "base_priorities",
                "context_residual",
                "context_residual_definition",
                "four_weights",
                "recommendations.reason",
                "recommendations.contribution_breakdown",
                "recommendations.meal_role",
                "recommendations.fictional_avg_wait_min",
            ],
            "forbidden_claims": [
                "真实营业状态、地址、电话、评分或食品安全承诺",
                "实时排队、实时天气或无需等待保证",
                "JSON未提供的室内环境、步行距离或翻台率",
                "把recommendation_probability解释成满意概率",
            ],
            "meal_presentation": (
                "晚餐场景下，把甜品或小食标为加餐；"
                "可指出排名中首个正餐候选，但不得改变模型排名。"
            ),
        },
        "explanation_policy": (
            "只依据answer_constraints.allowed_evidence解释；"
            "必须披露required_disclosures；不得补充JSON外的餐厅事实。"
        ),
        "disclaimer": "所有食店、等待时间与交互均为比赛用虚构数据。",
    }


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Hidden Bites v23 StackedPareto recommendation tool"
    )
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query")
    query_group.add_argument(
        "--query-file",
        type=Path,
        help="UTF-8 text file containing the user request",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON instead of indented JSON",
    )
    args = parser.parse_args()
    query = (
        args.query_file.read_text(encoding="utf-8")
        if args.query_file is not None
        else args.query
    )
    if not 1 <= args.top_k <= 10:
        raise SystemExit("--top-k must be between 1 and 10")
    if not query or not query.strip():
        raise SystemExit("--query must not be empty")
    result = recommend(
        query.strip(),
        args.top_k,
        Path(__file__).resolve().parent.parent,
    )
    json.dump(
        result,
        sys.stdout,
        ensure_ascii=False,
        indent=None if args.compact else 2,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
