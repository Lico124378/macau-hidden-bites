# QwenPaw推荐工具契约 — Hidden Bites v23

Notebook完成后生成：

`hidden_bites_v23_competition_final_qwenpaw_bundle.json`

QwenPaw只负责：

- 将自然语言转成用户画像与场景字段；
- 调用推荐模型；
- 解释基础优先级、场景修正、最终权重与餐厅贡献。

QwenPaw不能修改模型排名、权重、价格或食店属性。

核心输出：

```json
{
  "base_priorities": {},
  "context_residual": {},
  "four_weights": {},
  "blend_alpha": 0.40,
  "temperature": 0.05,
  "safety_mix": 0.10,
  "ranked_restaurants": [],
  "feature_contributions": [],
  "reason": ""
}
```

解释应区分：

1. 用户原本重视什么；
2. 当前场景如何改变权重；
3. 推荐餐厅在哪些维度贡献最大。
