---
name: hidden-bites-macau-food-agent
description: "当用户想在澳门找餐厅、选择吃什么、比较食店，或提到澳门美食推荐、预算、地点、天气、用餐时间、赶时间、不想排队、游客/本地人、人数、聚餐或忌口时使用。调用Hidden Bites v23 StackedPareto四权重模型，返回有输入识别说明、默认值披露和证据约束的虚拟澳门食店推荐。"
metadata:
  requires:
    bins: [python]
  qwenpaw:
    emoji: "🍽️"
---

# Hidden Bites 澳门美食推荐

使用已经训练完成的 Hidden Bites v23 `StackedPareto` 模型进行推荐。模型的
四个正权重分别是：

- 口味匹配
- 品质稳定
- 便利程度
- 性价比

本 Skill 中的50家澳门食店和全部训练交互均为比赛用虚构数据。始终向用户说明这一点，
不要把虚构食店描述成真实营业商家。

## 触发与澄清

用户提出澳门餐饮推荐、选店、比较餐厅或“吃什么”时使用。

优先直接推荐。模型对缺失信息有默认值，因此只有在用户明确要求高度定制、但完全没有
给出关键限制时，才追问一个简短问题，例如预算或大致区域。不要连续追问多个问题。

把用户原始需求完整保留，包括：

- 区域或附近地点
- 早餐、午餐、晚餐、宵夜
- 人均预算
- 游客或本地场景
- 雨天、赶时间、多人聚餐
- 辣、甜、清淡、海鲜、素食等偏好

## 调用推荐工具

不要把用户文本直接拼接进 shell 命令，避免命令注入。

1. 使用 `write_file` 将用户原始需求写入工作区相对路径：

   ```text
   .hidden_bites/request.txt
   ```

2. 使用 `execute_shell_command` 在当前工作区执行固定命令：

   ```bash
   python skills/hidden-bites-macau-food-agent/scripts/recommend.py --query-file .hidden_bites/request.txt --top-k 5
   ```

3. 如果当前工作区使用 `python3` 而不是 `python`，仅将命令中的解释器改成
   `python3`。不要改动脚本路径或模型文件。

4. 解析脚本输出的JSON。只有当顶层`ok`为`true`时才生成推荐答案。

不要在运行过程中修改`recommend.py`、模型JSON或餐厅CSV。工具报错时只报告部署或
依赖问题；不要展示逐步推理、英文调试记录、命令尝试过程或脚本源码。

如果提示缺少NumPy，告诉用户在运行QwenPaw的同一个Python环境执行：

```bash
python -m pip install "numpy>=1.26"
```

如果脚本、模型或餐厅数据不存在，明确报告部署不完整，不要退回到大模型自行编造排名。

## 核对输入识别

先检查`parsed_context`：

- `district_source`、`period_source`、`budget_source`为`default`时，必须向用户说明
  对应默认值。
- 保留`party_size`；`group`只是三人及以上或明确聚餐的模型特征，不能把两人说成
  “多人聚餐”。
- 使用`recognized_preferences`复述已识别的偏好与回避项，不把“不吃辣”等否定
  表达反向解释。
- 用户要求少等待时，先确认`urgent`为`1`。等待时间只是虚构数据中的模拟平均值，
  不能描述成实时排队。
- 原始需求中含过敏信息时，必须复述`limitations`，不提供食品安全保证。

## 组织回答

用中文给出：

1. 一句话概括已识别的场景，并紧接着披露`assumptions`和`limitations`。
2. 按“基础优先级 → 场景综合修正 → 最终四权重”解释：
   - `base_priorities`表示用户输入形成的基础优先级；
   - `context_residual`表示模型结合完整场景后的综合变化；
   - `four_weights`是最终权重，换算为百分比。
3. 按原排名推荐三至五家食店，包含名称、区域、菜系、招牌菜、人均价格、
   `meal_role`和模型理由。
4. 理由只取自`reason`、`contribution_breakdown`及
   `fictional_avg_wait_min`。只有用户要求少等待时才需要展示虚构平均等待时间。
5. 晚餐场景中，把“甜品或加餐”“特色小食”如实标注；可以指出排名中首个
   “正餐候选”，但不得改变模型排名。
6. 结尾使用JSON中的`disclaimer`。

`context_residual`是完整场景的综合修正。除非JSON另有明确字段，不要声称某一个词、
天气或身份单独造成了某个精确权重变化。

不要：

- 发明营业时间、地址、真实评分、联系电话或食品安全承诺。
- 发明室内环境、步行距离、翻台率、出餐速度或“无需等待”。
- 把虚构平均等待时间说成今晚的实时队列。
- 把`recommendation_probability`说成真实用户满意概率。
- 用大模型常识修改模型的排序。
- 隐藏默认区域、默认预算、默认时段或未建模限制。
- 把甜品店、烘焙店或小食店直接描述成完整正餐。

## 示例

用户：

```text
第一次去澳门，雨天晚上两个人，人均150，想吃地道特色。
```

应先调用工具，再按工具JSON说明默认区域、动态四权重和推荐结果。不要在调用前猜测
餐厅，也不要输出内部执行过程。
