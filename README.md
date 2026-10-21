# 巷味濠江 · Macau Hidden Bites

> 一个把预算、天气与等待意愿写进推荐逻辑的澳门餐饮推荐 Agent。
> 「千模百炼」AI 开发者系列之学生竞赛参赛作品（队伍：UM-琢光）。

**重要声明：本项目所有食店、用户与交互数据均为比赛用虚构/模拟数据**，仅用于验证推荐链路、因果切分、权重恢复与故障恢复方法，不代表在真实澳门游客场景中的效果。

## 项目简介

当游客说清楚预算、同行人数、天气和等待意愿后，Agent 给出一组可追溯的四维权重（口味匹配、品质稳定、便利程度、性价比），并据此完成店铺推荐。完整链路为：

```
需求解析 → 用户画像 → 四权重排序 → 精准推荐
```

- 结构化知识库：50 家虚构澳门食店，含区域、菜系、价格与四维特征，5 类标签体系
- 推荐引擎：Thompson Sampling 曝光分配 + StackedPareto 权重融合模型（四权重层融合 SoftLabel 的结构恢复能力与 ParetoHybrid 的策略信息）
- 评估体系：5,000 名仿真用户、100,000 条模拟交互，按用户 ID 划分 5 个互斥测试折，五折交叉验证
- 部署形态：封装为 QwenPaw 自定义 Skill 本地部署（见 `skill/`）

## 核心结果

| 方法 | 策略价值 | NDCG@5 | 四权重 Cosine | 四权重 MAE |
|---|---:|---:|---:|---:|
| StaticPriority | 0.61963 | 0.99677 | 0.97268 | 0.04804 |
| Naive | 0.63735 | 0.99755 | 0.96142 | 0.06638 |
| SoftLabel | 0.63789 | **0.99941** | **0.98924** | **0.03508** |
| DR | 0.63213 | 0.98821 | 0.75277 | 0.20095 |
| ParetoHybrid | 0.63713 | 0.99544 | 0.89820 | 0.11743 |
| **StackedPareto（部署模型）** | **0.64068** | 0.99902 | 0.98181 | 0.04637 |

统计证据（详见 `docs/FINAL_RESULTS.md` 与 `docs/training_report.pdf`）：

- StackedPareto − Naive：配对平均差 **+0.003325**，用户级 95% CI **[0.003188, 0.003451]**（2,000 次 bootstrap），区间完全大于 0；5,000 名用户中 3,744 名差值为正
- 独立未见子集（fold 2–4，3,000 名用户）：+0.003396，95% CI [0.003219, 0.003574]
- 三类压力测试（餐厅特征噪声、用户优先级漂移、特征—偏好组合偏移）下策略价值约 0.6402–0.6405，最坏下降仅 0.00047

### 迭代早期的关键失败（评估陷阱的定位与修正）

开发早期通过 **10 个数据种子的对照实验**定位到评估陷阱：SoftLabel 一次性交叉拟合方案在 **10/10 个数据种子上 NDCG 上升，但 10/10 个种子的真实策略价值下降**（均值约 −0.0123）。据此放弃"一次性交叉拟合蒸馏"路线，改为在四权重层融合两个模型（StackedPareto）。完整失败记录见 `docs/training_report.pdf` 第 4 节。

### 如实陈述

- 单独 DR 未优于 Naive（差值 −0.005227，95% CI 完全小于 0）；ParetoHybrid 也未单独超过 Naive
- StackedPareto 的收益折中可解释，但**不应被解读为"显著领先所有方法"**；数据为模拟数据，未在真实用户上验证
- 部署阶段曾出现两类非模型错误（需求解析映射缺失、大模型补写不存在的事实），已通过扩充短语解析、强制 UTF-8 输出与脚本生成受控 Markdown 修复，需求解析成功率由 27.3% 提升至 68.9%（见 `docs/qwenpaw_deployment_guide.pdf`）

## 快速开始

```bash
# 环境：Python 3.10+
python -m pip install -r requirements.txt

# 推荐演示（QwenPaw Skill 核心脚本）
python skill/hidden-bites-macau-food-agent/scripts/recommend.py

# 自检与回归测试
python skill/hidden-bites-macau-food-agent/scripts/self_check.py
python skill/hidden-bites-macau-food-agent/scripts/regression_test.py
```

核心推荐引擎的完整实现（Thompson Sampling 分配器、反馈回路、便利维度动态学习、权重优化等）见 `src/hidden_bites_core.py`。

## 目录结构

```
├── skill/hidden-bites-macau-food-agent/   # QwenPaw Skill（部署最终版 v23）
│   ├── SKILL.md                           # Skill 定义
│   ├── scripts/                           # recommend / self_check / regression_test
│   └── assets/                            # 50 家虚构食店数据、模型产物包
├── src/hidden_bites_core.py               # 核心代码整合版（推荐引擎等 6 个模块）
├── data/                                  # 仿真数据与实验报告（含虚构数据标记）
├── config/api_config_template.json        # API 配置模板（占位符，需自行申请填入）
├── workspace_template/                    # QwenPaw 工作区模板
├── docs/                                  # 终版结果卡、训练报告、部署说明、策划书
└── requirements.txt
```

## 数据合规

- `skill/*/assets/macau_restaurants_50.csv` 与 `data/` 下的数据均为虚构比赛数据（行内带 `fictional_competition_data` 标记）
- 真实商户数据接入仅提供框架（`config/api_config_template.json`），需自行申请高德/点评等 API 授权后使用

## 团队

**UM-琢光**

- 李浩瑞（队长）：推荐链路设计、Thompson Sampling 与 StackedPareto 模型、五折交叉验证与统计检验、训练评估报告撰写
- 段若约（队员）：仿真用户与交互数据生成、部署阶段端到端测试与回归验证、概念设计书撰写

## License

MIT（见 `LICENSE`）
