# Hidden Bites v23 → QwenPaw 2.x 部署说明

本包把已经训练完成的`StackedPareto`模型封装为QwenPaw自定义Skill。部署只做
CPU推理，不会重新训练，不需要CUDA。

## 包内文件

```text
hidden_bites_qwenpaw_skill.zip
skill/
  hidden-bites-macau-food-agent/
    SKILL.md
    requirements.txt
    scripts/
      recommend.py
      self_check.py
      regression_test.py
    assets/
      model_bundle.json
      macau_restaurants_50.csv
workspace_template/
  AGENTS.md
  SOUL.md
  PROFILE.md
FINAL_RESULTS.md
QWENPAW_AGENT_CONTRACT.md
```

`hidden_bites_qwenpaw_skill.zip`用于控制台直接上传。`workspace_template/`用于把
普通QwenPaw Agent设定为专用的Hidden Bites Agent。

本优化版保持最终`StackedPareto`模型参数和餐厅数据不变，修正了：

- Windows控制台UTF-8 JSON输出；
- “不想等太久”、人数、地点别名和否定偏好的解析；
- 默认区域、时段和预算的强制披露；
- 甜品/小食与正餐候选的展示区分；
- 只依据工具JSON解释、禁止补写实时商家事实的回答边界。

## 1. 安装并启动QwenPaw

QwenPaw当前要求Python 3.11至3.13。Windows PowerShell中运行：

```powershell
py -3.11 -m pip install -U qwenpaw
qwenpaw init --defaults
qwenpaw app
```

打开：

```text
http://127.0.0.1:8088/
```

在控制台的模型设置中启用一个可用模型。使用云模型时配置相应API Key；使用
QwenPaw Local、Ollama或LM Studio时按QwenPaw控制台提示配置。

## 2. 创建专用Agent

在QwenPaw控制台创建一个新Agent：

```text
Agent ID: hidden-bites
名称: Hidden Bites
描述: 澳门餐饮四权重可解释推荐Agent（比赛演示）
语言: 中文
```

创建后切换到该Agent。

## 3. 上传Skill

进入：

```text
工作区 → 技能 → 添加技能 → 通过Zip上传
```

选择本包中的：

```text
hidden_bites_qwenpaw_skill.zip
```

QwenPaw通过控制台上传的Skill默认启用。确认列表中出现：

```text
hidden-bites-macau-food-agent
```

如果安全扫描显示警告，检查内容后再启用。本Skill只读取包内CSV/JSON并执行本地
NumPy推理，不访问网络、不读取API Key、不修改模型。

## 4. 安装推理依赖

Skill只依赖NumPy。在启动QwenPaw的同一个Python环境中运行：

```powershell
py -3.11 -m pip install "numpy>=1.26"
```

若你使用的是虚拟环境，用该虚拟环境的`python`替代`py -3.11`。

## 5. 设置Agent身份

在QwenPaw工作区文件编辑器中，将`workspace_template/`里的三个文件内容分别复制到
该Agent工作区的：

```text
AGENTS.md
SOUL.md
PROFILE.md
```

如果原文件已有重要内容，先备份再替换。重新打开一个会话，让新身份和Skill清单生效。

默认磁盘位置通常是：

```text
%USERPROFILE%\.qwenpaw\workspaces\hidden-bites\
```

如果设置了`QWENPAW_WORKING_DIR`，以该环境变量指定的位置为准。

## 6. 自检

安装包根目录可直接运行：

```powershell
.\setup_and_test_windows.bat
```

它会依次安装NumPy、执行基础自检和解析/UTF-8回归测试。也可以从
`hidden-bites`工作区根目录单独运行：

```powershell
python skills\hidden-bites-macau-food-agent\scripts\self_check.py
python skills\hidden-bites-macau-food-agent\scripts\regression_test.py
```

期望输出：

```text
Hidden Bites QwenPaw Skill self-check: PASS
Hidden Bites regression tests: PASS
```

也可以直接测试推理：

```powershell
python skills\hidden-bites-macau-food-agent\scripts\recommend.py `
  --query "第一次去澳门，雨天晚上两个人，人均150，想吃地道特色" `
  --top-k 5
```

## 7. 在QwenPaw中测试

向Hidden Bites Agent发送：

```text
第一次去澳门，雨天晚上两个人，人均150，想吃地道特色。
```

正确行为：

1. Agent调用`hidden-bites-macau-food-agent`。
2. Skill把原始需求写入`.hidden_bites/request.txt`。
3. 调用本地推理脚本。
4. 将“不想等太久”识别为`urgent=1`，将“两个人”保留为`party_size=2`。
5. 因用户未指定区域，明确说明默认按“新马路/大三巴”生成候选。
6. 返回基础优先级、场景综合修正、最终四权重和3至5家虚构澳门食店。
7. 保持模型排名不变，并把甜品或小食标为加餐/小食，不冒充完整正餐。
8. 明确说明食店、等待时间与交互都是比赛用虚构数据。

工具已对Windows标准输出强制使用UTF-8。Agent不应在运行时修改Skill脚本来处理
编码问题。

如果Agent没有调用Skill：

- 确认Skill已启用。
- 新建一次会话，刷新Skill清单。
- 检查`AGENTS.md`是否已替换。
- 直接运行`self_check.py`区分“Agent触发问题”和“推理环境问题”。

## 8. 比赛展示建议

展示顺序控制在两分钟内：

1. 输入自然语言需求。
2. 展示解析出的预算、时间、天气和用户场景。
3. 展示四权重百分比。
4. 展示前三名和每家店的主要贡献维度。
5. 打开`FINAL_RESULTS.md`说明五折实验和StackedPareto证据。

不要在展示时重新运行训练Notebook。

## 提交文件建议

比赛若只允许一个ZIP，提交完整的
`hidden_bites_v23_verified_submission_final.zip`，并把本QwenPaw部署包作为
“运行附件”一并放入提交目录。

若允许分别提交：

- 算法、数据、实验结果：`hidden_bites_v23_verified_submission_final.zip`
- QwenPaw Agent部署：`hidden_bites_v23_qwenpaw_agent_deploy.zip`
- QwenPaw控制台导入：`hidden_bites_qwenpaw_skill.zip`

## 边界

- 食店、用户与交互全部为虚构数据。
- 本包验证的是比赛演示部署，不代表真实餐饮平台生产上线。
- QwenPaw本体和所选大语言模型需要单独安装、配置。
- QwenPaw只负责意图理解和表达；排序由本地StackedPareto模型完成。
- 虚构平均等待时间不是实时排队信息，不能承诺“无需等待”。
- 回答只能使用工具JSON中的证据，不补写室内环境、步行距离、翻台率或真实商家信息。
