[English](README.md) | [中文](README_CN.md)

![ds-skills 技能图谱](assets/graph.png)

# ds-skills

从优秀 Kaggle Notebook 中提炼的数据科学技能库。

340 个可复用技巧，覆盖 5 大领域，均从高票 Kaggle 竞赛方案中提取。每个技能是一个独立的 SKILL.md —— 可直接被 Claude Code、Cursor、Codex 或任何 AI 编程助手加载使用。

> 持续更新中 —— 600+ 场竞赛待处理，自动化蒸馏 Agent 定期提取新技能。Star 本仓库以获取最新动态。

## 技能一览

| 领域 | 数量 | 示例 |
|------|------|------|
| **tabular** (表格数据) | 84 | adversarial-validation, lgbm-dart-boosting, optuna-lgbm-tuning, tfidf-weighted-category-counts |
| **nlp** (自然语言处理) | 108 | deberta-classification, layerwise-lr-decay, keystroke-essay-reconstruction, last-hidden-states-concat |
| **cv** (计算机视觉) | 88 | arcface-metric-learning, bev-conv1-channel-expansion, deepsort-majority-vote-relabel, keypoint-aware-raster-augmentation |
| **timeseries** (时间序列) | 39 | k-mode-gaussian-nll-loss, multimodal-trajectory-head, availability-masked-regression-loss, event-anchored-frame-sync |
| **llm** (大语言模型) | 21 | wikipedia-rag-retrieval, kv-cache-prefix-scoring, counting-trap-adversarial-essay, weighted-attack-strategy-sampling |

完整列表：运行 `ds-skills list` 或浏览 [ds-skills.com](https://ds-skills.com)。查看[更新日志](CHANGELOG.md)了解最新动态。

## 安装技能到你的 AI 助手

### 方式一：让你的 AI 助手来安装

将本仓库链接复制给你的 AI 编程助手（Claude Code、Cursor、Codex 等），让它自行安装：

```
https://github.com/wenmin-wu/ds-skills
```

你的 AI 助手会读取 README，安装 CLI，并自动完成配置。

### 方式二：ds-skills CLI

```bash
pip install ds-skills-cli
ds-skills install --agent claude-code   # 或: cursor, codex
ds-skills setup --agent claude-code     # 让你的 Agent 学会使用 CLI
```

更多命令：

```bash
ds-skills install nlp/deberta-classification --agent claude-code  # 安装单个技能
ds-skills install --agent claude-code --domain tabular            # 安装某领域
ds-skills list                          # 列出所有技能
ds-skills search "gradient boosting"    # 按关键词搜索
ds-skills show nlp/deberta-classification  # 预览技能内容
ds-skills stats                         # 查看统计信息
```

所有命令均支持 `--json` 参数输出结构化 JSON（Agent 友好）。

### 方式二：本地脚本复制

如果你更喜欢克隆仓库并本地复制：

```bash
git clone https://github.com/wenmin-wu/ds-skills.git
cd ds-skills

# 复制到 Agent 的技能目录
scripts/skills-copy --dest ~/.claude/skills    # Claude Code
scripts/skills-copy --dest ~/.cursor/rules     # Cursor
scripts/skills-copy --dest ~/.codex/skills     # Codex
```

复制时自动扁平化：`skills/nlp/deberta-classification/` 变为 `nlp-deberta-classification/`。

## 目录结构

```
skills/<领域>/<技巧名>/SKILL.md
```

每个 SKILL.md 包含：
- **YAML 头部** —— 名称和描述（供 Agent 自动发现）
- **概述** —— 用途和使用场景
- **快速开始** —— 最小可运行代码
- **工作流** —— 分步骤说明
- **关键决策** —— 权衡取舍与参数选择
- **参考** —— 源 Kaggle Notebook 链接

## 脚本

| 脚本 | 用途 |
|------|------|
| `scripts/skills-list` | 列出所有技能及元数据。`--json` 输出结构化数据，`--html` 重新生成技能图谱 |
| `scripts/skills-copy --dest <目录>` | 将技能扁平化复制到任意 Agent 的技能目录 |

## 数据来源

每个技能均从 [Kaggle](https://www.kaggle.com) 高票 Notebook 中提炼。每个 SKILL.md 的参考部分链接到原始 Notebook。

已处理竞赛：Playground Series S6E3、HPA 单细胞分类、Google QUEST 问答、Feedback Prize 英语学习、CommonLit 学生摘要、Kaggle LLM 科学考试、H&M 时尚推荐、CommonLit 可读性、Data Science Bowl 2019、CHAMPS 分子耦合、Child Mind Institute 睡眠检测、LLM 检测 AI 文本、RANZCR CLiP、Riiid 答题预测、NBME 临床笔记评分、Deep Past 阿卡德语翻译、CMI 传感器行为检测、NeurIPS Ariel 数据挑战赛 2024、SIIM-FISABIO-RSNA COVID-19 检测、Shopee 商品匹配、NFL 大数据碗、Jigsaw 毒性分类偏差检测、MAP 学生数学误解图谱、OTTO 多目标推荐系统、American Express 信用违约预测、Tweet 情感提取、PLAsTiCC 天文分类、NeurIPS Ariel 数据挑战赛 2025、CZII 冷冻电子断层扫描粒子识别、Eedi 数学误解挖掘、Feedback Prize 论证效果预测、Coleridge Initiative 数据集识别挑战、BMS 分子翻译、Jigsaw 多语言毒性评论分类、Santander 客户交易预测、Quora 问题配对、Google AI4Code 代码理解、RSNA 2022 颈椎骨折检测、Foursquare 地点匹配、Feedback Prize 学生写作评估、TensorFlow 2.0 问答、APTOS 2019 眼底病变检测、Home Credit 信用违约风险、TalkingData 广告追踪欺诈检测、RSNA 2024 腰椎退行性病变分类、USPTO 专利可解释 AI、RSNA 乳腺癌筛查检测、VinBigData 胸部X光异常检测、Severstal 钢材缺陷检测、NeurIPS 开放聚合物预测 2025、Drawing with LLMs、PII 数据检测、ICR 年龄相关疾病识别、Open Problems 多模态单细胞整合、Jigsaw 毒性评论严重性评级、PANDA 前列腺癌分级评估、RSNA 颅内出血检测、SIIM-ACR 气胸分割、Quora 不良问题分类、LLMs - You Can't Please Them All、写作过程关联写作质量、NFL 头盔识别与球员指派、Lyft 自动驾驶运动预测。

## 许可

MIT
