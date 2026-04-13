[English](README.md) | [中文](README_CN.md)

![ds-skills mindmap](assets/graph.png)

# ds-skills

Data Science Skills Distilled from Awesome Kaggle Notebooks.

385 reusable techniques across 5 domains, each extracted from top-voted Kaggle competition solutions. Every skill is a self-contained SKILL.md — ready to be loaded by Claude Code, Cursor, Codex, or any AI coding agent.

> More skills coming soon — 600+ competitions in the pipeline, with new skills distilled regularly via an automated agent. Star this repo to stay updated.

## Skills

| Domain | Count | Examples |
|--------|-------|---------|
| **tabular** | 88 | adversarial-validation, yearly-partitioned-groupby, sparse-dense-hstack-lgbm, cross-dataset-user-aggregation |
| **nlp** | 108 | deberta-classification, layerwise-lr-decay, keystroke-essay-reconstruction, last-hidden-states-concat |
| **cv** | 115 | bigru-slice-feature-aggregator, exam-level-label-hierarchy-aggregation, slice-location-weighted-prior, exam-sequence-padded-mask-loss |
| **timeseries** | 48 | recursive-multistep-forecasting, tweedie-objective-zero-inflated, wavelet-denoising, snap-event-interaction-features |
| **llm** | 26 | vllm-lora-adapter-inference, last-token-logit-binary-scoring, completion-only-lm-training, multi-gpu-process-isolated-vllm |

Full list: run `ds-skills list` or browse [ds-skills.com](https://ds-skills.com). See [Changelog](CHANGELOG.md) for what's new.

## Install skills to your agent

### Option 1: Let your AI agent do it

Copy this repo URL into your AI coding agent (Claude Code, Cursor, Codex, etc.) and ask it to install ds-skills:

```
https://github.com/wenmin-wu/ds-skills
```

Your agent can read the README, install the CLI, and set everything up by itself.

### Option 2: ds-skills CLI

```bash
pip install ds-skills-cli
ds-skills install --agent claude-code   # or: cursor, codex
ds-skills setup --agent claude-code     # teach your agent the CLI
```

More commands:

```bash
ds-skills install nlp/deberta-classification --agent claude-code  # single skill
ds-skills install --agent claude-code --domain tabular            # single domain
ds-skills list                          # list all skills
ds-skills search "gradient boosting"    # search by keyword
ds-skills show nlp/deberta-classification  # preview a skill
ds-skills stats                         # aggregate stats
```

Add `--json` to any command for structured JSON output (agent-friendly).

### Option 2: Local copy script

If you prefer to clone the repo and copy skills locally:

```bash
git clone https://github.com/wenmin-wu/ds-skills.git
cd ds-skills

# Copy to your agent's skill directory
scripts/skills-copy --dest ~/.claude/skills    # Claude Code
scripts/skills-copy --dest ~/.cursor/rules     # Cursor
scripts/skills-copy --dest ~/.codex/skills     # Codex
```

Skills are flattened on copy: `skills/nlp/deberta-classification/` becomes `nlp-deberta-classification/`.

## Structure

```
skills/<domain>/<technique>/SKILL.md
```

Each SKILL.md contains:
- **YAML frontmatter** — name and description (for agent discovery)
- **Overview** — what and when to use
- **Quick Start** — minimal working code
- **Workflow** — step-by-step instructions
- **Key Decisions** — trade-offs and parameter choices
- **References** — link to source Kaggle notebook

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/skills-list` | List all skills with metadata. `--json` for structured output, `--html` to regenerate mindmap |
| `scripts/skills-copy --dest <dir>` | Copy skills flat to any agent's skills directory |

## Source

Every skill is distilled from a top-voted notebook on [Kaggle](https://www.kaggle.com). The References section in each SKILL.md links back to the original notebook.

Competitions processed so far: Playground Series S6E3, HPA Single Cell Classification, Google QUEST Q&A, Feedback Prize ELL, CommonLit Student Summaries, Kaggle LLM Science Exam, H&M Fashion Recommendations, CommonLit Readability, Data Science Bowl 2019, CHAMPS Scalar Coupling, Child Mind Institute Sleep States, LLM Detect AI Text, RANZCR CLiP, Riiid Answer Prediction, NBME Score Clinical Patient Notes, Deep Past Challenge (Akkadian Translation), CMI Detect Behavior with Sensor Data, NeurIPS Ariel Data Challenge 2024, SIIM-FISABIO-RSNA COVID-19 Detection, Shopee Price Match Guarantee, NFL Big Data Bowl, Jigsaw Unintended Bias in Toxicity Classification, MAP Charting Student Math Misunderstandings, OTTO Multi-Objective Recommender System, American Express Default Prediction, Tweet Sentiment Extraction, PLAsTiCC Astronomical Classification, NeurIPS Ariel Data Challenge 2025, CZII CryoET Object Identification, Eedi Mining Misconceptions in Mathematics, Feedback Prize Predicting Effective Arguments, Coleridge Initiative Show US the Data, Bristol-Myers Squibb Molecular Translation, Jigsaw Multilingual Toxic Comment Classification, Santander Customer Transaction Prediction, Quora Question Pairs, Google AI4Code, RSNA 2022 Cervical Spine Fracture Detection, Foursquare Location Matching, Feedback Prize - Evaluating Student Writing, TensorFlow 2.0 Question Answering, APTOS 2019 Blindness Detection, Home Credit Default Risk, TalkingData AdTracking Fraud Detection, RSNA 2024 Lumbar Spine Degenerative Classification, USPTO Explainable AI for Patent Professionals, RSNA Screening Mammography Breast Cancer Detection, VinBigData Chest X-ray Abnormalities Detection, Severstal Steel Defect Detection, NeurIPS Open Polymer Prediction 2025, Drawing with LLMs, PII Data Detection, ICR - Identifying Age-Related Conditions, Open Problems - Multimodal Single-Cell Integration, Jigsaw Rate Severity of Toxic Comments, PANDA Prostate Cancer Grade Assessment, RSNA Intracranial Hemorrhage Detection, SIIM-ACR Pneumothorax Segmentation, Quora Insincere Questions Classification, LLMs - You Can't Please Them All, Linking Writing Processes to Writing Quality, NFL Health & Safety Helmet Assignment, Lyft Motion Prediction for Autonomous Vehicles, SIIM-ISIC Melanoma Classification, Avito Demand Prediction Challenge, Corporación Favorita Grocery Sales Forecasting, NOAA Fisheries Steller Sea Lion Population Count, Jigsaw - Agile Community Rules Classification, RSNA 2023 Abdominal Trauma Detection, NFL 1st and Future - Impact Detection, RSNA STR Pulmonary Embolism Detection, M5 Forecasting - Accuracy.

## License

MIT
