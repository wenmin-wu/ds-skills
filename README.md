[English](README.md) | [中文](README_CN.md)

![ds-skills mindmap](assets/graph.png)

# ds-skills

Data Science Skills Distilled from Awesome Kaggle Notebooks.

126 reusable techniques across 5 domains, each extracted from top-voted Kaggle competition solutions. Every skill is a self-contained SKILL.md — ready to be loaded by Claude Code, Cursor, Codex, or any AI coding agent.

> More skills coming soon — 600+ competitions in the pipeline, with new skills distilled regularly via an automated agent. Star this repo to stay updated.

## Skills

| Domain | Count | Examples |
|--------|-------|---------|
| **tabular** | 37 | adversarial-validation, crps-cdf-loss, optuna-lgbm-tuning, spatial-distance-aggregation |
| **nlp** | 39 | deberta-classification, layerwise-lr-decay, mbr-decoding-reranking, multi-temperature-candidate-sampling |
| **cv** | 24 | arcface-metric-learning, chunked-gpu-similarity-search, dicom-hounsfield-normalization, knn-distance-threshold-matching |
| **timeseries** | 21 | mc-dropout-uncertainty, correlated-double-sampling, gradient-event-boundary-detection, learnable-fir-filter |
| **llm** | 5 | wikipedia-rag-retrieval, kv-cache-prefix-scoring, confidence-threshold-fallback |

Full list: run `scripts/skills-list` or browse the [interactive mindmap](graph/skills-map.html).

## Install skills to your agent

<details>
<summary><b>Claude Code</b></summary>

```bash
scripts/skills-copy --dest ~/.claude/skills
```

</details>

<details>
<summary><b>Cursor</b></summary>

```bash
scripts/skills-copy --dest ~/.cursor/rules
```

</details>

<details>
<summary><b>Codex</b></summary>

```bash
scripts/skills-copy --dest ~/.codex/skills
```

</details>

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

Competitions processed so far: Playground Series S6E3, HPA Single Cell Classification, Google QUEST Q&A, Feedback Prize ELL, CommonLit Student Summaries, Kaggle LLM Science Exam, H&M Fashion Recommendations, CommonLit Readability, Data Science Bowl 2019, CHAMPS Scalar Coupling, Child Mind Institute Sleep States, LLM Detect AI Text, RANZCR CLiP, Riiid Answer Prediction, NBME Score Clinical Patient Notes, Deep Past Challenge (Akkadian Translation), CMI Detect Behavior with Sensor Data, NeurIPS Ariel Data Challenge 2024, SIIM-FISABIO-RSNA COVID-19 Detection, Shopee Price Match Guarantee, NFL Big Data Bowl.

## License

MIT
