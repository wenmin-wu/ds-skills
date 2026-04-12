# Changelog

All notable changes to ds-skills are documented here.

Browse skills at [ds-skills.com](https://ds-skills.com) | Install via `pip install ds-skills-cli`

---

## 2026-04-12 — Hybrid Search & Visitor Tracking

### Hybrid Search (Vector + Full-Text)

Search on [ds-skills.com](https://ds-skills.com/search) is now powered by **hybrid search** combining PostgreSQL full-text ranking with vector similarity via [pgvector](https://github.com/pgvector/pgvector) and [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

- **Semantic understanding** — search by meaning, not just keywords. Searching "how to handle missing values" finds imputation skills even when the exact phrase doesn't appear.
- **Cross-language** — search in Chinese ("图像分割") finds English CV segmentation skills.
- **RRF fusion** — Reciprocal Rank Fusion combines text and vector rankings for best-of-both results.

### Visitor Tracking

- Unique visitor counter on the landing page (UUID-based, no cookies).
- Event logging for views, searches, and downloads — IP and User-Agent stored for analytics.
- CLI users are also tracked via a persistent UUID in `~/.ds-skills/config.json`.

---

## 2026-04-11 — ds-skills Hub Launch

### ds-skills.com

The official web hub is live at [ds-skills.com](https://ds-skills.com):

- **Interactive Skills Map** — vis.js force-directed graph showing all skills organized by domain.
- **Browse & Search** — filter by domain (tabular, nlp, cv, timeseries, llm), search across all skills.
- **Skill Detail** — full SKILL.md content rendered in-browser with syntax highlighting.
- **Download** — one-click ZIP download for individual skills or entire domains.
- **Wishlist** — request new skills via GitHub Issues, with Chinese template support.
- **Dark mode & i18n** — English and Chinese UI.

### CLI v0.3.0

```bash
pip install ds-skills-cli
```

- `ds-skills install --agent claude-code` — install all skills for your agent.
- `ds-skills install nlp/deberta-classification --agent claude-code` — install a single skill.
- `ds-skills setup --agent claude-code` — teach your agent to use the CLI.
- `ds-skills search "gradient boosting"` — search skills from the terminal.
- `ds-skills list`, `ds-skills show`, `ds-skills stats` — browse and inspect.
- All commands support `--json` for structured output (agent-friendly).
- Visitor tracking with persistent UUID.

---

## 2026-04-08 — Initial Release

### 290 Skills from 50+ Kaggle Competitions

First public release of ds-skills — a library of reusable data science techniques distilled from top-voted Kaggle notebooks.

- **5 domains**: tabular (78), nlp (97), cv (67), timeseries (32), llm (16)
- **Self-contained** — each SKILL.md includes overview, quick start, workflow, key decisions, and source references.
- **Agent-ready** — designed to be loaded directly into Claude Code, Cursor, Codex, or any AI coding agent.
- **CLI v0.1.0** — `ds-skills install`, `ds-skills list`, `ds-skills search`.
- **Local install script** — `scripts/skills-copy --dest <dir>` for manual setup.
