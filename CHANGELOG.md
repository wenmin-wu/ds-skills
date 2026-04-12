# What's New

---

## 2026-04-12 — Semantic Search

You can now search skills by meaning, not just keywords.

**On the web** — [ds-skills.com/search](https://ds-skills.com/search)

Try natural language queries:

- `how to handle missing values in tabular data`
- `ensemble methods for reducing overfitting`
- `图像分割` (Chinese works too)

**From the CLI**

```bash
ds-skills search "handle imbalanced classes"
ds-skills search "gradient boosting" --json   # structured output for agents
```

---

## 2026-04-11 — ds-skills.com & CLI v0.3.0

### Browse skills on the web

[ds-skills.com](https://ds-skills.com) is live — an interactive hub to explore, search, and download skills.

- Interactive skills map showing all 300+ skills by domain
- Full skill content with syntax-highlighted code blocks
- One-click ZIP download for any skill or entire domain
- [Request new skills](https://ds-skills.com/wishlist) via the wishlist
- English & Chinese UI, dark mode

### Install with the CLI

```bash
pip install ds-skills-cli
```

**Load all skills into your agent:**

```bash
ds-skills install --agent claude-code    # or: cursor, codex
ds-skills setup --agent claude-code      # teach your agent the CLI
```

**Pick what you need:**

```bash
ds-skills install nlp/deberta-classification --agent claude-code   # single skill
ds-skills install --agent claude-code --domain tabular             # single domain
```

**Explore:**

```bash
ds-skills list                            # list all skills
ds-skills search "gradient boosting"      # search by keyword
ds-skills show nlp/deberta-classification # preview a skill
ds-skills stats                           # aggregate stats
```

All commands support `--json` for structured output (agent-friendly).

---

## 2026-04-08 — Initial Release

**300+ data science skills distilled from 50+ Kaggle competitions.**

5 domains: tabular, nlp, cv, timeseries, llm. Each skill is a self-contained SKILL.md with overview, quick start code, step-by-step workflow, key decisions, and source references — ready to load into Claude Code, Cursor, Codex, or any AI coding agent.

```bash
pip install ds-skills-cli
ds-skills install --agent claude-code
```
