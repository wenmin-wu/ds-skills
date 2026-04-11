# ds-skills-cli

Agent-friendly CLI to browse, search, and pull data science skills from [ds-skills.com](https://ds-skills.com).

## Install

```bash
pip install ds-skills-cli
```

## Usage

```bash
# List all skills
ds-skills list
ds-skills list --domain nlp

# Search
ds-skills search "deberta"

# Show full skill content
ds-skills show nlp/deberta-classification

# Pull a skill to current directory
ds-skills pull nlp/deberta-classification
ds-skills pull nlp/deberta-classification --dest ./my-skills

# Pull all skills in a domain
ds-skills pull nlp --dest ./skills

# Install to an AI agent
ds-skills install --agent claude-code
ds-skills install --agent cursor --domain nlp

# Stats
ds-skills stats
```

All commands support `--json` for structured output (JSON to stdout, human messages to stderr).

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Not found |
| 3 | Invalid input |
