# ⭐ StarAutoManager

> Automatically organize your GitHub stars into Star Lists using LLM-powered categorization that **learns your habits**.

[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-automated-blue?logo=githubactions)](https://github.com/features/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## How It Works

StarAutoManager doesn't just randomly categorize your stars — it **learns from your existing Star Lists** first, then applies the same categorization philosophy to uncategorized repos.

```
┌─────────────────────────────────────────────────────────┐
│  1. Fetch your existing Star Lists (training data)      │
│  2. Fetch all your starred repos                        │
│  3. Identify uncategorized repos                        │
│  4. LLM analyzes your categorization patterns           │
│  5. Classify uncategorized repos following YOUR style   │
│  6. Apply changes via GitHub GraphQL API                │
│  7. Generate report (Issue + STARS.md)                  │
└─────────────────────────────────────────────────────────┘
```

### Learn → Classify Pipeline

The LLM receives your existing lists as context:
- Studies how granular your categories are (broad vs. narrow)
- Detects whether you organize by domain, technology, purpose, or a mix
- Learns your naming conventions
- Then classifies new repos using the same logic

**Cold start?** If you have zero Star Lists, it proposes a sensible category structure from scratch.

## Features

### Core
- 🧠 **Learn-then-classify** — LLM learns from your existing lists before categorizing
- 🔄 **Incremental processing** — cache prevents re-categorizing already-processed repos
- 🎯 **Confidence scoring** — high/medium/low confidence for each assignment
- 📖 **Two-pass analysis** — fetches README for low-confidence repos and re-evaluates
- 📦 **Batch processing** — handles hundreds of stars efficiently with concurrent LLM calls
- 🆕 **Smart list creation** — suggests new lists when repos don't fit existing ones
- 🏃 **Dry run mode** — preview changes before applying

### Smart Analysis
- 🗑️ **Stale repo detection** — flags archived/abandoned repos
- 🔄 **Duplicate detection** — finds repos serving similar purposes
- 🌐 **Language statistics** — breakdown of your star collection by language
- 🏷️ **Topic analysis** — most common topics across your stars
- 🏥 **List health check** — warns about oversized or undersized lists

### Automation
- ⏰ **Scheduled runs** — configurable cron via GitHub Actions
- ⭐ **Star trigger** — auto-categorize when you star a new repo
- 🖱️ **Manual trigger** — run on demand with custom parameters
- 📊 **GitHub Issue reports** — detailed run reports as Issues
- 📄 **STARS.md generation** — beautiful categorized star list in your repo

## Quick Start

### 1. Fork / Use Template

Fork this repository or use it as a template.

### 2. Configure Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Required | Description |
|--------|----------|-------------|
| `STAR_GITHUB_TOKEN` | ✅ | GitHub PAT with `repo`, `read:user`, `user` scopes |
| `LLM_BASE_URL` | ✅ | OpenAI-compatible API base URL (e.g., `https://api.openai.com/v1`) |
| `LLM_API_KEY` | ✅ | Your LLM API key |
| `LLM_MODEL` | Optional | Model name — overrides config if set |

> **⚠️ Important**: Use a [Fine-grained PAT](https://github.com/settings/tokens?type=beta) or classic PAT — NOT the default `GITHUB_TOKEN` (it can't manage Star Lists). Required scopes: `repo`, `read:user`, `user`.

### 3. Customize Config (Optional)

Copy an example config and edit it:

```bash
# English config
cp config.example.en.yaml config.yaml

# Or Chinese config
cp config.example.zh.yaml config.yaml
```

Edit `config.yaml` to adjust behavior:

```yaml
llm:
  model: "gpt-4o-mini"   # Model name (can also be overridden via LLM_MODEL secret)
  temperature: 0.3       # Lower = more consistent categorization
  batch_size: 20          # Repos per LLM call
  language: "en"          # Prompt language: "en" or "zh"

categorization:
  max_new_lists: 5        # Max new lists per run
  min_confidence: "medium" # Auto-apply threshold: high, medium, low
  fetch_readme: true      # Two-pass analysis for better accuracy
  dry_run: false          # Set true to preview without changes
  max_repos_per_run: 100  # Limit per run (0 = unlimited)
```

### 4. Run

- **Automatic**: Runs every Monday at 09:00 UTC (configurable in workflow)
- **Manual**: Go to **Actions → StarAutoManager → Run workflow**
- **On star**: Triggers when you star a new repo

## Manual Trigger Options

| Input | Default | Description |
|-------|---------|-------------|
| `dry_run` | `false` | Preview only — no changes applied |
| `max_repos` | `100` | Maximum repos to process |
| `force_recategorize` | `false` | Ignore cache, re-process everything |

## Configuration Reference

<details>
<summary>Full configuration reference</summary>

```yaml
github:
  username: ""              # Leave empty to auto-detect from token

llm:
  # base_url and api_key MUST be set via GitHub Secrets (not here)
  model: "gpt-4o-mini"     # Model name (can also override via LLM_MODEL secret)
  temperature: 0.3          # 0.1-0.4 recommended for stable categorization
  max_tokens: 4096          # Max tokens per LLM call
  batch_size: 20            # Repos per batch
  language: "en"            # "en" or "zh"

categorization:
  max_new_lists: 5          # New lists per run (GitHub max: 32 total)
  min_confidence: "medium"  # Minimum confidence to auto-apply
  fetch_readme: true        # Fetch README for uncertain repos
  dry_run: false            # Preview mode
  ignore_archived: true     # Skip archived repos
  ignore_forks: false       # Skip forked repos
  max_repos_per_run: 100    # 0 = no limit

notifications:
  issue: true               # Create GitHub Issue report
  issue_label: "star-manager"
  summary_in_readme: true   # Generate STARS.md

advanced:
  cache_file: ".star-cache.json"
  rate_limit_buffer: 500    # GraphQL API rate limit headroom
  concurrent_llm_calls: 3   # Parallel LLM requests
  retry_attempts: 3         # API retry count
  retry_delay: 5            # Seconds between retries
```

</details>

## How the LLM Categorization Works

```
User's existing Star Lists
         │
         ▼
┌──────────────────────┐
│  System Prompt:      │
│  "Here are the       │
│   user's lists with  │──→  LLM analyzes patterns:
│   sample repos..."   │     - Granularity level
│                      │     - Domain vs tech vs purpose
└──────────────────────┘     - Naming conventions

Uncategorized repos
         │
         ▼
┌──────────────────────┐
│  User Prompt:        │
│  "Categorize these   │──→  LLM assigns each repo to
│   repos following    │     a list with confidence score
│   the user's style"  │
└──────────────────────┘
         │
         ▼
  Low confidence repos ──→ Fetch README ──→ Second pass
         │
         ▼
  Apply via GraphQL mutations
```

### GitHub Star Lists API

This project uses GitHub's **GraphQL API** to manage Star Lists directly:

- `user.lists` — fetch existing lists and their repos
- `viewer.starredRepositories` — fetch all starred repos
- `createUserList` — create new lists
- `updateUserListsForItem` — assign repos to lists

> **⚠️ Critical**: `updateUserListsForItem` **replaces** all list memberships for a repo. The code always preserves existing memberships when adding new ones.

## Project Structure

```
StarAutoManager/
├── .github/workflows/
│   └── star-manager.yml     # GitHub Actions workflow
├── scripts/
│   ├── __init__.py
│   ├── models.py            # Data models (Repository, StarList, etc.)
│   ├── github_client.py     # GraphQL API client
│   ├── llm_client.py        # LLM categorization engine
│   ├── star_manager.py      # Orchestration pipeline
│   ├── reporter.py          # Issue creation & STARS.md generation
│   └── main.py              # Entry point
├── config.example.en.yaml   # English config template
├── config.example.zh.yaml   # Chinese config template
├── requirements.txt         # Python dependencies
└── README.md
```

## Supported LLM Providers

Any OpenAI-compatible API endpoint works:

| Provider | `LLM_BASE_URL` | Recommended Model |
|----------|-----------------|-------------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| Anthropic (via proxy) | varies | `claude-3-haiku` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3` |
| Any OpenAI-compatible | your endpoint | your model |

## Limitations

- **GitHub Star Lists maximum**: 32 lists per user (GitHub hard limit)
- **GraphQL API rate limit**: 5,000 points/hour — the tool monitors and waits automatically
- **LLM accuracy**: Results depend on model quality; `gpt-4o-mini` or `deepseek-chat` recommended
- **Cold start**: First run without existing lists produces broader categories

## License

MIT

---

# ⭐ StarAutoManager （中文说明）

> 使用 LLM 智能分析你的 Star 分类习惯，自动整理未分类的 GitHub Star。

## 工作原理

StarAutoManager 不是随机分类，而是**先学习你已有的 Star Lists 分类风格**，再用相同的逻辑对未分类仓库进行归类。

### 核心流程

1. 获取你已有的 Star Lists（作为训练数据）
2. 获取所有已 Star 的仓库
3. 找出未分类的仓库
4. LLM 分析你的分类模式（粒度、命名习惯、分类维度）
5. 按照你的风格对未分类仓库进行分类
6. 通过 GitHub GraphQL API 直接应用修改
7. 生成报告（GitHub Issue + STARS.md）

## 快速开始

### 1. Fork 本仓库

### 2. 配置 Secrets

在 **Settings → Secrets and variables → Actions** 中添加：

| Secret | 必需 | 说明 |
|--------|------|------|
| `STAR_GITHUB_TOKEN` | ✅ | GitHub PAT，需要 `repo`、`read:user`、`user` 权限 |
| `LLM_BASE_URL` | ✅ | OpenAI 兼容的 API 地址 |
| `LLM_API_KEY` | ✅ | LLM API 密钥 |
| `LLM_MODEL` | 可选 | 模型名称 — 设置后覆盖配置文件中的 `model` |

### 3. 自定义配置（可选）

复制示例配置文件并编辑：`cp config.example.zh.yaml config.yaml`，支持中文提示语言（`language: "zh"`）。

### 4. 运行

- **自动运行**：每周一 09:00 UTC
- **手动运行**：Actions → StarAutoManager → Run workflow
- **Star 触发**：当你 Star 新仓库时自动触发

## 功能特点

- 🧠 **学习后分类** — 先分析你的分类习惯，再做决定
- 🔄 **增量处理** — 缓存机制避免重复分类
- 🎯 **置信度评分** — 每个分类结果标注 high/medium/low 置信度
- 📖 **二次分析** — 对低置信度仓库获取 README 重新评估
- 🗑️ **过期检测** — 标记已归档或长期未更新的仓库
- 🔄 **重复检测** — 发现功能相似的仓库
- 🏃 **预览模式** — `dry_run: true` 仅预览不修改
