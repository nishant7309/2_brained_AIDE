# AIDE Lab 🧪

**A dual-model fork of [AIDE ML](https://github.com/WecoAI/aideml) with end-to-end autonomous plan review.**

## What's Different?

| Feature | Original AIDE | AIDE Lab |
|---------|---------------|----------|
| **Model Architecture** | Single model for plan+code | **Dual-model**: separate Planner + Coder |
| **Planner Model** | Same as coder | Reasoning model (o1-preview, Gemini Pro) |
| **Coder Model** | Same as planner | Fast model (Claude Sonnet, GPT-4o-mini) |
| **Plan Review** | None (post-execution only) | **Pre-execution**: none, critic, or human |
| **End-to-End Autonomous** | ❌ | ✅ (critic mode with GPT-4o) |
| **Human-in-the-Loop** | ❌ | ✅ (CLI + Web API) |
| **Tree Search** | ✅ | ✅ (unchanged) |

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Planner        │     │  Plan Review     │     │  Coder          │
│  (o1-preview)   │ ──▶ │  (none/critic/   │ ──▶ │  (Claude)       │
│  Creates plan   │     │   human)         │     │  Writes code    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                              ┌────────────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  Execute Code   │
                    │  (Tree Search)  │◀──┐
                    └─────────────────┘   │
                              │           │
                    ┌─────────▼───────────┴──┐
                    │ Debug (Coder) / Improve │
                    └────────────────────────┘
```

## Plan Review Modes

| Mode | Command | Description |
|------|---------|-------------|
| `none` | `agent.plan_review.mode=none` | Execute immediately (fastest) |
| `critic` | `agent.plan_review.mode=critic` | **GPT-4o reviews & improves plans** |
| `human` | `agent.plan_review.mode=human` | CLI/Web approval before execution |

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/aide-lab.git
cd aide-lab
pip install -e .
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."  # For Claude models
export GEMINI_API_KEY="..."             # For Gemini models
```

Or copy `.env.example` to `.env` and fill in your keys.

### 3. Run

```bash
# End-to-end autonomous (recommended)
aide data_dir="example_tasks/house_prices" \
     goal="Predict house prices" \
     agent.plan_review.mode=critic

# With custom models
aide data_dir="example_tasks/house_prices" \
     goal="Predict house prices" \
     agent.planner.model=o1-preview \
     agent.coder.model=claude-3-5-sonnet-20241022 \
     agent.plan_review.mode=critic
```

### Python API

```python
import aide

# End-to-end autonomous
exp = aide.Experiment(
    data_dir="data/",
    goal="Predict house prices",
    eval="RMSE",
    planner_model="o1-preview",
    coder_model="claude-3-5-sonnet-20241022",
    plan_review_mode="critic",  # GPT-4o reviews plans
)
best = exp.run(steps=20)
print(f"Best metric: {best.valid_metric}")
```

### Web API (The Lab)

```bash
uvicorn aide.lab_api:app --reload --port 8000
# Open http://localhost:8000/docs for Swagger UI
```

## Configuration

```yaml
# config.yaml
agent:
  planner:
    model: o1-preview      # Reasoning model for planning
    temp: 1.0
    thinking_level: high
  
  coder:
    model: claude-3-5-sonnet-20241022  # Fast model for coding
    temp: 0.0
  
  plan_review:
    mode: critic           # none | human | critic
    save_plans: true
  
  feedback:
    model: gpt-4.1-mini    # For code review + plan critic
    temp: 0.5
```

## Why Dual-Model?

1. **Cost Optimization**: Use expensive reasoning models only for planning
2. **Quality**: Detailed plans from o1-preview lead to better code
3. **Speed**: Fast models (Claude Sonnet) for rapid DEBUG/IMPROVE iterations
4. **Oversight**: Critic mode catches bad plans before wasting compute

## Credits

This is a fork of [AIDE ML](https://github.com/WecoAI/aideml) by WecoAI. We've extended it with dual-model architecture and plan review capabilities.

## License

Same as original AIDE - see [LICENSE](LICENSE).
