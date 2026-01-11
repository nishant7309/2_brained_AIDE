# AIDE Benchmark Results: Gemini Dual-Brain vs Baseline

This document summarizes the results of running AIDE (AI-Driven Exploration) on the **House Prices** prediction task using differnet Gemini model configurations.

## Configurations Tested

| Configuration | Planner Model | Coder Model | Description |
|---|---|---|---|
| **Baseline** | `gemini-2.0-flash` | `gemini-2.0-flash` | Fast, cost-effective configuration. |
| **Dual-Brain** | `gemini-2.5-pro` | `gemini-2.0-flash` | Uses "Pro" model for high-level reasoning/planning and "Flash" for coding. |

## Results (House Prices Task)

The goal was to minimize **RMSE** (Root Mean Squared Error) on the test set.

| Run Name | Best RMSE | Log Directory | Notes |
|---|---|---|---|
| **Baseline (All Flash)** | **0.1376** | `logs/2-laughing-aboriginal-lynx` | Good performance using XGBoost. |
| **Dual-Brain (Pro + Flash)** | **0.108** | `logs/2-quick-starfish-from-hell` | **SOTA Performance!** Used complex Stacking Ensemble (LightGBM + XGBoost + CatBoost + Ridge). |

## Key Findings

1. **Dual-Brain Superiority**: The Dual-Brain architecture achieved an RMSE of **0.108**, which is significantly better than the baseline's 0.1376. In Kaggle terms, this is often the difference between Top 50% and Top 1%.
2. **Complexity Handling**: The `gemini-2.5-pro` planner was able to propose sophisticated architectures (Stacking Ensembles) and handle complex feature engineering strategies that the Flash model didn't attempt.
3. **Self-Correction**: Both models demonstrated the ability to fix their own code errors, but the Pro-planned strategy was more robust in the long run despite initial implementation bugs.

## generated Files

For each run, the full logs, code, and reports are available in the `logs/` directory.
- **Best Solution Code**: `logs/<run_name>/best_solution.py`
- **Technical Report**: `logs/<run_name>/report.md`
- **Search Tree**: `logs/<run_name>/tree_plot.html`
