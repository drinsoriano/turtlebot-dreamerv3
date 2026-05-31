# DreamerV3 Experiment Dashboard

Streamlit dashboard for monitoring and comparing DreamerV3 TurtleBot3 training runs.

## Setup

```bash
pip install -r dreamerv3-torch/dashboard/requirements.txt
```

## Run

```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501` by default.

## Usage

- **Sidebar** — select one or more runs to display. Runs are discovered automatically from `csv_logs/`.
- **Overview tab** — per-run summary cards and a side-by-side comparison table.
- **Blackbox Charts tab** — per-episode navigation metrics (success rate, collision rate, steps to goal, path directness, obstacle distance, near collisions). Multiple runs are overlaid on the same chart for direct comparison.
- **Whitebox Charts tab** — training diagnostics (eval return, model/actor/value loss, KL, entropy) over training steps.
- **Auto-refresh** — enable in the sidebar to redraw periodically while training is running. Cache TTL is 20 s; data is re-read from disk on each refresh.

## Notes

- The dashboard reads CSVs **read-only** and never touches training files.
- Both `path_efficiency` (old runs) and `path_directness` (new runs) are handled — old column names are silently normalised on load.
- Missing columns (e.g. whitebox file not yet created, or a run that predates a metric) produce an empty caption instead of crashing.
- Each run requires `blackbox_{run_name}.csv` and/or `whitebox_{run_name}.csv` under `csv_logs/`. If only one file exists the other tab simply shows no data.

## Ablation naming convention

Runs follow the pattern `stage{N}_{lidar}_{odometry_mode}_seed{S}`, e.g.:

| run_name                    | odometry mode |
|-----------------------------|---------------|
| `stage1_360_none_seed0`     | none (baseline) |
| `stage1_360_twist_seed0`    | twist           |
| `stage1_360_delta_seed0`    | delta           |
| `stage1_360_full_seed0`     | full            |
