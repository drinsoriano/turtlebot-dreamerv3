"""
DreamerV3 TurtleBot3 — Experiment Dashboard
Run: streamlit run dashboard/app.py  (from dreamerv3-torch/)
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DreamerV3 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_BASE   = Path(__file__).parent.parent / "csv_logs"
CSV_DIR    = CSV_BASE   # active CSV dir; sidebar may switch to a subfolder (e.g. none_stage1, full_stage1, full_imu_stage1, tune_stage1)
PLOTS_BASE = Path(__file__).parent.parent / "path_plots"
PLOTS_DIR  = PLOTS_BASE  # mirrors CSV_DIR: same subfolder name under path_plots/

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

CACHE_TTL = 20  # seconds between cache invalidations

OUTCOME_TEXT_COLORS = {
    "success":   "#2ca02c",
    "collision": "#d62728",
    "timeout":   "#ff7f0e",
}


# ─── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL)
def list_csv_folders() -> list[str]:
    """`"."` (the csv_logs base) plus any immediate subfolders holding run CSVs.

    Every run auto-organizes into csv_logs/{odometry_mode}_stage{N}/ (e.g.
    none_stage1/, full_stage1/, full_imu_stage1/), and BO trials into
    csv_logs/tune_stage{N}_{mode}/ (e.g. tune_stage1_none/, tune_stage1_full_imu/),
    so this surfaces those (and any other run-CSV subfolder) as selectable folders.
    `"."` itself holds only legacy flat CSVs from before per-run foldering.
    """
    folders = ["."]
    if CSV_BASE.exists():
        for d in sorted(CSV_BASE.iterdir()):
            if d.is_dir() and any(
                next(d.glob(f"{pfx}_*.csv"), None) is not None
                for pfx in ("blackbox", "planning", "whitebox")
            ):
                folders.append(d.name)
    return folders


@st.cache_data(ttl=CACHE_TTL)
def scan_runs() -> list[str]:
    if not CSV_DIR.exists():
        return []
    runs: set = set()
    for f in CSV_DIR.glob("blackbox_*.csv"):
        runs.add(f.stem.removeprefix("blackbox_"))
    for f in CSV_DIR.glob("whitebox_*.csv"):
        runs.add(f.stem.removeprefix("whitebox_"))
    for f in CSV_DIR.glob("planning_*.csv"):
        runs.add(f.stem.removeprefix("planning_"))
    return sorted(runs)


def _safe_read_csv(path, **kwargs) -> pd.DataFrame:
    """pd.read_csv with a fallback for live / mid-write CSVs.

    The env writes CSVs while training, and a run resumed after a mid-run schema
    change (e.g. the appended `goal_id` column) leaves a file whose header is
    narrower than its newer rows. Either case makes the fast pandas parser raise
    (`Error tokenizing data` / `line contains NUL`). The fallback strips NUL bytes
    from partial writes and normalises ragged rows — padding short rows and naming
    any extra trailing column(s) — so **all** rows are preserved and dtypes are
    still inferred. Returns an empty frame only if the file is truly unreadable.
    """
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        pass
    try:
        import csv as _csv
        from io import StringIO
        with open(path, "rb") as fh:
            text = fh.read().replace(b"\x00", b"").decode("utf-8", "replace")
        rows = [r for r in _csv.reader(StringIO(text)) if r]
        if not rows:
            return pd.DataFrame()
        header, maxw = rows[0], max(len(r) for r in rows)
        if len(header) < maxw:
            # Known mid-run append is `goal_id`; name a single extra column that,
            # else fall back to positional names so nothing is dropped.
            extra = (["goal_id"] if maxw - len(header) == 1
                     else [f"col_{i}" for i in range(len(header), maxw)])
            header = header + extra
        out = StringIO()
        w = _csv.writer(out)
        w.writerow(header)
        for r in rows[1:]:
            w.writerow((r + [""] * (maxw - len(r)))[:maxw])
        out.seek(0)
        return pd.read_csv(out, **kwargs)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def load_bb(run_name: str) -> pd.DataFrame:
    p = CSV_DIR / f"blackbox_{run_name}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = _safe_read_csv(p)
    if df.empty:
        return df
    df["_idx"] = range(len(df))
    if "path_efficiency" in df.columns and "path_directness" not in df.columns:
        df = df.rename(columns={"path_efficiency": "path_directness"})
    return df


@st.cache_data(ttl=CACHE_TTL)
def load_wb(run_name: str) -> pd.DataFrame:
    p = CSV_DIR / f"whitebox_{run_name}.csv"
    if not p.exists():
        return pd.DataFrame()
    return _safe_read_csv(p)


@st.cache_data(ttl=CACHE_TTL)
def load_tune_trials() -> pd.DataFrame:
    """Load tune_trials_stage*.csv from the active CSV_DIR."""
    if not CSV_DIR.exists():
        return pd.DataFrame()
    files = sorted(CSV_DIR.glob("tune_trials_stage*.csv"))
    if not files:
        return pd.DataFrame()
    try:
        dfs = [_safe_read_csv(f) for f in files]
        dfs = [d for d in dfs if not d.empty]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def load_pl(run_name: str) -> pd.DataFrame:
    p = CSV_DIR / f"planning_{run_name}.csv"
    if not p.exists():
        return pd.DataFrame()
    return _safe_read_csv(p)


@st.cache_data(ttl=CACHE_TTL)
def load_resume_events(logdir_str: str) -> pd.DataFrame:
    """Tiny, path-keyed loader for {logdir}/resume_events.csv (written by
    dreamer.py on every process start — see CLAUDE.md "Crash-Safe Manual Resume
    Workflow"). Unlike the other load_* functions this does NOT read from the
    global CSV_DIR — resume_events.csv lives inside the training --logdir, a
    different filesystem location. The file is at most a handful of rows (one per
    process start), so no row cap is needed and this stays fast.
    """
    p = Path(logdir_str).expanduser() / "resume_events.csv"
    if not p.exists():
        return pd.DataFrame()
    return _safe_read_csv(p)


# ─── Cross-folder loaders (used by the Compare / Convergence tabs) ────────────
# The per-run tabs read from the single sidebar-selected CSV_DIR. The Compare and
# Convergence tabs instead overlay runs from *different* auto-organized folders
# (e.g. none_stage4/ vs full_imu_stage4/), so they discover and load by explicit
# (folder, run) — independent of CSV_DIR.

def _folder_base(folder: str) -> Path:
    return CSV_BASE if folder == "." else CSV_BASE / folder


@st.cache_data(ttl=CACHE_TTL)
def scan_all_runs() -> list[tuple[str, str, str]]:
    """(folder, run, label) across every CSV folder. label = run for the '.' base,
    else 'folder/run'. Excludes eval_* stems — the eval variant of a run is reached
    via the phase toggle, not listed as its own run."""
    out: list[tuple[str, str, str]] = []
    for folder in list_csv_folders():
        base, runs = _folder_base(folder), set()
        for pfx in ("blackbox", "whitebox", "planning"):
            for f in base.glob(f"{pfx}_*.csv"):
                stem = f.stem.removeprefix(f"{pfx}_")
                if not stem.startswith("eval_"):
                    runs.add(stem)
        for r in sorted(runs):
            out.append((folder, r, r if folder == "." else f"{folder}/{r}"))
    return out


@st.cache_data(ttl=CACHE_TTL)
def load_bb_in(folder: str, run_name: str) -> pd.DataFrame:
    """Folder-aware twin of load_bb (reads {folder}/blackbox_{run}.csv)."""
    p = _folder_base(folder) / f"blackbox_{run_name}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = _safe_read_csv(p)
    if df.empty:
        return df
    df["_idx"] = range(len(df))
    if "path_efficiency" in df.columns and "path_directness" not in df.columns:
        df = df.rename(columns={"path_efficiency": "path_directness"})
    return df


@st.cache_data(ttl=CACHE_TTL)
def load_pl_in(folder: str, run_name: str) -> pd.DataFrame:
    """Folder-aware twin of load_pl (reads {folder}/planning_{run}.csv)."""
    p = _folder_base(folder) / f"planning_{run_name}.csv"
    return _safe_read_csv(p) if p.exists() else pd.DataFrame()


RESOURCE_NUMERIC_COLS = [
    "episode", "episode_wall_time_sec", "total_wall_time_sec",
    "cpu_percent_process", "ram_used_mb_process", "gpu_memory_used_mb_process",
    "cpu_percent_system", "ram_percent_system",
    "gpu_util_percent_device", "gpu_memory_used_mb_device",
    "gpu_memory_total_mb", "gpu_memory_percent_device",
    "gpu_power_watts", "gpu_temperature_c",
    "lidar_beams", "depth_camera_count",
]


@st.cache_data(ttl=CACHE_TTL)
def load_resource(run_name: str) -> pd.DataFrame:
    p = CSV_DIR / f"resource_{run_name}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = _safe_read_csv(p, dtype=str)  # str to avoid mixed-type on blank GPU fields
    if df.empty:
        return df
    for col in RESOURCE_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=CACHE_TTL)
def load_resource_eval(run_name: str) -> pd.DataFrame:
    if run_name.startswith("eval_"):
        return pd.DataFrame()  # already an eval run; avoid resource_eval_eval_* lookup
    p = CSV_DIR / f"resource_eval_{run_name}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = _safe_read_csv(p, dtype=str)
    if df.empty:
        return df
    for col in RESOURCE_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─── Stat helpers ─────────────────────────────────────────────────────────────

def _last(df: pd.DataFrame, col: str):
    if df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else None


def _mean(df: pd.DataFrame, col: str, mask=None):
    if df.empty or col not in df.columns:
        return None
    s = df[col] if mask is None else df.loc[mask, col]
    s = s.dropna()
    return float(s.mean()) if not s.empty else None


def _median(df: pd.DataFrame, col: str, mask=None):
    if df.empty or col not in df.columns:
        return None
    s = df[col] if mask is None else df.loc[mask, col]
    s = s.dropna()
    return float(s.median()) if not s.empty else None


def _fmt(val, digits: int = 2, suffix: str = "") -> str:
    return f"{val:.{digits}f}{suffix}" if val is not None else "—"


def _clr(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


# ─── Outcome metrics vs path metrics ──────────────────────────────────────────
# Path/trajectory metrics stay numerically defined on FAILED episodes — a robot
# that collides 1 m from the start still produces a path-efficiency number, and a
# short doomed path can score *higher* than a real solved detour. Summarising them
# over all outcomes is therefore misleading, so every reported path-metric mean /
# rolling line is success-only by default.
# Outcome + safety metrics (success/collision/timeout rates, obstacle distance,
# near-collisions) must instead use EVERY episode — filtering them to successes
# would make them meaningless (a success-only "success rate" is always 100%).
# See CLAUDE.md "Metrics for Final Analysis".
_PATH_METRIC_COLS = {
    "planner_path_efficiency_hybrid",
    "planner_path_efficiency",
    "planner_path_efficiency_center",
    "path_directness",
    "local_efficiency",
    "actual_path_length",
    "episode_steps",
    "steps_to_goal",
}


def _is_path_metric(col: str) -> bool:
    return col in _PATH_METRIC_COLS


# Each planner variant carries its own status column; using the plain
# `planner_status` for the hybrid/centre metric would validate the wrong planner.
_PLANNER_STATUS_COL = {
    "planner_path_efficiency_hybrid": "planner_status_hybrid",
    "planner_path_efficiency_center": "planner_status_center",
    "planner_path_efficiency":        "planner_status",
}


def _planner_valid_mask(df: pd.DataFrame, col: str, success_only: bool = True):
    """Rows usable for a planning-CSV metric: the matching planner status is `ok`
    AND (for path metrics, when success_only) the episode actually succeeded.
    Status alone is not enough — a collision episode can still have a perfectly
    valid plan and a flatteringly high efficiency number."""
    if df is None or df.empty:
        return None
    status_col = _PLANNER_STATUS_COL.get(col, "planner_status")
    if status_col not in df.columns and "planner_status" in df.columns:
        status_col = "planner_status"
    ok = (df[status_col] == "ok") if status_col in df.columns \
        else pd.Series([True] * len(df), index=df.index)
    if success_only and _is_path_metric(col):
        sm = _success_mask(df)
        if sm is not None:
            ok = ok & sm
    return ok


def _success_mask(df: pd.DataFrame):
    """Boolean success mask. Prefers the explicit `outcome` label — it exists in
    BOTH blackbox and planning frames (so this works for planner metrics too) and
    is correct under any reward mode. Falls back to the blackbox-only
    steps_to_goal == -1 sentinel for pre-outcome CSVs."""
    if df is None or df.empty:
        return None
    if "outcome" in df.columns:
        return df["outcome"].astype(str) == "success"
    if "steps_to_goal" in df.columns:
        return df["steps_to_goal"] != -1
    return None


# ─── Perf helpers: row-capped tables + chart-trace downsampling ───────────────
# Neither changes any metric/formula — purely how much is shipped to the browser.

def _capped_table(df: pd.DataFrame, key: str, default_n: int = 50) -> pd.DataFrame:
    """Show only the last `default_n` rows by default; a checkbox opts into the
    full table. Caller must reuse the RETURNED frame for any positional (.iloc)
    follow-up logic (e.g. click-to-select), not the original df — the returned
    frame is what's actually rendered."""
    if len(df) <= default_n:
        return df
    show_full = st.checkbox(
        f"Show full table ({len(df)} rows)", value=False, key=f"{key}_full",
    )
    return df if show_full else df.tail(default_n)


def _thin(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    """Stride-sample a raw scatter/marker trace above max_points so large runs
    (thousands of episodes) don't ship one point per row to the browser. Only
    ever applied to raw traces — rolling-mean lines are computed on the full
    series first, then thinned the same way, so trend shape is unaffected."""
    n = len(df)
    return df.iloc[:: max(1, n // max_points)] if n > max_points else df


def _convergence_point(values: pd.Series, tol: float, tail_frac: float = 0.25):
    """Earliest index where the (already-rolled) series enters and STAYS within
    ±tol of the plateau (median of the last tail_frac of the series).
    Returns (conv_label_index, plateau). 'stays' = >=80% of remaining points
    are inside the band (robust to the noisy efficiency curve)."""
    v = values.dropna()
    if len(v) < 10:
        return None, None
    plateau = v.iloc[int(len(v) * (1 - tail_frac)):].median()
    within = ((v - plateau).abs() <= tol).to_numpy()
    for i in range(len(within)):
        if within[i] and within[i:].mean() >= 0.8:
            return v.index[i], plateau
    return v.index[-1], plateau


# ─── Path plot helpers ────────────────────────────────────────────────────────

def _find_png(run_name: str, episode: int, outcome: str) -> Path | None:
    # eval CSV run names are "eval_{base}"; plot subfolders are "{base}_eval/"
    if run_name.startswith("eval_"):
        plot_dir = run_name[len("eval_"):] + "_eval"
    else:
        plot_dir = run_name
    p = PLOTS_DIR / plot_dir / f"ep{episode:05d}_{outcome}.png"
    return p if p.exists() else None


def _color_outcome_text(val: object) -> str:
    color = OUTCOME_TEXT_COLORS.get(str(val), "")
    return f"color: {color}; font-weight: bold" if color else ""


# ─── Chart builders ───────────────────────────────────────────────────────────

def _bb_chart(
    run_map: dict,
    col: str,
    title: str,
    extra: list | None = None,
    success_only: bool = False,
    show_cumulative: bool = True,
    show_r100: bool = True,
    show_r500: bool = True,
) -> go.Figure | None:
    fig = go.Figure()
    has_data = False
    for i, (run, df) in enumerate(run_map.items()):
        if df.empty or col not in df.columns:
            continue
        if success_only:
            sm = _success_mask(df)
            d = df[sm] if sm is not None else df
        else:
            d = df
        if d.empty:
            continue
        d = _thin(d)  # long runs (thousands of episodes) ship fewer points to the browser
        clr = _clr(i)
        if show_cumulative:
            fig.add_trace(go.Scatter(
                x=d["_idx"], y=d[col], mode="lines",
                name=run, line=dict(color=clr, width=1.5),
                legendgroup=run,
            ))
            has_data = True
        for ecol, dash, suffix in (extra or []):
            if "100" in ecol and not show_r100:
                continue
            if "500" in ecol and not show_r500:
                continue
            if ecol not in d.columns:
                continue
            sub = d[["_idx", ecol]].dropna(subset=[ecol])
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["_idx"], y=sub[ecol], mode="lines",
                name=f"{run} {suffix}",
                line=dict(color=clr, dash=dash, width=1),
                opacity=0.6, legendgroup=run,
            ))
            has_data = True
    if not has_data:
        return None
    fig.update_layout(
        title=title, xaxis_title="Episode",
        height=360, margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(font_size=10),
    )
    return fig


def _wb_chart(
    run_map: dict,
    cols: str | list,
    title: str,
) -> go.Figure | None:
    if isinstance(cols, str):
        cols = [cols]
    dashes = ["solid", "dash", "dot", "dashdot"]
    fig = go.Figure()
    has_data = False
    for i, (run, df) in enumerate(run_map.items()):
        if df.empty or "step" not in df.columns:
            continue
        clr = _clr(i)
        for j, col in enumerate(cols):
            if col not in df.columns:
                continue
            sub = df[["step", col]].dropna(subset=[col])
            if sub.empty:
                continue
            lbl = f"{run} — {col}" if len(cols) > 1 else run
            fig.add_trace(go.Scatter(
                x=sub["step"], y=sub[col], mode="lines",
                name=lbl,
                line=dict(color=clr, dash=dashes[j % 4], width=1.5),
                legendgroup=run,
            ))
            has_data = True
    if not has_data:
        return None
    fig.update_layout(
        title=title, xaxis_title="Training Step",
        height=360, margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(font_size=10),
    )
    return fig


def _resource_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
) -> go.Figure | None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    has_phase = "_phase" in df.columns
    cols_needed = [x_col, y_col] + (["_phase"] if has_phase else [])
    sub = df[cols_needed].copy()
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna(subset=[y_col])
    if sub.empty:
        return None
    fig = go.Figure()
    phases = sub["_phase"].unique().tolist() if has_phase else [None]
    for i, ph in enumerate(phases):
        part = sub[sub["_phase"] == ph] if ph is not None else sub
        part = _thin(part)  # long runs ship fewer points to the browser
        label = f"{y_col} ({ph})" if ph else y_col
        fig.add_trace(go.Scatter(
            x=part[x_col], y=part[y_col], mode="lines",
            name=label, line=dict(color=_clr(i), width=1.5),
        ))
    fig.update_layout(
        title=title, xaxis_title=x_col,
        height=300, margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(font_size=10),
    )
    return fig


# ─── Page sections ────────────────────────────────────────────────────────────

def _section_summary(selected: list, bb_data: dict, wb_data: dict):
    st.subheader("Per-Run Summary")
    for run in selected:
        bb = bb_data[run]
        wb = wb_data[run]
        sm = _success_mask(bb)

        with st.expander(f"**{run}**", expanded=True):
            row1 = st.columns(4)
            row2 = st.columns(4)

            items_r1 = [
                ("Success Rate",     _fmt(_last(bb, "success_rate"),   1, "%")),
                ("Collision Rate",   _fmt(_last(bb, "collision_rate"),  1, "%")),
                ("Median Steps ✓",   _fmt(_median(bb, "steps_to_goal", sm), 1)),
                ("Median Path Dir.", _fmt(_median(bb, "path_directness", sm), 3)),
            ]
            items_r2 = [
                ("Avg Obs. Dist.",   _fmt(_mean(bb, "min_obstacle_dist"),  3, " m")),
                ("Avg Near Col.",    _fmt(_mean(bb, "near_collisions"),    2)),
                ("Eval Return",      _fmt(_last(wb, "eval_return"),        3)),
                ("Model Loss",       _fmt(_last(wb, "model_loss"),         4)),
            ]
            for c, (lbl, val) in zip(row1, items_r1):
                c.metric(lbl, val)
            for c, (lbl, val) in zip(row2, items_r2):
                c.metric(lbl, val)


def _section_comparison(selected: list, bb_data: dict, wb_data: dict):
    st.subheader("Comparison Table")
    rows = []
    for run in selected:
        bb = bb_data[run]
        wb = wb_data[run]
        sm = _success_mask(bb)

        eval_ret = _last(wb, "eval_return")
        mdl_loss = _last(wb, "model_loss")

        rows.append({
            "run_name":             run,
            "odometry_mode":        bb["odometry_mode"].iloc[0] if not bb.empty and "odometry_mode" in bb.columns else "—",
            "stage":                int(bb["stage"].iloc[0])    if not bb.empty and "stage" in bb.columns else "—",
            "total_episodes":       len(bb),
            "final_success_%":      round(_last(bb, "success_rate")  or 0, 2),
            "final_collision_%":    round(_last(bb, "collision_rate") or 0, 2),
            "median_steps_to_goal": round(_median(bb, "steps_to_goal",   sm) or 0, 1),
            "median_path_dir.":     round(_median(bb, "path_directness", sm) or 0, 3),
            "latest_eval_return":   round(eval_ret, 3) if eval_ret is not None else None,
            "latest_model_loss":    round(mdl_loss, 4) if mdl_loss is not None else None,
        })
    if rows:
        comparison_df = pd.DataFrame(rows)
        display_comparison = comparison_df.copy()
        for col in display_comparison.select_dtypes(include="object").columns:
            display_comparison[col] = display_comparison[col].astype(str)
        st.dataframe(display_comparison, width="stretch", hide_index=True)


def _section_eval_checkpoints(selected: list, n_per_ckpt: int):
    """Eval Checkpoint Summary — last and best checkpoint performance."""
    st.subheader("Eval Checkpoint Summary")
    st.caption(
        "Slices `blackbox_eval_*.csv` and `planning_eval_*.csv` into fixed-size blocks "
        "(one block = one eval checkpoint). "
        "**Last** = final trained policy. "
        "**Best** = checkpoint with highest eval success rate (peak capability)."
    )

    summary_rows = []

    for run in selected:
        eval_key = f"eval_{run}"
        bb_eval = load_bb(eval_key)
        pl_eval = load_pl(eval_key)

        if bb_eval.empty:
            st.caption(f"*{run} — no eval data (`blackbox_eval_{run}.csv` not found or empty)*")
            continue

        n_total = len(bb_eval)
        n_ckpts = max(1, n_total // n_per_ckpt)

        def _chunk_stats(bb_chunk: pd.DataFrame, pl_chunk: pd.DataFrame) -> dict:
            if bb_chunk.empty:
                return {}
            n = len(bb_chunk)
            outcomes = bb_chunk["outcome"] if "outcome" in bb_chunk.columns else pd.Series(dtype=str)
            suc_pct  = round((outcomes == "success").sum()   / n * 100, 1)
            col_pct  = round((outcomes == "collision").sum() / n * 100, 1)
            mean_ppe = None
            if not pl_chunk.empty and "planner_path_efficiency" in pl_chunk.columns:
                ok = pl_chunk[pl_chunk["planner_status"] == "ok"] \
                    if "planner_status" in pl_chunk.columns else pl_chunk
                vals = pd.to_numeric(ok["planner_path_efficiency"], errors="coerce").dropna()
                if not vals.empty:
                    mean_ppe = round(float(vals.mean()), 4)
            return {"n": n, "success_%": suc_pct, "collision_%": col_pct, "mean_ppe": mean_ppe}

        # Last checkpoint
        last_bb   = bb_eval.tail(n_per_ckpt)
        last_pl   = pl_eval.tail(n_per_ckpt) if not pl_eval.empty else pd.DataFrame()
        last_stat = _chunk_stats(last_bb, last_pl)

        # Best checkpoint — scan all complete blocks
        best_idx  = 0
        best_suc  = -1.0
        for i in range(n_ckpts):
            chunk = bb_eval.iloc[i * n_per_ckpt:(i + 1) * n_per_ckpt]
            if chunk.empty or "outcome" not in chunk.columns:
                continue
            s = (chunk["outcome"] == "success").sum() / len(chunk) * 100
            if s > best_suc:
                best_suc = s
                best_idx = i
        best_bb   = bb_eval.iloc[best_idx * n_per_ckpt:(best_idx + 1) * n_per_ckpt]
        best_pl   = pl_eval.iloc[best_idx * n_per_ckpt:(best_idx + 1) * n_per_ckpt] \
                    if not pl_eval.empty else pd.DataFrame()
        best_stat = _chunk_stats(best_bb, best_pl)

        with st.expander(
            f"**{run}** — {n_total} eval eps · {n_ckpts} checkpoint(s)",
            expanded=True,
        ):
            c_last, c_best = st.columns(2)

            with c_last:
                st.markdown("**Last checkpoint** *(final trained policy)*")
                if last_stat:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Success",    f"{last_stat['success_%']}%")
                    m2.metric("Collision",  f"{last_stat['collision_%']}%")
                    m3.metric("Mean PPE",
                              _fmt(last_stat["mean_ppe"], 4) if last_stat["mean_ppe"] else "—")

            with c_best:
                st.markdown(
                    f"**Best checkpoint** *(ckpt #{best_idx + 1} of {n_ckpts}, "
                    f"by success rate)*"
                )
                if best_stat:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Success",    f"{best_stat['success_%']}%")
                    m2.metric("Collision",  f"{best_stat['collision_%']}%")
                    m3.metric("Mean PPE",
                              _fmt(best_stat["mean_ppe"], 4) if best_stat["mean_ppe"] else "—")

        summary_rows.append({
            "run":               run,
            "total_eval_eps":    n_total,
            "n_checkpoints":     n_ckpts,
            "last_success_%":    last_stat.get("success_%"),
            "last_collision_%":  last_stat.get("collision_%"),
            "last_mean_ppe":     last_stat.get("mean_ppe"),
            "best_ckpt#":        best_idx + 1,
            "best_success_%":    best_stat.get("success_%"),
            "best_collision_%":  best_stat.get("collision_%"),
            "best_mean_ppe":     best_stat.get("mean_ppe"),
        })

    if len(summary_rows) > 1:
        st.divider()
        st.markdown("**Cross-run comparison**")
        cmp_df = pd.DataFrame(summary_rows)
        for col in cmp_df.select_dtypes(include="object").columns:
            cmp_df[col] = cmp_df[col].astype(str)
        st.dataframe(cmp_df, width="stretch", hide_index=True, key="eval_ckpt_cmp")


def _section_bb(selected: list, bb_data: dict, show_cumulative: bool, show_r100: bool, show_r500: bool):
    run_map = {r: bb_data[r] for r in selected}
    specs = [
        ("success_rate",      "Success Rate (%)",
         [("rolling_success_rate_100", "dash", "rolling-100"),
          ("rolling_success_rate_500", "dot",  "rolling-500")], False),
        ("collision_rate",    "Collision Rate (%)",
         [("rolling_collision_rate_100", "dash", "rolling-100"),
          ("rolling_collision_rate_500", "dot",  "rolling-500")], False),
        # Path/trajectory metrics -> success-only (see _PATH_METRIC_COLS: a failed
        # episode still yields a number, and a short doomed path can look "good").
        ("steps_to_goal",     "Steps to Goal — success-only",              None, True),
        ("episode_steps",     "Episode Steps — success-only",              None, True),
        ("path_directness",   "Path Directness  [0 – 1] — success-only",   None, True),
        ("local_efficiency",  "Local Efficiency  [−1 – 1] — success-only", None, True),
        # Safety/outcome metrics -> every episode (filtering them would distort them).
        ("min_obstacle_dist", "Min Obstacle Distance (m)  (all episodes)",  None, False),
        ("near_collisions",   "Near Collisions  (all episodes)",            None, False),
    ]
    for col, title, extra, success_only in specs:
        fig = _bb_chart(run_map, col, title, extra, success_only,
                        show_cumulative=show_cumulative,
                        show_r100=show_r100,
                        show_r500=show_r500)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"bb_{col}")
        else:
            st.caption(f"*{title} — column not available in selected runs*")

    # ── Raw per-episode black-box table ──────────────────────────────────────
    st.divider()
    st.subheader("Black-box Episode Log")
    st.caption(
        "Per-episode training metrics straight from `blackbox_{run}.csv`. "
        "`steps_to_goal` is the step count for **successful** episodes only "
        "(−1 for collision / timeout). Click a column header to sort."
    )
    exclude_invalid = st.checkbox(
        "Exclude non-terminal / hang-affected rows",
        value=False, key="bb_exclude_invalid",
        help="Drops rows with an unrecognised outcome or episode_steps <= 0 — a "
             "belt-and-suspenders filter for logging affected by a mid-episode hang. "
             "Off by default so no data is hidden without asking.",
    )
    for run in selected:
        df = run_map.get(run)
        if df is None or df.empty:
            st.caption(f"*{run} — no black-box data*")
            continue
        if exclude_invalid:
            mask = pd.Series(True, index=df.index)
            if "outcome" in df.columns:
                mask &= df["outcome"].isin(["success", "collision", "timeout"])
            if "episode_steps" in df.columns:
                mask &= pd.to_numeric(df["episode_steps"], errors="coerce").fillna(0) > 0
            df = df[mask]
            if df.empty:
                st.caption(f"*{run} — no rows left after excluding invalid ones*")
                continue
        if len(selected) > 1:
            st.markdown(f"**{run}**")
        show = df.drop(columns=[c for c in ["_idx"] if c in df.columns])
        show = _capped_table(show, key=f"bb_tbl_{run}")
        # Styler (outcome coloring) is fine for modest logs; for very long runs
        # fall back to a plain frame to keep rendering snappy.
        use_style = "outcome" in show.columns and len(show) <= 1500
        try:
            if use_style:
                try:
                    styled = show.style.map(_color_outcome_text, subset=["outcome"])
                except AttributeError:
                    styled = show.style.applymap(_color_outcome_text, subset=["outcome"])
                st.dataframe(styled, width="stretch", hide_index=True,
                             key=f"bb_tbl_{run}")
            else:
                st.dataframe(show, width="stretch", hide_index=True,
                             key=f"bb_tbl_{run}")
        except Exception:
            disp = show.copy()
            for c in disp.select_dtypes(include="object").columns:
                disp[c] = disp[c].astype(str)
            st.dataframe(disp, width="stretch", hide_index=True,
                         key=f"bb_tbl_fb_{run}")


def _section_wb(selected: list, wb_data: dict):
    run_map = {r: wb_data[r] for r in selected}
    specs = [
        ("eval_return",                                "Eval Return"),
        (["eval_success_rate", "eval_collision_rate"], "Eval Success & Collision Rate (%)"),
        ("model_loss",                                 "Model Loss"),
        (["actor_loss", "value_loss"],                 "Actor & Value Loss"),
        (["kl", "prior_ent", "post_ent"],              "KL Divergence & Entropy"),
        # Diagnostics added 2026-06-12 (skip silently if column absent in older CSVs)
        ("actor_entropy",                              "Actor Entropy (policy-collapse signal)"),
        (["model_grad_norm", "actor_grad_norm", "value_grad_norm"], "Gradient Norms (instability spikes)"),
        (["reward_loss", "dyn_loss", "rep_loss"],      "World-Model Losses (reward / dyn / rep)"),
        (["reward_variance", "reward_variance_100"],   "Reward Variance (cumulative vs rolling-100)"),
    ]
    for cols, title in specs:
        fig = _wb_chart(run_map, cols, title)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"wb_{title}")
            if cols == ["eval_success_rate", "eval_collision_rate"]:
                st.caption(
                    "Scored from terminal outcome labels — correct under shaped "
                    "reward, and timeouts are excluded from the collision count. "
                    "For the authoritative per-episode breakdown use the Black-box "
                    "tab (blackbox_eval). Rows logged before the 2026-06-09 fix may "
                    "read 0% in shaped-reward runs."
                )
            elif cols == "actor_entropy":
                st.caption(
                    "Policy entropy. A healthy run stays well above 0; a collapse "
                    "toward ~0 means the policy stopped exploring (often the cause of "
                    "looping / getting stuck). Watch for it dropping too early."
                )
            elif cols == ["model_grad_norm", "actor_grad_norm", "value_grad_norm"]:
                st.caption(
                    "Gradient L2 norms. Sudden spikes = training instability "
                    "(divergence). Steady, bounded values are healthy."
                )
            elif cols == ["reward_loss", "dyn_loss", "rep_loss"]:
                st.caption(
                    "World-model diagnostics. `reward_loss` high ⇒ the model can't "
                    "predict reward (planning degrades). `dyn`/`rep` = the KL split "
                    "between dynamics and representation (posterior-collapse check)."
                )
            elif cols == ["reward_variance", "reward_variance_100"]:
                st.caption(
                    "`reward_variance` is cumulative (all episodes — stays high "
                    "forever because early chaos is never dropped). "
                    "`reward_variance_100` is the rolling last-100 window — the "
                    "honest 'is it stable *now*?' signal."
                )
        else:
            st.caption(f"*{title} — no data available*")


def _section_resource(run_name: str, res_df: pd.DataFrame, res_eval_df: pd.DataFrame):
    # Combine train + eval phases into one DataFrame with a _phase label
    frames = []
    if not res_df.empty:
        tmp = res_df.copy(); tmp["_phase"] = "train"; frames.append(tmp)
    if not res_eval_df.empty:
        tmp = res_eval_df.copy(); tmp["_phase"] = "eval"; frames.append(tmp)

    if not frames:
        st.info(
            f"No resource CSV found for **{run_name}**. "
            "Enable with `--resource_logging True` when running `dreamer.py`."
        )
        return

    df = pd.concat(frames, ignore_index=True)

    # Determine x-axis column (future-proof: prefer session_episode if added later)
    x_col = next(
        (c for c in ["session_episode", "global_episode", "episode"] if c in df.columns),
        None,
    )
    if x_col is None:
        st.warning("No episode column found in resource CSV.")
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            if "session_id" in df.columns:
                sessions = sorted(df["session_id"].dropna().unique().tolist())
                sel_sessions = st.multiselect(
                    "Session ID", sessions, default=sessions,
                    key=f"res_sess_{run_name}",
                )
                if sel_sessions:
                    df = df[df["session_id"].isin(sel_sessions)]

        with col_f2:
            phases = sorted(df["_phase"].unique().tolist())
            if len(phases) > 1:
                sel_phase = st.radio(
                    "Phase", ["All"] + phases, key=f"res_phase_{run_name}"
                )
                if sel_phase != "All":
                    df = df[df["_phase"] == sel_phase]
            else:
                st.caption(f"Phase: **{phases[0]}**")

        with col_f3:
            ep_vals = df[x_col].dropna() if x_col in df.columns else pd.Series(dtype=float)
            if not ep_vals.empty:
                ep_min = int(ep_vals.min())
                ep_max = int(ep_vals.max())
                if ep_min < ep_max:
                    ep_range = st.slider(
                        "Episode range", ep_min, ep_max, (ep_min, ep_max),
                        key=f"res_ep_{run_name}",
                    )
                    df = df[df[x_col].between(*ep_range)]

    if df.empty:
        st.warning("No data after filtering.")
        return

    # ── Summary helpers (work on filtered df) ─────────────────────────────────
    def _rval(col, fn):
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(fn(s)) if not s.empty else None

    # ── Summary cards ─────────────────────────────────────────────────────────
    st.subheader("Resource Summary")
    r1 = st.columns(5)
    r2 = st.columns(4)
    r1[0].metric("Episodes logged",        str(len(df)))
    r1[1].metric("Mean ep. time (s)",      _fmt(_rval("episode_wall_time_sec", lambda s: s.mean()), 2))
    r1[2].metric("Total wall time (s)",    _fmt(_rval("total_wall_time_sec",   lambda s: s.max()),  0))
    r1[3].metric("Mean CPU % (process)",   _fmt(_rval("cpu_percent_process",   lambda s: s.mean()), 1, "%"))
    r1[4].metric("Peak RAM MB (process)",  _fmt(_rval("ram_used_mb_process",   lambda s: s.max()),  1))
    r2[0].metric("Mean GPU mem MB (proc)", _fmt(_rval("gpu_memory_used_mb_process", lambda s: s.mean()), 1))
    r2[1].metric("Peak GPU mem MB (proc)", _fmt(_rval("gpu_memory_used_mb_process", lambda s: s.max()),  1))
    r2[2].metric("Mean GPU power (W)",     _fmt(_rval("gpu_power_watts",            lambda s: s.mean()), 1))
    r2[3].metric("Peak GPU temp (°C)",     _fmt(_rval("gpu_temperature_c",          lambda s: s.max()),  1))

    # ── Trend charts ──────────────────────────────────────────────────────────
    st.subheader("Resource Trends")
    st.caption(
        "**Process metrics** = DreamerV3 Python process only (use these for thesis reporting).  \n"
        "**Device metrics** = GPU device context — includes ALL GPU users "
        "(Xorg ≈370 MB, gnome-shell ≈70 MB, VSCode ≈88 MB, Chrome ≈55 MB, "
        "gzserver/gzclient ≈29 MB).  \n"
        "**`cpu_percent_process` can exceed 100%** — psutil sums across all CPU cores; "
        "PyTorch uses multiple threads so 150–400% is normal during GPU training.  \n"
        "**High CPU with CUDA is expected** — GPU handles only the model forward/backward pass; "
        "ROS2 callbacks, episode I/O, and kernel dispatch still run on CPU."
    )
    chart_specs = [
        ("episode_wall_time_sec",       "Episode Wall Time (s)  [process]"),
        ("cpu_percent_process",         "CPU %  [process — sum over all cores; >100% = multi-core]"),
        ("ram_used_mb_process",         "RAM Used MB  [process]"),
        ("gpu_memory_used_mb_process",  "GPU Memory MB  [process — primary, thesis metric]"),
        ("gpu_memory_used_mb_device",   "GPU Memory MB  [device — inflated by desktop ~600–1100 MB]"),
        ("gpu_power_watts",             "GPU Power (W)  [device context]"),
        ("gpu_temperature_c",           "GPU Temperature (°C)  [device context]"),
    ]
    for y_col, title in chart_specs:
        fig = _resource_chart(df, x_col, y_col, title)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            key=f"res_{run_name}_{y_col}")
        else:
            st.caption(f"*{title} — not available or all values blank*")

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("Raw resource log", expanded=False):
        display_df = df.drop(columns=["_phase"], errors="ignore").copy()
        for col in display_df.select_dtypes(include="object").columns:
            display_df[col] = display_df[col].astype(str)
        display_df = _capped_table(display_df, key=f"res_raw_{run_name}")
        st.dataframe(display_df, width="stretch", key=f"res_raw_tbl_{run_name}")


# Per-goal drill-down metrics: label -> (source, column). "_success01" is derived
# on the fly from outcome (its rolling mean = the per-goal success rate over time).
_GOAL_METRICS = {
    "Path Efficiency (Hybrid-A*)": ("planning", "planner_path_efficiency_hybrid"),
    "Path Efficiency (A*)":   ("planning", "planner_path_efficiency"),
    "Success (per episode)":  ("blackbox", "_success01"),
    "Actual Path Length (m)": ("planning", "actual_path_length"),
    "Episode Steps":          ("blackbox", "episode_steps"),
    "Path Directness":        ("blackbox", "path_directness"),
    "Local Efficiency":       ("blackbox", "local_efficiency"),
    "Min Obstacle Dist (m)":  ("blackbox", "min_obstacle_dist"),
    "Near Collisions":        ("blackbox", "near_collisions"),
}
# Metrics bounded to [0, 1] — fix the y-axis for these so trends are comparable.
_GOAL_METRICS_UNIT = {"planner_path_efficiency_hybrid", "planner_path_efficiency",
                      "_success01", "path_directness"}


def _section_goal_outcome_trend(run_name: str, bb_df: pd.DataFrame, goal: str,
                                roll_window: int) -> None:
    """Rolling success / collision / timeout rate for one goal. Deliberately uses
    EVERY episode (no success filter) — this is the outcome half of the split
    described in _PATH_METRIC_COLS, and filtering it would be meaningless."""
    if bb_df is None or bb_df.empty or "outcome" not in bb_df.columns \
            or "goal_id" not in bb_df.columns or "episode" not in bb_df.columns:
        return
    g = bb_df[bb_df["goal_id"].astype(str) == str(goal)].copy()
    if g.empty:
        return
    g = g.sort_values("episode")
    st.markdown("**Outcome trend for this goal** — all episodes (no success filter)")
    fig = go.Figure()
    for oc, oc_color in OUTCOME_TEXT_COLORS.items():
        ind = (g["outcome"].astype(str) == oc).astype(float)
        if ind.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=g["episode"], y=ind.rolling(roll_window, min_periods=1).mean() * 100,
            mode="lines", name=f"{oc} rate", line=dict(width=2, color=oc_color),
        ))
    fig.update_layout(
        xaxis_title="Episode", yaxis_title=f"Rolling-{roll_window} rate (%)",
        yaxis=dict(range=[0, 105]), height=300,
        margin=dict(l=50, r=20, t=30, b=40), legend=dict(font_size=10),
    )
    st.plotly_chart(fig, use_container_width=True,
                    key=f"plan_goaloutcome_{run_name}")
    st.caption(
        f"Computed over all {len(g)} appearances of this goal — success, collision "
        "and timeout rates always include every episode."
    )


def _section_goal_learning(run_name: str, pl_df: pd.DataFrame, bb_df: pd.DataFrame,
                           roll_window: int) -> None:
    """🔍 Per-goal learning — pick one goal_id and watch a chosen metric over its
    episode sequence: is the robot improving on the *same* goal (shorter paths,
    more reliable), or regressing (colliding again after earlier successes)?"""
    # goal_id may live in pl_df and/or bb_df; use whichever has non-blank values.
    def _gids(df):
        if df is None or df.empty or "goal_id" not in df.columns:
            return pd.Series(dtype=str)
        s = df["goal_id"].astype(str)
        return s[s.str.strip().ne("") & s.ne("nan") & s.ne("None")]

    gids = pd.concat([_gids(pl_df), _gids(bb_df)], ignore_index=True)
    if gids.empty:
        return  # no goal_id data — render nothing (graceful)

    st.subheader("🔍 Per-goal learning — same goal over time")
    st.caption(
        "Pick one goal (a fixed coordinate) and a metric to see whether the robot "
        "**improves on that goal** as training proceeds. Markers are coloured by "
        "outcome, so a red point after green points = it collided again though it "
        "had succeeded on this goal before."
    )
    st.info(
        "**Outcome metrics are computed over all episodes. Path-efficiency metrics "
        "are summarized over successful episodes only**, because path quality is "
        "meaningful for final comparison only when the robot reaches the goal."
    )

    # Appearance counts come from ONE frame (blackbox preferred — it has outcome +
    # goal_id for every episode). Concatenating pl_df and bb_df here would count the
    # same episode twice whenever both carry goal_id, which is the normal case now.
    _count_src = bb_df if (bb_df is not None and not bb_df.empty
                           and "goal_id" in bb_df.columns) else pl_df
    _cs = _count_src["goal_id"].astype(str) if (
        _count_src is not None and not _count_src.empty
        and "goal_id" in _count_src.columns) else pd.Series(dtype=str)
    _cs = _cs[_cs.str.strip().ne("") & _cs.ne("nan") & _cs.ne("None")]
    counts = _cs.value_counts() if not _cs.empty else gids.value_counts()
    _sm_count = _success_mask(_count_src)
    succ_counts = (_count_src.loc[_sm_count, "goal_id"].astype(str).value_counts()
                   if _sm_count is not None else pd.Series(dtype=int))

    def _goal_label(gid: str) -> str:
        tot = int(counts.get(gid, 0))
        suc = int(succ_counts.get(gid, 0))
        return f"{gid}  (total n={tot}, success n={suc})"

    c1, c2 = st.columns([3, 3])
    with c1:
        goal = st.selectbox(
            "Goal (by appearance count)", list(counts.index),
            format_func=_goal_label, key=f"plan_goalsel_{run_name}",
        )
    with c2:
        avail = [lbl for lbl, (src, col) in _GOAL_METRICS.items()
                 if col == "_success01"
                 or col in (pl_df.columns if src == "planning" else bb_df.columns)]
        metric_label = st.selectbox("Metric", avail, key=f"plan_goalmetric_{run_name}")

    src, col = _GOAL_METRICS[metric_label]
    is_path = _is_path_metric(col)

    # ── Path-metric filter (success-only by default) ──────────────────────────
    path_mode = "Success-only (recommended for thesis)"
    show_failed = False
    if is_path:
        f1, f2 = st.columns([3, 2])
        with f1:
            path_mode = st.radio(
                "Path metric filter",
                ["Success-only (recommended for thesis)",
                 "All episodes (diagnostic only)"],
                index=0, horizontal=False, key=f"plan_goalfilter_{run_name}",
            )
        with f2:
            show_failed = st.checkbox(
                "Show failed episodes as diagnostic markers", value=True,
                key=f"plan_goalshowfail_{run_name}",
                help="Collision/timeout points stay visible for diagnosis but are "
                     "excluded from rolling lines and summary cards in success-only mode.",
            )
    success_only = is_path and path_mode.startswith("Success-only")
    if is_path and not success_only:
        st.warning(
            "**Diagnostic mode:** failed episodes may show high path-efficiency "
            "values because the metric is mathematically computed from travelled "
            "path length even when the robot did not reach the goal. Do not use "
            "this mode for final thesis path-efficiency claims."
        )

    base = pl_df if src == "planning" else bb_df
    if base is None or base.empty or "goal_id" not in base.columns \
            or "episode" not in base.columns:
        st.caption(f"*No `{metric_label}` data with goal_id for this run.*")
        return

    g = base[base["goal_id"].astype(str) == str(goal)].copy()
    if src == "planning" and "planner_status" in g.columns:
        g = g[g["planner_status"] == "ok"]
    if "outcome" in g.columns and col == "_success01":
        g["_success01"] = (g["outcome"] == "success").astype(int)
    if g.empty or col not in g.columns:
        st.caption(f"*No `{metric_label}` rows for goal {goal}.*")
        return

    g = g.sort_values("episode")
    yv = pd.to_numeric(g[col], errors="coerce")

    # Split the frame: `g_stat` drives every rolling line / mean / summary card,
    # `g` still drives the outcome-coloured markers so failures stay *visible*
    # without ever entering a path-metric aggregate.
    sm_g = _success_mask(g)
    if success_only and sm_g is not None:
        g_stat, y_stat = g[sm_g], yv[sm_g]
    else:
        g_stat, y_stat = g, yv
    if g_stat.empty:
        st.caption(f"*No successful episodes yet for goal {goal} — "
                   f"nothing to summarise in success-only mode.*")
        return

    chart_label = f"{metric_label} — success-only" if success_only else metric_label

    # ── Outcome-coloured scatter + rolling trend ──────────────────────────────
    fig = go.Figure()
    if "outcome" in g.columns:
        for oc, oc_color in OUTCOME_TEXT_COLORS.items():
            # In success-only mode failed markers are optional diagnostics; they are
            # never part of y_stat, so they cannot move the rolling line or the cards.
            if oc != "success" and success_only and not show_failed:
                continue
            m = g["outcome"] == oc
            if m.any():
                is_diag = success_only and oc != "success"
                fig.add_trace(go.Scatter(
                    x=g.loc[m, "episode"], y=yv[m], mode="markers",
                    name=f"{oc} (diagnostic, excluded)" if is_diag else oc,
                    marker=dict(size=6, color=oc_color,
                                symbol="x" if is_diag else "circle",
                                opacity=0.55 if is_diag else 1.0),
                ))
    else:
        fig.add_trace(go.Scatter(x=g["episode"], y=yv, mode="markers",
                                 name=metric_label, marker=dict(size=6)))
    fig.add_trace(go.Scatter(
        x=g_stat["episode"], y=y_stat.rolling(roll_window, min_periods=1).mean(),
        mode="lines",
        name=f"Rolling {roll_window}" + (" (success-only)" if success_only else ""),
        line=dict(width=2, color="#1f77b4", dash="dash"),
    ))
    ylo_hi = dict(range=[0, 1.05]) if col in _GOAL_METRICS_UNIT else {}
    fig.update_layout(
        xaxis_title="Episode", yaxis_title=chart_label, yaxis=ylo_hi,
        height=360, margin=dict(l=50, r=20, t=30, b=40), legend=dict(font_size=10),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"plan_goalcurve_{run_name}")
    if success_only:
        st.caption(
            "Collision and timeout episodes are shown for diagnosis only and are "
            "excluded from success-only path metric summaries."
        )

    # ── Improvement summary: first third vs last third (by episode order) ──────
    # Outcome rates use ALL appearances of the goal (`g`); the path-metric mean uses
    # the filtered frame (`g_stat`) so failures never inflate it.
    n = len(g)
    if n >= 6:
        k = max(1, n // 3)
        first, last = g.iloc[:k], g.iloc[-k:]

        def _rate(df, oc):
            return round((df["outcome"] == oc).mean() * 100, 1) if "outcome" in df else None

        def _delta(a, b):
            return f"{b - a:+.1f}" if (a is not None and b is not None) else None

        ns = len(g_stat)
        ks = max(1, ns // 3)
        first_s, last_s = g_stat.iloc[:ks], g_stat.iloc[-ks:]

        def _avg(df):
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(s.mean()) if not s.empty else None

        st.caption(
            f"**Improvement** — outcome rates over first {k} vs last {k} appearances; "
            f"{metric_label} mean over first {ks} vs last {ks} "
            f"{'successful ' if success_only else ''}appearances"
        )
        m1, m2, m3 = st.columns(3)
        sf, sl = _rate(first, "success"), _rate(last, "success")
        cf, cl = _rate(first, "collision"), _rate(last, "collision")
        af, al = _avg(first_s), _avg(last_s)
        m1.metric("Success rate", _fmt(sl, 1, "%"), _delta(sf, sl))
        m2.metric("Collision rate", _fmt(cl, 1, "%"),
                  _delta(cf, cl), delta_color="inverse")
        m3.metric(f"{metric_label} mean" + (", success-only" if success_only else ""),
                  _fmt(al, 3), _delta(af, al))

    # ── Outcome trend for this goal (ALL episodes, always) ────────────────────
    _section_goal_outcome_trend(run_name, bb_df, goal, roll_window)

    # ── Filtered episode table (planning + blackbox merged on episode) ────────
    cols_pl = [c for c in ["episode", "datetime", "outcome",
                           "planner_path_efficiency", "actual_path_length"]
               if c in pl_df.columns]
    tbl = pl_df[pl_df["goal_id"].astype(str) == str(goal)][cols_pl].copy() \
        if (not pl_df.empty and "goal_id" in pl_df.columns) else pd.DataFrame()
    if not bb_df.empty and "goal_id" in bb_df.columns:
        cols_bb = [c for c in ["episode", "episode_steps", "min_obstacle_dist",
                               "near_collisions"] if c in bb_df.columns]
        bbg = bb_df[bb_df["goal_id"].astype(str) == str(goal)][cols_bb]
        if not tbl.empty and "episode" in cols_bb:
            tbl = tbl.merge(bbg, on="episode", how="left")
        elif tbl.empty:
            tbl = bbg
    if not tbl.empty:
        tbl = tbl.sort_values("episode")
        try:
            styled = tbl.style.map(_color_outcome_text, subset=["outcome"]) \
                if "outcome" in tbl.columns else tbl.style
            st.dataframe(styled, width="stretch", hide_index=True,
                         key=f"plan_goaltbl_{run_name}")
        except Exception:
            st.dataframe(tbl.astype(str), width="stretch", hide_index=True,
                         key=f"plan_goaltbl_fb_{run_name}")


def _section_planner(run_name: str, pl_df: pd.DataFrame, bb_df: pd.DataFrame):
    if pl_df.empty:
        st.info(f"No planning CSV found for **{run_name}**. "
                "Run training on the `planner-efficiency-metric` branch to generate it.")
        return

    ok_mask = pl_df["planner_status"] == "ok"
    ok_df   = pl_df[ok_mask]
    suc_mask = ok_mask & (pl_df["outcome"] == "success")

    # ── Summary cards ────────────────────────────────────────────────────────
    st.subheader("Path Efficiency Summary")
    has_center = "planner_path_efficiency_center" in pl_df.columns

    st.caption("**Region metric** — A* path to goal acceptance region (0.4 m radius)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean (region)",   _fmt(_mean(ok_df,           "planner_path_efficiency"), 3))
    c2.metric("Median (region)", _fmt(_median(ok_df,         "planner_path_efficiency"), 3))
    c3.metric("Success-only (region)",
              _fmt(_mean(pl_df[suc_mask], "planner_path_efficiency"), 3))
    invalid_count = int((pl_df["planner_status"] != "ok").sum())
    c4.metric("Invalid / No-path", str(invalid_count))
    raw_over_1 = int((pl_df["planner_path_efficiency_raw"].dropna() > 1.0).sum()) \
        if "planner_path_efficiency_raw" in pl_df.columns else 0
    c5.metric("Raw > 1.0 (region)", str(raw_over_1))

    if has_center:
        ok_cen_mask = pl_df["planner_status_center"] == "ok"
        ok_cen_df   = pl_df[ok_cen_mask]
        suc_cen_mask = ok_cen_mask & (pl_df["outcome"] == "success")
        st.caption("**Centre metric** — A* path to exact goal centre (0.05 m threshold)")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Mean (centre)",   _fmt(_mean(ok_cen_df,          "planner_path_efficiency_center"), 3))
        d2.metric("Median (centre)", _fmt(_median(ok_cen_df,        "planner_path_efficiency_center"), 3))
        d3.metric("Success-only (centre)",
                  _fmt(_mean(pl_df[suc_cen_mask], "planner_path_efficiency_center"), 3))
        cen_invalid = int((pl_df["planner_status_center"] != "ok").sum()) \
            if "planner_status_center" in pl_df.columns else 0
        d4.metric("Centre No-path", str(cen_invalid))
        cen_raw_over_1 = int((pl_df["planner_path_efficiency_center_raw"].dropna() > 1.0).sum()) \
            if "planner_path_efficiency_center_raw" in pl_df.columns else 0
        d5.metric("Raw > 1.0 (centre)", str(cen_raw_over_1))

    if "planner_path_efficiency_hybrid" in pl_df.columns:
        ok_hyb_mask = pl_df["planner_status_hybrid"] == "ok"
        ok_hyb_df   = pl_df[ok_hyb_mask]
        suc_hyb_mask = ok_hyb_mask & (pl_df["outcome"] == "success")
        st.caption("**Hybrid-A\\* metric** — nonholonomic + obstacle-aware (fair denominator; primary)")
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Mean (hybrid)",   _fmt(_mean(ok_hyb_df,   "planner_path_efficiency_hybrid"), 3))
        h2.metric("Median (hybrid)", _fmt(_median(ok_hyb_df, "planner_path_efficiency_hybrid"), 3))
        h3.metric("Success-only (hybrid)",
                  _fmt(_mean(pl_df[suc_hyb_mask], "planner_path_efficiency_hybrid"), 3))
        h4.metric("Hybrid No-path", str(int((pl_df["planner_status_hybrid"] != "ok").sum())))
        hyb_raw_over_1 = int((pl_df["planner_path_efficiency_hybrid_raw"].dropna() > 1.0).sum()) \
            if "planner_path_efficiency_hybrid_raw" in pl_df.columns else 0
        h5.metric("Raw > 1.0 (hybrid)", str(hyb_raw_over_1))

    # ── Planner status breakdown ──────────────────────────────────────────────
    with st.expander("Planner status breakdown", expanded=False):
        status_counts = pl_df["planner_status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        st.dataframe(status_counts, width="stretch", hide_index=True,
                     key=f"plan_status_{run_name}")

    # ── Efficiency chart ──────────────────────────────────────────────────────
    st.subheader("Planner Path Efficiency vs Episode")
    roll_window = st.slider("Rolling average window", 5, 100, 25, key=f"roll_{run_name}")

    # Success-only view: paper-grade filtering — efficiency/directness claims should
    # be made on successful episodes only (failed episodes mix in truncated paths).
    success_only = st.checkbox(
        "Success-only (recommended for reported results)",
        value=True, key=f"plan_suconly_{run_name}",
        help="On by default: path efficiency is only meaningful when the robot "
             "actually reached the goal. Unticking shows failed episodes too — "
             "diagnostic only, not for reported path-efficiency claims.",
    ) if "outcome" in pl_df.columns else False
    chart_df = pl_df[pl_df["outcome"] == "success"] if success_only else pl_df
    if "outcome" in pl_df.columns and not success_only:
        st.warning(
            "**Diagnostic mode:** failed episodes may show high path-efficiency "
            "values because the metric is computed from travelled path length even "
            "when the robot did not reach the goal. Do not use this mode for final "
            "thesis path-efficiency claims."
        )

    # Per-goal grouping (fixed-goal experiments): one rolling-mean series per goal_id,
    # so you see whether each goal's path tightens over training. Degrades silently
    # on older CSVs without the column.
    has_goal_id = "goal_id" in chart_df.columns and chart_df["goal_id"].nunique() > 1
    group_by_goal = False
    if has_goal_id:
        group_by_goal = st.checkbox(
            f"Group by goal_id ({chart_df['goal_id'].nunique()} goals) — one rolling series per goal",
            value=False, key=f"plan_goalgrp_{run_name}",
        )

    # Raw marker traces are thinned above max_points; rolling means below are
    # always computed on the FULL chart_df first (so the rolling window keeps
    # its real meaning), only the marker scatter itself ships fewer points.
    raw_view = _thin(chart_df)

    fig = go.Figure()
    if group_by_goal:
        for i, (gid, g) in enumerate(chart_df.groupby("goal_id")):
            ok = g[g["planner_status"] == "ok"] if "planner_status" in g.columns else g
            if ok.empty or "planner_path_efficiency" not in ok.columns:
                continue
            rolled_g = ok["planner_path_efficiency"].rolling(roll_window, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=ok["episode"], y=rolled_g,
                mode="lines+markers", name=str(gid),
                marker=dict(size=3), line=dict(width=1.6, color=_clr(i)),
                connectgaps=False,
            ))
    elif "episode" in chart_df.columns and "planner_path_efficiency" in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=raw_view["episode"],
            y=raw_view["planner_path_efficiency"],
            mode="lines+markers",
            name="Efficiency (region)",
            marker=dict(size=3),
            line=dict(width=1, color=PALETTE[0]),
            connectgaps=False,
        ))
        rolled = chart_df["planner_path_efficiency"].rolling(roll_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=chart_df["episode"],
            y=rolled,
            mode="lines",
            name=f"Rolling {roll_window} (region)",
            line=dict(width=2, color=PALETTE[0], dash="dash"),
        ))
    if not group_by_goal and "planner_path_efficiency_center" in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=raw_view["episode"],
            y=raw_view["planner_path_efficiency_center"],
            mode="lines+markers",
            name="Efficiency (centre)",
            marker=dict(size=3),
            line=dict(width=1, color="#9467bd"),
            connectgaps=False,
        ))
        rolled_cen = chart_df["planner_path_efficiency_center"].rolling(roll_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=chart_df["episode"],
            y=rolled_cen,
            mode="lines",
            name=f"Rolling {roll_window} (centre)",
            line=dict(width=2, color="#9467bd", dash="dot"),
        ))
    if not group_by_goal and "planner_path_efficiency_hybrid" in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=raw_view["episode"],
            y=raw_view["planner_path_efficiency_hybrid"],
            mode="lines+markers",
            name="Efficiency (Hybrid-A*)",
            marker=dict(size=3),
            line=dict(width=1, color="#17becf"),
            connectgaps=False,
        ))
        rolled_hyb = chart_df["planner_path_efficiency_hybrid"].rolling(roll_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=chart_df["episode"],
            y=rolled_hyb,
            mode="lines",
            name=f"Rolling {roll_window} (Hybrid-A*)",
            line=dict(width=2.4, color="#17becf", dash="dash"),
        ))
    fig.update_layout(
        xaxis_title="Episode", yaxis_title="Efficiency",
        yaxis=dict(range=[0, 1.05]),
        height=360, margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(font_size=10),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"plan_{run_name}")

    # ── Per-goal learning drill-down (one goal over time, any metric) ──────────
    st.divider()
    _section_goal_learning(run_name, pl_df, bb_df, roll_window)

    # ── Planning table (clickable — click a row to view its path plot) ──────────
    st.subheader("Planning Log  —  click a row to view its path plot")
    display_cols = [c for c in [
        "episode", "outcome", "goal_id",
        "planner_status_hybrid", "planner_path_efficiency_hybrid", "planner_path_efficiency_hybrid_raw",
        "planner_status", "planner_path_efficiency", "planner_path_efficiency_raw",
        "planner_status_center", "planner_path_efficiency_center", "planner_path_efficiency_center_raw",
        "initial_distance", "actual_path_length",
        "planned_path_length", "planned_path_length_center", "planned_path_length_hybrid",
    ] if c in pl_df.columns]

    selected_ep = None
    outcome_val = ""
    sel_row_goal = ""
    # Set by _section_goal_learning's selectbox just above; absent when the run has
    # no goal_id data (that function returns early before creating the widget).
    sel_goal = st.session_state.get(f"plan_goalsel_{run_name}")
    has_goal_col = "goal_id" in pl_df.columns

    exclude_bad_planner = st.checkbox(
        "Exclude non-ok planner rows (no_path / planner_error / …)",
        value=False, key=f"plan_exclude_invalid_{run_name}",
        help="Off by default. planner_status != 'ok' rows have blank efficiency "
             "by design (see CLAUDE.md) — this just hides them from the table.",
    ) if "planner_status" in pl_df.columns else False

    # Keeps the clicked row — and therefore the path plot below — on the same goal
    # the per-goal chart above is filtering, so the two can't visually disagree.
    filter_to_goal = st.checkbox(
        f"Filter planning log to selected per-goal goal_id  (`{sel_goal}`)",
        value=True, key=f"plan_filter_to_goal_{run_name}",
        help="On by default so the row you click — and therefore the path plot "
             "below — always belongs to the goal selected in the per-goal chart "
             "above. Turn off to browse every goal.",
    ) if (sel_goal and has_goal_col) else False

    log_df = pl_df[pl_df["planner_status"] == "ok"] if exclude_bad_planner else pl_df
    if filter_to_goal:
        log_df = log_df[log_df["goal_id"].astype(str) == str(sel_goal)]
        if log_df.empty:
            st.caption(f"*No planning rows for goal {sel_goal} under the current filters.*")

    # Goal-scoped widget key: changing the per-goal dropdown creates a NEW widget
    # with an empty selection instead of carrying a stale row index onto a
    # different goal's rows. This is why the goal mismatch is fixed by filtering
    # rather than by hand-clearing the selection (Streamlit has no API for that).
    _sel_tag = f"_{sel_goal}" if filter_to_goal else ""

    # Row cap applied BEFORE display — the click handler below must resolve
    # row_idx against this SAME frame (table_df), since Streamlit's on_select
    # returns a position within whatever was actually rendered, not pl_df.
    table_df = _capped_table(log_df[display_cols], key=f"plan_table_{run_name}")

    try:
        subset = ["outcome"] if "outcome" in display_cols else None
        base = table_df.style
        if subset:
            try:
                styled = base.map(_color_outcome_text, subset=subset)
            except AttributeError:
                styled = base.applymap(_color_outcome_text, subset=subset)
        else:
            styled = base
        event = st.dataframe(
            styled, width="stretch",
            key=f"plan_table_render_{run_name}{_sel_tag}",
            on_select="rerun", selection_mode="single-row",
        )
    except Exception:
        display_pl = table_df.copy()
        for col in display_pl.select_dtypes(include="object").columns:
            display_pl[col] = display_pl[col].astype(str)
        event = st.dataframe(
            display_pl, width="stretch",
            key=f"plan_table_fb_{run_name}{_sel_tag}",
            on_select="rerun", selection_mode="single-row",
        )

    # Resolve selected episode from the clicked row — positional into table_df
    # (the rendered frame), NOT pl_df.
    sel_rows = getattr(getattr(event, "selection", None), "rows", [])
    if sel_rows:
        row_idx = sel_rows[0]
        if 0 <= row_idx < len(table_df):
            clicked = table_df.iloc[row_idx]
            if "episode" in clicked.index:
                selected_ep = int(clicked["episode"])
            if "outcome" in clicked.index:
                outcome_val = str(clicked["outcome"])
            if "goal_id" in clicked.index:
                sel_row_goal = str(clicked["goal_id"])

    # ── Episode Path Plot (driven by table row click) ─────────────────────────
    st.subheader("Episode Path Plot")
    st.caption(
        "The path plot is generated from the selected Planning Log row, not "
        "directly from the per-goal dropdown."
    )

    if "episode" not in pl_df.columns:
        st.info("No episodes in planning CSV.")
        return

    if selected_ep is None:
        st.info("Click a row in the Planning Log table above to view its path plot.")
    else:
        ep_row = pl_df[pl_df["episode"] == selected_ep]

        # Provenance of the image below: which row it came from, and what the
        # per-goal chart above is currently filtering.
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Episode", str(selected_ep))
        i2.metric("Outcome", outcome_val or "—")
        i3.metric("Row goal_id", sel_row_goal or "—")
        i4.metric("Per-goal filter", str(sel_goal) if sel_goal else "—")
        # Unreachable while the goal filter is ON (it is filtered out by
        # construction); this is the labelling path for the filter-OFF case.
        if sel_goal and sel_row_goal and str(sel_row_goal) != str(sel_goal):
            st.warning(
                f"The selected path plot belongs to goal_id {sel_row_goal}, while "
                f"the Per-goal chart above is filtering goal_id {sel_goal}."
            )

        _plot_dir = (run_name[len("eval_"):] + "_eval"
                     if run_name.startswith("eval_")
                     else run_name)
        png_path = _find_png(run_name, int(selected_ep), outcome_val)
        if png_path:
            st.image(str(png_path), use_container_width=True)
        else:
            st.info(
                f"No path plot PNG found for episode {selected_ep} "
                f"(expected: `path_plots/{_plot_dir}/ep{int(selected_ep):05d}_{outcome_val}.png`)."
            )

        # Details row below image
        det_left, det_right = st.columns(2)

        with det_left:
            st.markdown("**Planning metrics (this episode)**")
            if not ep_row.empty:
                detail_cols = [c for c in [
                    "episode", "outcome", "planner_status",
                    "planner_path_efficiency", "planner_path_efficiency_raw",
                    "initial_distance", "actual_path_length", "planned_path_length",
                    "start_x", "start_y", "target_x", "target_y",
                ] if c in ep_row.columns]
                tbl = ep_row[detail_cols].T.rename(columns={ep_row.index[0]: "value"})
                tbl["value"] = tbl["value"].astype(str)
                st.table(tbl)

        with det_right:
            st.markdown("**Blackbox metrics (this episode)**")
            if not bb_df.empty and "episode" in bb_df.columns:
                bb_ep = bb_df[bb_df["episode"] == selected_ep]
                if not bb_ep.empty:
                    bb_cols = [c for c in [
                        "steps_to_goal", "path_directness",
                        "min_obstacle_dist", "near_collisions",
                        "success_rate", "collision_rate",
                    ] if c in bb_ep.columns]
                    tbl = bb_ep[bb_cols].T.rename(columns={bb_ep.index[0]: "value"})
                    tbl["value"] = tbl["value"].astype(str)
                    st.table(tbl)
                else:
                    st.caption("No matching blackbox row for this episode.")
            else:
                st.caption("No blackbox CSV for this run.")


# BO-searched flags (emitted verbatim as --{key}). actor_entropy is a top-level flag
# (not --reward_*), so it appears in the leaderboard and validation command too.
_BO_REWARD_KEYS = [
    "reward_progress_scale", "reward_step_penalty", "reward_turn_penalty",
    "reward_near_obstacle_scale", "reward_near_obstacle_sigma", "actor_entropy",
]


def _section_bo_trials(df: pd.DataFrame):
    """BO trial comparison — bar charts, scatter trade-off, leaderboard."""
    if df.empty:
        st.info(
            "No BO trial data found. Switch the sidebar folder to a "
            "`tune_stage{N}/` subfolder (or `tune_stage{N}_{mode}/` for a "
            "non-`none` odometry-mode study, e.g. `tune_stage1_full_imu`), or run:\n\n"
            "`python3 export_tune_results.py --stage N` "
            "(add `--odometry-mode full_imu` for a mode-specific study)"
        )
        return

    # ── Coerce numeric columns ────────────────────────────────────────────────
    df = df.copy()
    for col in ["efficiency", "success", "constraint", "trial"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "feasible" in df.columns:
        df["feasible"] = df["feasible"].apply(
            lambda x: str(x).strip().lower() == "true" if pd.notna(x) else False
        )

    completed = df[df["efficiency"].notna()]
    feasible  = completed[completed["feasible"]]
    best_row  = (
        feasible.loc[feasible["efficiency"].idxmax()] if not feasible.empty
        else (completed.loc[completed["efficiency"].idxmax()] if not completed.empty else None)
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Trials",    len(df))
    m2.metric("Completed",       len(completed))
    m3.metric("Feasible",        len(feasible))
    best_eff = float(best_row["efficiency"]) if best_row is not None and pd.notna(best_row["efficiency"]) else None
    best_suc_raw = best_row.get("success") if best_row is not None else None
    best_suc = float(best_suc_raw) if best_suc_raw is not None and pd.notna(best_suc_raw) else None
    m4.metric("Best Efficiency", _fmt(best_eff, 4))
    m5.metric("Best Success %",  _fmt(best_suc, 1, "%"))

    if completed.empty:
        st.info("No completed trials yet.")
        return

    # ── Hover text helper ─────────────────────────────────────────────────────
    def _htxt(r):
        eff = f"{r['efficiency']:.4f}" if pd.notna(r.get("efficiency")) else "—"
        suc = f"{r['success']:.1f}%" if pd.notna(r.get("success")) else "—"
        return (f"Trial {int(r['trial'])}<br>run: {r.get('run_name', '')}"
                f"<br>efficiency: {eff}<br>success: {suc}")

    # ── Color per trial: gold=best, green=feasible, red=infeasible ────────────
    def _colors(sub: pd.DataFrame) -> list[str]:
        out = []
        for _, row in sub.iterrows():
            if best_row is not None and int(row["trial"]) == int(best_row["trial"]):
                out.append("gold")
            elif row["feasible"]:
                out.append("#2ca02c")
            else:
                out.append("#d62728")
        return out

    # ── Bar chart: Path Efficiency per trial ──────────────────────────────────
    st.subheader("Path Efficiency per Trial")
    st.caption("Gold = best feasible.  Green = feasible.  Red = infeasible.")
    eff_df = completed.sort_values("trial")
    fig_eff = go.Figure(go.Bar(
        x=eff_df["trial"].astype(int).astype(str),
        y=eff_df["efficiency"],
        marker_color=_colors(eff_df),
        text=[f"{v:.3f}" if pd.notna(v) else "" for v in eff_df["efficiency"]],
        textposition="outside",
        hovertext=[_htxt(r) for _, r in eff_df.iterrows()],
        hoverinfo="text",
    ))
    ymax = float(eff_df["efficiency"].max()) if not eff_df.empty else 1.0
    fig_eff.update_layout(
        xaxis_title="Trial #", yaxis_title="Planner Path Efficiency",
        yaxis=dict(range=[0, min(1.5, ymax * 1.2 + 0.05)]),
        height=380, margin=dict(l=50, r=20, t=40, b=50), showlegend=False,
    )
    st.plotly_chart(fig_eff, use_container_width=True, key="bo_eff")

    # ── Bar chart: Success Rate per trial ─────────────────────────────────────
    suc_df = completed[completed["success"].notna()].sort_values("trial")
    st.subheader("Success Rate per Trial")
    if not suc_df.empty:
        fig_suc = go.Figure(go.Bar(
            x=suc_df["trial"].astype(int).astype(str),
            y=suc_df["success"],
            marker_color=_colors(suc_df),
            text=[f"{v:.1f}%" if pd.notna(v) else "" for v in suc_df["success"]],
            textposition="outside",
            hovertext=[_htxt(r) for _, r in suc_df.iterrows()],
            hoverinfo="text",
        ))
        fig_suc.update_layout(
            xaxis_title="Trial #", yaxis_title="Success Rate (%)",
            yaxis=dict(range=[0, 115]),
            height=380, margin=dict(l=50, r=20, t=40, b=50), showlegend=False,
        )
        st.plotly_chart(fig_suc, use_container_width=True, key="bo_suc")
    else:
        st.caption("*Success rate — not available in this data*")

    # ── Scatter: Efficiency vs Success trade-off ──────────────────────────────
    sc_df = completed[completed["success"].notna() & completed["efficiency"].notna()]
    if not sc_df.empty:
        st.subheader("Efficiency vs Success Rate Trade-off")
        fig_sc = go.Figure()
        for feas_val, label, color in [
            (True,  "Feasible",   "#2ca02c"),
            (False, "Infeasible", "#d62728"),
        ]:
            sub = sc_df[sc_df["feasible"] == feas_val]
            if sub.empty:
                continue
            fig_sc.add_trace(go.Scatter(
                x=sub["success"], y=sub["efficiency"],
                mode="markers", name=label,
                marker=dict(color=color, size=9, opacity=0.8),
                hovertext=[_htxt(r) for _, r in sub.iterrows()],
                hoverinfo="text",
            ))
        if best_row is not None and pd.notna(best_row.get("success")) and pd.notna(best_row.get("efficiency")):
            fig_sc.add_trace(go.Scatter(
                x=[float(best_row["success"])], y=[float(best_row["efficiency"])],
                mode="markers", name="Best",
                marker=dict(color="gold", size=14, symbol="star"),
                hovertext=[f"Best — {_htxt(best_row)}"],
                hoverinfo="text",
            ))
        fig_sc.update_layout(
            xaxis_title="Success Rate (%)", yaxis_title="Planner Path Efficiency",
            height=380, margin=dict(l=50, r=20, t=40, b=50),
        )
        st.plotly_chart(fig_sc, use_container_width=True, key="bo_scatter")

    # ── Leaderboard table ─────────────────────────────────────────────────────
    st.subheader("Trial Leaderboard")
    st.caption("Feasible trials first, then by efficiency descending.")
    hide_incomplete = (
        st.checkbox("Hide incomplete trials (RUNNING / FAIL)", value=True, key="bo_hide_incomplete")
        if "state" in df.columns else False
    )
    board_df = df[df["state"] == "COMPLETE"] if hide_incomplete else df
    display_cols = ["trial", "run_name", "state", "efficiency", "success",
                    "feasible", "constraint", "partial", "n_eff_rows"] + \
                   [k for k in _BO_REWARD_KEYS if k in board_df.columns]
    display_cols = [c for c in display_cols if c in board_df.columns]
    disp = board_df[display_cols].copy()
    disp["_sf"] = (~disp["feasible"].fillna(False)).astype(int)
    disp["_se"] = -disp["efficiency"].fillna(float("-inf"))
    disp = disp.sort_values(["_sf", "_se"]).drop(columns=["_sf", "_se"])
    for col in ["efficiency", "success", "constraint"] + _BO_REWARD_KEYS:
        if col in disp.columns:
            disp[col] = pd.to_numeric(disp[col], errors="coerce").round(4)
    for col in disp.select_dtypes(include="object").columns:
        disp[col] = disp[col].astype(str)
    st.dataframe(disp, width="stretch", hide_index=True, key="bo_table")

    # ── Best weights — ready-to-run validation command ────────────────────────
    if best_row is not None:
        st.subheader("Best Config — Validation Command")
        avail = [k for k in _BO_REWARD_KEYS if k in best_row.index and pd.notna(best_row.get(k))]
        weight_flags = " \\\n  ".join(f"--{k} {float(best_row[k]):.6g}" for k in avail)
        # Derive stage + odometry mode from the selected BO folder (tune_stage{N}_{mode})
        # so the validation command matches the study that was actually tuned.
        _m = re.match(r"tune_stage(\d+)_(.+)$", CSV_DIR.name)
        _stage, _odom = (_m.group(1), _m.group(2)) if _m else ("1", "none")
        st.code(
            f"python3 dreamer.py --configs turtle --task turtle \\\n"
            f"  --logdir ./logdir/stage{_stage}_360_{_odom}_seed0_reward_tuned \\\n"
            f"  --stage {_stage} --lidar 360 --odometry_mode {_odom} --seed 0 \\\n"
            f"  --device cuda --steps 300000 --eval_episode_num 100 \\\n"
            f"  --reward_mode shaped \\\n"
            f"  {weight_flags}",
            language="bash",
        )
        tag = "feasible" if best_row["feasible"] else "best completed — no feasible trial yet"
        eff_str = f"{best_eff:.4f}" if best_eff is not None else "—"
        suc_str = f"{best_suc:.1f}%" if best_suc is not None else "—"
        st.caption(f"Trial {int(best_row['trial'])} ({tag}) | efficiency: {eff_str} | success: {suc_str}")

    # ── Trial filter — applied BEFORE the per-trial CSV loads below ───────────
    # The learning-curve sections each load one planning + one blackbox CSV per
    # trial (~5-6k rows each). With 100+ trials in a study that's hundreds of
    # multi-thousand-row reads on every rerun unless narrowed first — so filter
    # the trial list here, against the already-loaded (small) `df`, before any
    # per-episode file is touched.
    st.divider()
    st.subheader("Trial data to load")
    c_f1, c_f2, c_f3 = st.columns([2, 1, 1])
    with c_f1:
        state_opts = sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else []
        default_states = [s for s in ["COMPLETE"] if s in state_opts] or state_opts
        sel_states = st.multiselect(
            "Trial state", state_opts, default=default_states, key="bo_load_states",
        ) if state_opts else []
    with c_f2:
        load_all = st.checkbox("Load all trials", value=False, key="bo_load_all")
    with c_f3:
        last_n = st.number_input(
            "Last N", min_value=1, max_value=max(1, len(df)),
            value=min(20, max(1, len(df))), step=5,
            key="bo_load_lastn", disabled=load_all,
        )

    df_load = df[df["state"].isin(sel_states)] if sel_states else df
    if not load_all and "trial" in df_load.columns:
        df_load = df_load.sort_values("trial").tail(int(last_n))
    st.caption(f"Loading {len(df_load)} of {len(df)} trials below "
               f"({'all trials' if load_all else f'last {int(last_n)}'}"
               f"{', state ' + '/'.join(sel_states) if sel_states and sel_states != state_opts else ''}).")

    # ── Planning Efficiency — Learning Curves ─────────────────────────────────
    st.subheader("Planning Efficiency Over Episodes")
    st.caption(
        "One line per trial — how path efficiency evolves as training progresses. "
        "Gold = best trial · Green = feasible · Red = infeasible."
    )

    c_phase, c_roll = st.columns([2, 3])
    with c_phase:
        phase = st.radio(
            "Episodes source", ["Train", "Eval"],
            horizontal=True, key="bo_pl_phase",
            help="Train = all training episodes (more points, noisier). "
                 "Eval = evaluation checkpoints only.",
        )
    with c_roll:
        roll_bo = st.slider("Rolling window", 3, 50, 15, key="bo_pl_roll")

    # Load planning CSVs for the FILTERED trial set only (not every trial in the study)
    trial_pl: dict = {}  # trial_num → (pl_df, feasible_bool)
    for _, row in df_load.iterrows():
        rn = str(row.get("run_name", "")) if pd.notna(row.get("run_name")) else ""
        if not rn:
            continue
        key = f"eval_{rn}" if phase == "Eval" else rn
        pl = load_pl(key)
        if not pl.empty and "episode" in pl.columns and "planner_path_efficiency" in pl.columns:
            trial_pl[int(row["trial"])] = (pl, bool(row["feasible"]))

    if not trial_pl:
        st.caption(
            "*No planning CSVs found for any trial in this folder.* "
            "Planning data is generated by `dreamer.py` on the `reward-shaping` branch."
        )
    else:
        all_t = sorted(trial_pl.keys())
        sel_trials = st.multiselect(
            "Show trials", all_t, default=all_t,
            format_func=lambda n: f"trial{n:03d}",
            key="bo_pl_sel",
            help="Deselect trials to reduce clutter.",
        )

        fig_pl = go.Figure()
        best_t = int(best_row["trial"]) if best_row is not None else None

        for t_num in sorted(sel_trials):
            if t_num not in trial_pl:
                continue
            pl_df, feas = trial_pl[t_num]

            if t_num == best_t:
                color, width, opacity = "gold", 2.5, 1.0
            elif feas:
                color, width, opacity = "#2ca02c", 1.5, 0.8
            else:
                color, width, opacity = "#d62728", 1.0, 0.55

            # ok status AND success-only (path metric) — see _planner_valid_mask.
            ok_mask = _planner_valid_mask(pl_df, "planner_path_efficiency")
            eff = pl_df["planner_path_efficiency"].where(ok_mask)
            rolled = eff.rolling(roll_bo, min_periods=1).mean()

            fig_pl.add_trace(go.Scatter(
                x=pl_df["episode"],
                y=rolled,
                mode="lines",
                name=f"trial{t_num:03d}",
                line=dict(color=color, width=width),
                opacity=opacity,
            ))

        fig_pl.update_layout(
            xaxis_title="Episode",
            yaxis_title=f"Planner Path Efficiency (rolling {roll_bo})",
            yaxis=dict(range=[0, 1.05]),
            height=460, margin=dict(l=50, r=20, t=30, b=40),
            legend=dict(font_size=10, orientation="v"),
        )
        st.plotly_chart(fig_pl, use_container_width=True, key="bo_pl_lines")

        # ── Final-episode efficiency per trial (sorted bar) ───────────────────
        st.subheader("Final Efficiency per Trial  (last episode in planning log)")
        final_rows = []
        for t_num in all_t:
            if t_num not in trial_pl:
                continue
            pl_df, feas = trial_pl[t_num]
            ok = pl_df[pl_df["planner_status"] == "ok"] \
                if "planner_status" in pl_df.columns else pl_df
            if ok.empty:
                continue
            last_eff = float(ok["planner_path_efficiency"].dropna().tail(roll_bo).mean())
            final_rows.append({"trial": t_num, "efficiency": last_eff, "feasible": feas})

        if final_rows:
            fin_df = pd.DataFrame(final_rows).sort_values("efficiency", ascending=False)
            fin_colors = [
                "gold" if r["trial"] == best_t
                else ("#2ca02c" if r["feasible"] else "#d62728")
                for _, r in fin_df.iterrows()
            ]
            fig_fin = go.Figure(go.Bar(
                x=fin_df["trial"].astype(int).astype(str),
                y=fin_df["efficiency"],
                marker_color=fin_colors,
                text=[f"{v:.3f}" for v in fin_df["efficiency"]],
                textposition="outside",
            ))
            fig_fin.update_layout(
                xaxis_title="Trial #", yaxis_title="Mean efficiency (last window)",
                yaxis=dict(range=[0, min(1.3, fin_df["efficiency"].max() * 1.2 + 0.05)]),
                xaxis=dict(categoryorder="array",
                           categoryarray=fin_df["trial"].astype(int).astype(str).tolist()),
                height=360, margin=dict(l=50, r=20, t=40, b=50), showlegend=False,
            )
            st.plotly_chart(fig_fin, use_container_width=True, key="bo_pl_final")

    # ── Blackbox Metrics — Learning Curves ────────────────────────────────────
    st.divider()
    st.subheader("Blackbox Metrics Over Episodes")
    st.caption(
        "Compare any per-episode training metric across trials. "
        "Gold = best trial · Green = feasible · Red = infeasible."
    )

    BB_METRIC_OPTIONS = {
        "Rolling Success Rate 100":   "rolling_success_rate_100",
        "Rolling Collision Rate 100": "rolling_collision_rate_100",
        "Rolling Success Rate 500":   "rolling_success_rate_500",
        "Rolling Collision Rate 500": "rolling_collision_rate_500",
        "Cumulative Success Rate":    "success_rate",
        "Cumulative Collision Rate":  "collision_rate",
        "Path Directness":            "path_directness",
        "Steps to Goal (success only)": "steps_to_goal",
        "Episode Steps (all outcomes)": "episode_steps",
        "Min Obstacle Distance (m)":  "min_obstacle_dist",
        "Near Collisions":            "near_collisions",
    }

    c_bb1, c_bb2, c_bb3 = st.columns([2, 3, 3])
    with c_bb1:
        bb_phase = st.radio(
            "Episodes source", ["Train", "Eval"],
            horizontal=True, key="bo_bb_phase",
            help="Train = all training episodes.  Eval = evaluation checkpoints only.",
        )
    with c_bb2:
        bb_metric_label = st.selectbox(
            "Metric", list(BB_METRIC_OPTIONS.keys()), key="bo_bb_metric",
        )
    with c_bb3:
        bb_roll = st.slider("Rolling window", 1, 100, 20, key="bo_bb_roll")

    bb_col = BB_METRIC_OPTIONS[bb_metric_label]
    # Every path/trajectory metric is success-only, not just steps_to_goal — a
    # failed episode still produces a path_directness / efficiency number.
    success_only_bb = _is_path_metric(bb_col)

    # Load blackbox CSVs for the same FILTERED trial set (df_load, from the
    # "Trial data to load" filter above) — not every trial in the study.
    trial_bb: dict = {}  # trial_num → (bb_df, feasible_bool)
    for _, row in df_load.iterrows():
        rn = str(row.get("run_name", "")) if pd.notna(row.get("run_name")) else ""
        if not rn:
            continue
        bb_key = f"eval_{rn}" if bb_phase == "Eval" else rn
        bb = load_bb(bb_key)
        if not bb.empty:
            trial_bb[int(row["trial"])] = (bb, bool(row["feasible"]))

    if not trial_bb:
        st.caption("*No blackbox CSVs found for any trial in this folder.*")
    else:
        all_bb_t = sorted(trial_bb.keys())
        sel_bb = st.multiselect(
            "Show trials", all_bb_t, default=all_bb_t,
            format_func=lambda n: f"trial{n:03d}",
            key="bo_bb_sel",
            help="Deselect trials to reduce clutter.",
        )

        fig_bb = go.Figure()
        for t_num in sorted(sel_bb):
            if t_num not in trial_bb:
                continue
            bb_df, feas = trial_bb[t_num]
            if bb_col not in bb_df.columns:
                continue

            if t_num == best_t:
                color, width, opacity = "gold", 2.5, 1.0
            elif feas:
                color, width, opacity = "#2ca02c", 1.5, 0.8
            else:
                color, width, opacity = "#d62728", 1.0, 0.55

            if success_only_bb:
                sm = _success_mask(bb_df)
                d = bb_df[sm] if sm is not None else bb_df
            else:
                d = bb_df

            if d.empty or bb_col not in d.columns:
                continue

            series = pd.to_numeric(d[bb_col], errors="coerce")
            rolled = series.rolling(bb_roll, min_periods=1).mean()

            fig_bb.add_trace(go.Scatter(
                x=d["_idx"],
                y=rolled,
                mode="lines",
                name=f"trial{t_num:03d}",
                line=dict(color=color, width=width),
                opacity=opacity,
            ))

        if fig_bb.data:
            fig_bb.update_layout(
                xaxis_title="Episode",
                yaxis_title=f"{bb_metric_label} (rolling {bb_roll})",
                height=460, margin=dict(l=50, r=20, t=30, b=40),
                legend=dict(font_size=10),
            )
            st.plotly_chart(fig_bb, use_container_width=True, key="bo_bb_lines")
        else:
            st.caption(f"*`{bb_col}` — column not found in any trial's blackbox CSV.*")


_CMP_METRICS = {
    "Path Efficiency":              ("planning", "planner_path_efficiency"),
    "Success Rate (rolling 100)":   ("blackbox", "rolling_success_rate_100"),
    "Collision Rate (rolling 100)": ("blackbox", "rolling_collision_rate_100"),
    "Success Rate (rolling 500)":   ("blackbox", "rolling_success_rate_500"),
    "Path Directness":              ("blackbox", "path_directness"),
    "Steps to Goal (success only)": ("blackbox", "steps_to_goal"),
    "Episode Steps (all outcomes)": ("blackbox", "episode_steps"),
    "Min Obstacle Distance (m)":    ("blackbox", "min_obstacle_dist"),
}


def _section_compare():
    """🔀 Compare — overlay any metric vs episode across arbitrary runs (any folder)."""
    run_index = scan_all_runs()  # (folder, run, label) across ALL folders
    if not run_index:
        st.info("← No runs detected in any CSV folder.")
        return
    labels = [label for _, _, label in run_index]
    label_to_fr = {label: (folder, run) for folder, run, label in run_index}

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([4, 2, 3, 2])
    with c1:
        sel_runs = st.multiselect(
            "Runs to compare (any folder)", labels,
            default=labels[:min(2, len(labels))],
            key="cmp_runs",
            help="Pick any combination across folders — e.g. none_stage4/… vs "
                 "full_imu_stage4/…, different reward modes, stages, seeds, trials.",
        )
    with c2:
        phase = st.radio(
            "Phase", ["Train", "Eval", "Both"],
            key="cmp_phase", horizontal=True,
            help="Train = training episodes.  Eval = evaluation checkpoints.  "
                 "Both = solid line (train) + dashed line (eval) per run.",
        )
    with c3:
        metric_label = st.selectbox(
            "Metric", list(_CMP_METRICS.keys()), key="cmp_metric",
        )
    with c4:
        roll = st.slider("Rolling window", 1, 100, 20, key="cmp_roll")

    if not sel_runs:
        st.info("← Select at least one run above.")
        return

    src, col = _CMP_METRICS[metric_label]
    phases = {"Train": ["train"], "Eval": ["eval"], "Both": ["train", "eval"]}[phase]
    # All path/trajectory metrics are success-only (not just steps_to_goal).
    success_only = _is_path_metric(col)

    # ── Build chart ───────────────────────────────────────────────────────────
    fig = go.Figure()
    skipped: list[str] = []

    for i, sel_label in enumerate(sel_runs):
        color = _clr(i)
        folder, run = label_to_fr[sel_label]
        for ph in phases:
            load_run = f"eval_{run}" if ph == "eval" else run
            dash = "solid" if ph == "train" else "dash"
            label = f"{sel_label} ({ph})" if phase == "Both" else sel_label

            if src == "planning":
                df = load_pl_in(folder, load_run)
                if df.empty or col not in df.columns:
                    skipped.append(label)
                    continue
                ok_mask = _planner_valid_mask(df, col, success_only)
                series = pd.to_numeric(df[col], errors="coerce").where(ok_mask)
                x = df["episode"] if "episode" in df.columns else pd.RangeIndex(len(df))
            else:
                df = load_bb_in(folder, load_run)
                if df.empty or col not in df.columns:
                    skipped.append(label)
                    continue
                if success_only:
                    sm = _success_mask(df)
                    df = df[sm] if sm is not None else df
                series = pd.to_numeric(df[col], errors="coerce")
                x = df["_idx"] if "_idx" in df.columns else pd.RangeIndex(len(df))

            if df.empty or series.dropna().empty:
                skipped.append(label)
                continue

            rolled = series.rolling(roll, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=x, y=rolled,
                mode="lines", name=label,
                line=dict(color=color, dash=dash, width=1.8),
            ))

    if not fig.data:
        st.caption(f"*No `{col}` data found for the selected runs / phase.*")
        if skipped:
            st.caption(f"Skipped (no data): {', '.join(skipped)}")
        return

    fig.update_layout(
        xaxis_title="Episode",
        yaxis_title=f"{metric_label} (rolling {roll})",
        height=500, margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(font_size=11),
    )
    st.plotly_chart(fig, use_container_width=True, key="cmp_lines")

    if skipped:
        st.caption(f"Skipped — no data: {', '.join(skipped)}")


_CONV_METRICS = {
    # label: (source, column, default_tol, tol_min, tol_max, tol_step, is_pct)
    "Success Rate (%)":  ("blackbox", "rolling_success_rate_100", 2.0, 0.5, 10.0, 0.5, True),
    "Path Efficiency":   ("planning", "planner_path_efficiency",  0.03, 0.01, 0.10, 0.01, False),
}


def _section_convergence():
    """🎯 Convergence — where a run plateaus, and how much budget was 'excess'."""
    run_index = scan_all_runs()  # (folder, run, label) across ALL folders
    if not run_index:
        st.info("← No runs detected in any CSV folder.")
        return
    labels = [label for _, _, label in run_index]
    label_to_fr = {label: (folder, run) for folder, run, label in run_index}

    st.caption(
        "Detects where each run **plateaus** (metric settles within a tolerance "
        "band of its final value) and reports how much of the run came *after* "
        "convergence — i.e. how much training budget was likely excess."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([4, 3, 2, 2])
    with c1:
        sel_runs = st.multiselect(
            "Runs (any folder)", labels,
            default=labels[:min(2, len(labels))],
            key="conv_runs",
            help="Pick any runs across folders — none vs full_imu, reward modes, stages.",
        )
    with c2:
        metric_label = st.selectbox(
            "Metric", list(_CONV_METRICS.keys()), key="conv_metric",
            help="Success Rate plateaus near the ceiling; Path Efficiency plateaus "
                 "lower and noisier — the real target of reward shaping.",
        )
    src, col, def_tol, tmin, tmax, tstep, is_pct = _CONV_METRICS[metric_label]
    with c3:
        tol = st.slider(
            "Plateau tolerance", tmin, tmax, def_tol, tstep, key="conv_tol",
            help="How close to the final value counts as 'converged'. "
                 "Units match the metric (percentage points or efficiency ratio).",
        )
    with c4:
        roll = st.slider("Rolling window", 1, 100, 20, key="conv_roll")

    if not sel_runs:
        st.info("← Select at least one run above.")
        return

    fig = go.Figure()
    table_rows: list[dict] = []
    skipped: list[str] = []
    # Path/trajectory metrics converge on SUCCESSFUL episodes only — including
    # failures would let a short doomed path flatter the curve.
    success_only = _is_path_metric(col)

    for i, run in enumerate(sel_runs):
        color = _clr(i)
        folder, run_name = label_to_fr[run]  # `run` is the folder-qualified label
        if src == "planning":
            df = load_pl_in(folder, run_name)
            if df.empty or col not in df.columns:
                skipped.append(run)
                continue
            ok_mask = _planner_valid_mask(df, col, success_only)
            series = pd.to_numeric(df[col], errors="coerce").where(ok_mask)
            x = df["episode"] if "episode" in df.columns else pd.RangeIndex(len(df))
        else:
            df = load_bb_in(folder, run_name)
            if df.empty or col not in df.columns:
                skipped.append(run)
                continue
            if success_only:
                sm = _success_mask(df)
                df = df[sm] if sm is not None else df
                if df.empty:
                    skipped.append(run)
                    continue
            series = pd.to_numeric(df[col], errors="coerce")
            x = df["_idx"] if "_idx" in df.columns else pd.RangeIndex(len(df))

        rolled = series.rolling(roll, min_periods=1).mean()
        if rolled.dropna().empty:
            skipped.append(run)
            continue

        n_eps = len(df)
        conv_idx, plateau = _convergence_point(rolled, tol)

        fig.add_trace(go.Scatter(
            x=x, y=rolled, mode="lines", name=run,
            line=dict(color=color, width=1.8),
        ))

        conv_ep = pct = excess = None
        if conv_idx is not None:
            # conv_idx is a label index into rolled; map to its episode/x value.
            conv_ep = int(x.loc[conv_idx]) if conv_idx in x.index else int(conv_idx)
            pos = rolled.index.get_loc(conv_idx)
            pct = round((pos + 1) / n_eps * 100, 1)
            excess = round(100 - pct, 1)
            # Plateau line + convergence marker.
            if plateau is not None:
                fig.add_hline(y=plateau, line=dict(color=color, dash="dot"),
                              opacity=0.25)
            fig.add_trace(go.Scatter(
                x=[x.loc[conv_idx] if conv_idx in x.index else conv_idx],
                y=[rolled.loc[conv_idx]],
                mode="markers", showlegend=False,
                marker=dict(color=color, size=13, symbol="star",
                            line=dict(color="white", width=1)),
                hovertext=f"{run}: converged @ ep {conv_ep} ({pct}% of run)",
                hoverinfo="text",
            ))

        table_rows.append({
            "run": run,
            "episodes": n_eps,
            "plateau": round(plateau, 3) if plateau is not None else None,
            "converged_at_episode": conv_ep,
            "pct_of_run": pct,
            "excess_pct": excess,
        })

    if not fig.data:
        st.caption(f"*No `{col}` data found for the selected runs.*")
        if skipped:
            st.caption(f"Skipped (no data): {', '.join(skipped)}")
        return

    unit = "%" if is_pct else "ratio"
    fig.update_layout(
        xaxis_title="Episode",
        yaxis_title=f"{metric_label} (rolling {roll}, {unit})",
        height=500, margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(font_size=11),
    )
    st.plotly_chart(fig, use_container_width=True, key="conv_lines")
    st.caption("★ = convergence point (where the curve settles into the plateau band).")

    if table_rows:
        st.subheader("Convergence Summary")
        st.caption(
            "`excess_pct` = share of episodes that ran *after* convergence — "
            "a rough estimate of wasted budget. High excess_pct ⇒ you could train "
            "with fewer `--steps` for the same result on this stage."
        )
        st.dataframe(pd.DataFrame(table_rows), width="stretch",
                     hide_index=True, key="conv_table")

    if skipped:
        st.caption(f"Skipped — no data: {', '.join(skipped)}")


def _section_commands():
    """📖 Commands — terminal cheat-sheet for training, BO, and validation runs."""
    st.caption(
        "Copy-paste terminal commands for every kind of run (each `st.code` block "
        "has a copy button). Replace the `1` / `stage1` parts with your target "
        "stage (1–8) — the running Gazebo stage must match `--stage`."
    )
    st.warning(
        "**⚠ NEW kinematics default (branch `path-efficiency-hybrid`):** "
        "`--max_angular_vel` now defaults to **1.0 rad/s** ⇒ **min turn radius = "
        "`max_linear_vel/max_angular_vel` = 0.1/1.0 = 0.1 m** (the robot can near-pivot → "
        "tight paths). The **OLD** setup was **0.2 rad/s** (radius **0.5 m**) — the cause of "
        "the CCW-looping / low path efficiency. This **changes the default MDP**: new runs "
        "are **NOT comparable** to old 0.2-rad/s runs. **Pass `--max_angular_vel 0.2`** to "
        "reproduce the legacy behaviour or resume an old run. Path efficiency is now also "
        "measured by the fair **Hybrid-A\\*** metric (`planner_path_efficiency_hybrid`, "
        "nonholonomic + obstacle-aware, never blank) — the **blue** path on every PNG."
    )
    st.info(
        "**Recent defaults (2026-06-12):**  `eval_every` = **20000** (eval interval "
        "in steps — the old hardcoded `% 4` was removed; lower it for a finer "
        "curve).  BO `--timeout-per-trial` = **24 h** (86400 s) so a trial is never "
        "cut off unnoticed.  BO `SCORE_WINDOW` = **60** (scores the last 3 trained "
        "evals) — so **use `--steps ≥80000` for BO** (5 evals; excludes the untrained "
        "first eval). A 40k trial (3 evals) would pollute the score with the "
        "untrained eval."
    )

    # ── 0. Gazebo ─────────────────────────────────────────────────────────────
    st.subheader("0 · Start Gazebo first (separate terminal)")
    st.markdown(
        "Every training / BO run needs Gazebo up in its own terminal. Use the "
        "**full path** — `turtlebot3_gazebo/` is not a built ROS2 package."
    )
    st.code(
        "export TURTLEBOT3_MODEL=burger\n"
        "# add gui:=false for HEADLESS (recommended on the eGPU — see note below)\n"
        "ros2 launch ~/turtlebot-dreamerv3/turtlebot3_gazebo/launch/turtle_stage1.py gui:=false",
        language="bash",
    )
    st.code(
        "# verify Gazebo is ready\n"
        "ros2 service list | grep reset\n"
        "ros2 topic list | grep -E \"/odom|/scan|/cmd_vel|/imu\"",
        language="bash",
    )
    st.caption(
        "Expect `/reset_simulation` from the service check and all four topics from the topic check. "
        "`/imu` is only needed for `odometry_mode=full_imu`. "
        "If a separate terminal sees no topics, run `export ROS_LOCALHOST_ONLY=1` to match the launch env."
    )
    st.markdown(
        "**Prove the program is subscribed to the ROS2 topics (panel demo).** "
        "Needs Gazebo up **and** a live `dreamer.py` run (the node `trainer_node` only "
        "exists while training/eval is running). One headline command shows the whole wiring:"
    )
    st.code(
        "# all subscriptions + publishers of our program in one view\n"
        "ros2 node info /trainer_node",
        language="bash",
    )
    st.code(
        "# per-topic proof — --verbose prints the node name of each endpoint\n"
        "ros2 topic info /scan    --verbose   # subscriber = trainer_node\n"
        "ros2 topic info /odom    --verbose   # subscriber = trainer_node\n"
        "ros2 topic info /imu     --verbose   # subscriber = trainer_node (full_imu run only)\n"
        "ros2 topic info /cmd_vel --verbose   # trainer_node = PUBLISHER here (action out)",
        language="bash",
    )
    st.caption(
        "`trainer_node` **subscribes** to the 3 sensors — `/scan` (LiDAR), `/odom` (odometry), "
        "`/imu` (IMU) — and **publishes** the velocity command to `/cmd_vel` (the action). "
        "`/cmd_vel` is *published, not subscribed* — frame it as the sense→act loop. "
        "`/imu` appears under Subscribers **only** when an `--odometry_mode full_imu` run is live; "
        "`none/twist/delta/full` show just `/scan` + `/odom` (IMU sub is `full_imu`-only by design). "
        "`trainer_node` appears twice (train env + eval env), so each subscribed topic shows count 2."
    )
    st.warning(
        "**Headless on the eGPU.** `gzclient` (the GUI) renders via OpenGL on the GPU; on the "
        "Thunderbolt eGPU that competes with CUDA training and can drop the link mid-run "
        "(`No CUDA GPUs are available` → every BO trial crashes). Launch with **`gui:=false`** "
        "for long/BO runs. To drop the GUI of an already-running sim: `pkill -f gzclient` "
        "(gzserver/training keep going)."
    )
    st.info(
        "**Runs auto-organize by mode+stage (2026-06-12):** every `dreamer.py` run writes "
        "its CSVs/plots to `csv_logs/{odometry_mode}_stage{N}/` and "
        "`path_plots/{odometry_mode}_stage{N}/` automatically (e.g. `none_stage1/`, "
        "`full_imu_stage1/`) — no `--csv_dir` needed. BO trials go to "
        "`csv_logs/tune_stage{N}_{mode}/`. Pick the folder in the sidebar."
    )

    # ── 1. Smoke test ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("1 · Smoke test (verify the pipeline, ~minutes)")
    st.code(
        """cd ~/turtlebot-dreamerv3/dreamerv3-torch
python3 dreamer.py --configs turtle --task turtle \\
  --logdir ./logdir/gpu_smoke_stage1_none_seed0 \\
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \\
  --device cuda --steps 5000 --eval_episode_num 2""",
        language="bash",
    )

    # ── 2. Full / long run ────────────────────────────────────────────────────
    st.divider()
    st.subheader("2 · Full / long training run")
    st.markdown("**Default reward** (original sparse baseline):")
    st.code(
        """cd ~/turtlebot-dreamerv3/dreamerv3-torch
python3 dreamer.py --configs turtle --task turtle \\
  --logdir ./logdir/stage1_360_none_seed0_reward_default \\
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \\
  --device cuda --steps 300000 --eval_episode_num 100 \\
  --reward_mode default""",
        language="bash",
    )
    st.markdown(
        "**Kinematics A/B** (the headline lever) — same everything, vary only "
        "`--max_angular_vel`: **1.0** (default, radius 0.1 m, capable) vs **0.2** "
        "(legacy, radius 0.5 m). Fresh logdir each; compare `planner_path_efficiency_hybrid`:"
    )
    st.code(
        """cd ~/turtlebot-dreamerv3/dreamerv3-torch
# NEW default kinematics (1.0 rad/s) — no flag needed
python3 dreamer.py --configs turtle --task turtle \\
  --logdir ./logdir/stage1_360_none_seed0_reward_default_ang10 \\
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \\
  --device cuda --steps 300000 --eval_episode_num 100 --reward_mode default

# LEGACY kinematics (0.2 rad/s, radius 0.5 m) — the old looping setup
python3 dreamer.py --configs turtle --task turtle \\
  --logdir ./logdir/stage1_360_none_seed0_reward_default_ang02 \\
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \\
  --device cuda --steps 300000 --eval_episode_num 100 --reward_mode default \\
  --max_angular_vel 0.2""",
        language="bash",
    )
    st.markdown("**Shaped reward** (additive shaping, mild defaults):")
    st.code(
        """cd ~/turtlebot-dreamerv3/dreamerv3-torch
python3 dreamer.py --configs turtle --task turtle \\
  --logdir ./logdir/stage1_360_none_seed0_reward_shaped \\
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \\
  --device cuda --steps 300000 --eval_episode_num 100 \\
  --reward_mode shaped""",
        language="bash",
    )
    st.markdown("**Shaped reward with tuned weights** (post-BO validation):")
    st.code(
        """cd ~/turtlebot-dreamerv3/dreamerv3-torch
python3 dreamer.py --configs turtle --task turtle \\
  --logdir ./logdir/stage1_360_none_seed0_reward_tuned \\
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \\
  --device cuda --steps 300000 --eval_episode_num 100 \\
  --reward_mode shaped \\
  --reward_progress_scale <v> --reward_step_penalty <v> --reward_turn_penalty <v> \\
  --reward_near_obstacle_scale <v> --reward_near_obstacle_sigma <v>""",
        language="bash",
    )
    st.markdown(
        "**PBRS reward** (potential-based, policy-invariant — `+100/−10` terminal "
        "unchanged, plus `F = pbrs_scale·(γ·Φ(s′) − Φ(s))` from the relative goal "
        "distance/angle; A\\* is **not** used). Defaults shown; all CLI-overridable. "
        "Keep `--pbrs_gamma` equal to the agent discount (0.997). Logs the five "
        "`sum_pbrs*` columns in `reward_*.csv`:"
    )
    st.code(
        """cd ~/turtlebot-dreamerv3/dreamerv3-torch
python3 dreamer.py --configs turtle --task turtle \\
  --logdir ./logdir/stage1_360_none_seed0_reward_pbrs \\
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \\
  --device cuda --steps 300000 --eval_episode_num 100 \\
  --reward_mode pbrs \\
  --pbrs_scale 1.0 --pbrs_distance_weight 1.0 --pbrs_angle_weight 0.2 \\
  --pbrs_distance_scale 5.0 --pbrs_gamma 0.997""",
        language="bash",
    )
    st.markdown(
        "**Odometry / IMU ablation** — swap `--odometry_mode` for `twist` / `delta` / "
        "`full` / `full_imu`. `full_imu` (7-dim: `full` + 2D `/imu` linear acceleration) "
        "needs no Gazebo change — `/imu` is already published. Use a **fresh logdir** "
        "(`.npz`/checkpoints are not compatible across modes); CSVs auto-organize into "
        "`csv_logs/full_imu_stage1/`:"
    )
    st.code(
        """cd ~/turtlebot-dreamerv3/dreamerv3-torch
python3 dreamer.py --configs turtle --task turtle \\
  --logdir ./logdir/stage1_360_full_imu_seed0 \\
  --stage 1 --lidar 360 --odometry_mode full_imu --seed 0 \\
  --device cuda --steps 300000 --eval_episode_num 100""",
        language="bash",
    )

    # ── Fixed-goal experiment ─────────────────────────────────────────────────
    st.markdown(
        "**Fixed-goal experiment** (diagnostic path-efficiency test) — pin the goal to "
        "a small **fixed set** instead of random sampling, so path efficiency is measured "
        "on the *same* navigation problem(s) every episode. `--fixed_goals` is a "
        "`;`-separated `x,y` list (any count: 1, 4, 5, …); it **composes with any** "
        "`--odometry_mode` and `--reward_mode`. `--fixed_goals_random` picks the order: "
        "`False` = round-robin (default, balanced), `True` = random pick from the set. "
        "Use a **fresh logdir** with a `_fixedgoal` suffix. Every episode is tagged with "
        "a coordinate-derived `goal_id` in `planning_*`/`blackbox_*.csv` so you can group "
        "and compare per goal (Path Efficiency tab → *Group by goal_id*)."
    )
    st.markdown("**Step 0 — preview the goals first** (no Gazebo/GPU; red = A\\*-blocked):")
    st.code(
        """cd ~/turtlebot-dreamerv3/dreamerv3-torch
# Stage 1 (empty — sanity, optimal ≈ straight line)
python3 preview_fixed_goals.py --stage 1 \\
  --fixed_goals "1.7,1.7;-1.7,1.7;1.7,-1.7;-1.7,-1.7"
# Stage 4 (cluttered — goals require detours)
python3 preview_fixed_goals.py --stage 4 \\
  --fixed_goals "2.0,2.0;-2.0,-2.0;2.0,-2.0;-2.0,2.0"
# → path_plots/fixed_goals_preview_stage{N}.png""",
        language="bash",
    )
    st.markdown(
        "**The 3-way comparison** — same `--fixed_goals`, vary only `--reward_mode` "
        "(`default` / `shaped` / `pbrs`). `--odometry_mode` composes freely "
        "(`none` shown; swap for `full` / `full_imu` with a fresh logdir):"
    )
    st.code(
        """cd ~/turtlebot-dreamerv3/dreamerv3-torch
GOALS="2.0,2.0;-2.0,-2.0;2.0,-2.0;-2.0,2.0"   # stage 4

# default (sparse baseline)
python3 dreamer.py --configs turtle --task turtle \\
  --logdir ./logdir/stage4_360_none_seed0_reward_default_fixedgoal \\
  --stage 4 --lidar 360 --odometry_mode none --seed 0 \\
  --device cuda --steps 300000 --eval_episode_num 100 \\
  --reward_mode default --fixed_goals "$GOALS"

# pbrs (potential-based) — same goals, fresh logdir
python3 dreamer.py --configs turtle --task turtle \\
  --logdir ./logdir/stage4_360_none_seed0_reward_pbrs_fixedgoal \\
  --stage 4 --lidar 360 --odometry_mode none --seed 0 \\
  --device cuda --steps 300000 --eval_episode_num 100 \\
  --reward_mode pbrs --fixed_goals "$GOALS"

# random order from the set instead of round-robin: add
#   --fixed_goals_random True""",
        language="bash",
    )

    st.info(
        "**Eval overhead on long runs:** eval fires every `--eval_every` (default "
        "**20000**) steps, so a 300k run ≈ 16 evals and 600k ≈ 31 evals. At "
        "`--eval_episode_num 100` that is ~1,600–3,100 eval episodes/plots. Cap it "
        "with a smaller `--eval_episode_num` and/or a larger `--eval_every`; for a "
        "**finer** curve on short runs, **lower** `--eval_every` (e.g. `5000`). "
        "With no `--steps`, the `turtle` default is **600000**."
    )

    # ── 3. BO from scratch ────────────────────────────────────────────────────
    st.divider()
    st.subheader("3 · Bayesian Optimization (BO) from scratch")
    st.markdown(
        "Needs **two terminals**. **Terminal 1 — Gazebo** (stays up the whole study; "
        "use `gui:=false` — a multi-hour BO on the eGPU is exactly where the GUI's "
        "OpenGL load drops the link):"
    )
    st.code(
        "export TURTLEBOT3_MODEL=burger\n"
        "ros2 launch ~/turtlebot-dreamerv3/turtlebot3_gazebo/launch/turtle_stage1.py gui:=false",
        language="bash",
    )
    st.markdown(
        "**Terminal 2 — the tuner** (run the steps in order). The BO now **co-tunes "
        "`actor_entropy`** (exploration) with the reward weights (step/turn widened to "
        "0–0.1) and **scores on the fair Hybrid-A\\* efficiency**, on the new 1.0-rad/s "
        "kinematics. ⚠ **Use a fresh `--study-name`** (e.g. `reward_stage1_none_eff`) — the "
        "search space changed, so resuming an old study would corrupt its TPE model:"
    )
    st.code(
        """cd ~/turtlebot-dreamerv3/dreamerv3-torch

# (optional) preview the exact dreamer.py commands — no training, no Gazebo needed
python3 tune_reward.py --stage 1 --n-trials 2 --dry-run --study-name reward_stage1_none_eff

# 1) default-reward baseline → sets the success-rate constraint floor (new kinematics)
python3 tune_reward.py --stage 1 --run-baseline --steps 80000 --eval-episode-num 20 \\
  --study-name reward_stage1_none_eff

# 2) the search: 40 trials, each an 80k-step run; tunes actor_entropy + reward weights
python3 tune_reward.py --stage 1 --n-trials 40 --steps 80000 --eval-episode-num 20 \\
  --study-name reward_stage1_none_eff

# tune under a specific obs space (e.g. IMU) — namespaces the study/db/csv by mode:
python3 tune_reward.py --stage 1 --odometry-mode full_imu --n-trials 40 --steps 80000 \\
  --eval-episode-num 20 --study-name reward_stage1_full_imu_eff""",
        language="bash",
    )
    st.caption(
        "`--odometry-mode` (default `none`) sets the obs space and **always** namespaces the "
        "study/db/csv/plots by mode. Resumes automatically — re-run the **same** command "
        "(same `--study-name`) after a crash/reboot and it continues, not from trial 0. The "
        "db is keyed by stage+mode (`tune_reward_stage1_none.db`); the `_eff` study lives "
        "inside it alongside any older study. Sanity-test first with "
        "`--study-name smoke --n-trials 2 --steps 3000 --eval-episode-num 2`."
    )

    # ── 4. After BO ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("4 · After BO — export best config, then validate")
    st.code(
        """cd ~/turtlebot-dreamerv3/dreamerv3-torch

# write tune_trials_stage1.csv + tune_best_stage1.csv (read-only on the study DB)
python3 export_tune_results.py --stage 1
# for a mode-specific study, match the mode (targets tune_stage1_full_imu/):
python3 export_tune_results.py --stage 1 --odometry-mode full_imu

# then plug the printed --reward_* weights into the 'tuned' full run in section 2,
# repeat for seeds 0,1,2 on stage 1, then stages 2–4 for the transfer test.""",
        language="bash",
    )

    # ── 5. Monitoring tools ───────────────────────────────────────────────────
    st.divider()
    st.subheader("5 · Monitoring tools")
    st.markdown("**Dashboard** — run this Streamlit app:")
    st.code(
        "cd ~/turtlebot-dreamerv3/dreamerv3-torch\n"
        "streamlit run dashboard/app.py",
        language="bash",
    )
    st.markdown("**GPU monitor — `nvtop`** (per-process GPU/CPU/mem, live):")
    st.code("nvtop", language="bash")
    st.caption(
        "`nvtop` shows every GPU process with live utilisation, memory, and power. "
        "Install with: `sudo apt install nvtop`"
    )
    st.markdown("**GPU snapshot — `nvidia-smi`** (one-shot or watch mode):")
    st.code(
        "# one-shot snapshot\n"
        "nvidia-smi\n\n"
        "# live — refresh every 1 second\n"
        "watch -n 1 nvidia-smi\n\n"
        "# compact per-process table\n"
        "nvidia-smi pmon -s um -d 2",
        language="bash",
    )
    st.caption(
        "`nvidia-smi` reports device-wide GPU utilisation, memory, power, and temperature. "
        "`pmon` (`-s um` = utilisation + memory, `-d 2` = 2 s interval) shows a live "
        "per-process breakdown — useful for confirming that only DreamerV3 is on CUDA "
        "(Gazebo's `gzserver`/`gzclient` use OpenGL and do NOT appear in CUDA pmon)."
    )

    # ── Parameter reference ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Parameter reference — `dreamer.py`")
    dreamer_params = pd.DataFrame([
        ("--stage",                     "(required)",  "1–8; must match the running Gazebo stage"),
        ("--logdir",                    "(required)",  "output dir; run_name = its last component"),
        ("--lidar",                     "360",         "LiDAR beam count"),
        ("--odometry_mode",             "none",        "none / twist / delta / full / full_imu (full_imu = full + 2D /imu accel)"),
        ("--seed",                      "0",           "RNG seed"),
        ("--device",                    "cpu",         "pass `cuda` to train on GPU"),
        ("--steps",                     "600000",      "total training steps (turtle default; 300k common)"),
        ("--eval_every",                "20000",       "eval interval in steps (single knob; lower = finer curve)"),
        ("--eval_episode_num",          "100",         "episodes per eval (2 = smoke; 100 = reporting)"),
        ("--reward_mode",               "default",     "default (sparse) / shaped (additive) / pbrs (potential-based, policy-invariant)"),
        ("--reward_progress_scale",     "1.0",         "shaped: +scale·(prev_dist − curr_dist)"),
        ("--reward_step_penalty",       "0.01",        "shaped: −c per step"),
        ("--reward_turn_penalty",       "0.01",        "shaped: −k·|ang_vel_cmd|"),
        ("--reward_near_obstacle_scale","0.1",         "shaped: near-obstacle penalty scale"),
        ("--reward_near_obstacle_sigma","0.25",        "shaped: near-obstacle penalty σ"),
        ("--pbrs_scale",                "1.0",         "pbrs: overall weight on F = pbrs_scale·(γ·Φ(s′) − Φ(s))"),
        ("--pbrs_distance_weight",      "1.0",         "pbrs: weight of distance term in Φ"),
        ("--pbrs_angle_weight",         "0.2",         "pbrs: weight of heading-alignment term in Φ"),
        ("--pbrs_distance_scale",       "5.0",         "pbrs: distance normaliser (m); fixed, not stage-aware"),
        ("--pbrs_gamma",                "0.997",       "pbrs: discount; keep == agent discount (0.997) for policy invariance"),
        ("--fixed_goals",               "''",          "'' = random goals (default). 'x1,y1;x2,y2;…' = fixed set; any stage 1–8; composes with any --odometry_mode / --reward_mode"),
        ("--fixed_goals_random",        "False",       "fixed-goal order: False = round-robin through the set (default); True = random pick each episode"),
        ("--max_angular_vel",           "1.0",         "⚠ turn-rate cap (rad/s). min turn radius = max_linear_vel/this = 0.1/1.0 = 0.1 m (NEW default, robot can near-pivot → tight paths). OLD setup = 0.2 (radius 0.5 m) = the CCW-looping / low-efficiency cause. Pass 0.2 to reproduce the legacy MDP (new runs ≠ old 0.2 runs)"),
        ("--max_linear_vel",            "0.1",         "forward-speed cap (m/s) on |action[0]|; unchanged"),
        ("--actor_entropy",             "-1.0",        "exploration: overrides actor.entropy (default 3e-4) when ≥0; higher = sustained exploration (delays policy collapse). -1.0 = untouched. Searched by the BO"),
        ("--prefill",                   "500",         "random steps before training (once)"),
        ("--time_limit",                "250",         "max steps per episode before timeout"),
        ("--resource_logging",          "true",        "CPU/RAM/GPU logging (False to disable)"),
        ("--csv_dir",                   "./csv_logs",  "base for CSVs; auto → ./csv_logs/{mode}_stage{N}/ unless overridden"),
        ("--plots_dir",                 "./path_plots","base for path-plot PNGs; auto → ./path_plots/{mode}_stage{N}/ unless overridden"),
    ], columns=["flag", "default", "notes"])
    st.dataframe(dreamer_params, width="stretch", hide_index=True, key="cmd_params_dreamer")

    st.subheader("Parameter reference — `tune_reward.py` (BO)")
    bo_params = pd.DataFrame([
        ("--stage",             "1",                        "stage to tune"),
        ("--n-trials",          "30",                       "candidate weight sets to try"),
        ("--steps",             "80000",                    "per-trial proxy budget (≥80k so SCORE_WINDOW=60 skips the untrained eval)"),
        ("--eval-episode-num",  "20",                       "eval episodes per trial"),
        ("--seed",              "0",                        "RNG seed"),
        ("--device",            "cuda",                     "training device"),
        ("--odometry-mode",     "none",                     "obs space to tune under; non-none namespaces study/db/csv by mode (e.g. tune_stage{N}_full_imu)"),
        ("--margin",            "5.0",                      "allowed success drop vs baseline (pts)"),
        ("--timeout-per-trial", "86400",                    "sec (24 h); a timed-out trial is scored on partial data"),
        ("--run-baseline",      "(flag)",                   "run the default-reward baseline (sets the floor)"),
        ("--baseline-success",  "None",                     "skip baseline; give the floor success % directly"),
        ("--csv-dir",           "./csv_logs/tune_stage{N}_{mode}", "per-stage-per-mode CSV subfolder (incl. _none)"),
        ("--study-name",        "reward_stage{N}_{mode}", "Optuna study + db name (mode-suffixed, incl. _none)"),
        ("--dry-run",           "(flag)",                   "print the dreamer.py commands only; no training"),
    ], columns=["flag", "default", "notes"])
    st.dataframe(bo_params, width="stretch", hide_index=True, key="cmd_params_bo")

    st.caption(
        "BO search space now co-tunes **actor_entropy** (log 1e-4–1e-2, exploration) with the "
        "reward weights (step/turn widened to 0–0.1); it **scores on the Hybrid-A\\* efficiency** "
        "(`planner_path_efficiency_hybrid` — the fair nonholonomic + obstacle-aware metric, never "
        "blank). Trials run on the new 1.0-rad/s kinematics default. "
        "Full reference: see `CLAUDE.md` (Reward-Weight Tuning, Example Commands) "
        "and `docs/evaluation_loop.md` for the eval cadence."
    )


# Trial034 validation-protocol weights, exactly as documented in CLAUDE.md
# "Validation Protocol" / Example Commands. 'default' emits no extra --reward_*
# flags (--reward_mode default, original sparse reward, byte-for-byte unchanged).
_REWARD_PRESETS = {
    "default": {},
    "trial034_eff_tuned": {
        "reward_progress_scale": "1.73768",
        "reward_step_penalty": "0.037479",
        "reward_turn_penalty": "0.0160898",
        "reward_near_obstacle_scale": "0.000422429",
        "reward_near_obstacle_sigma": "0.454763",
        "actor_entropy": "0.000102611",
    },
}


def _section_training_commands():
    """Display-only command builder — never executes anything. Every action
    (start/resume training, scan corrupt episodes, repair CSV NULs, delete a
    logdir) stays a copy-ready st.code() block for the user's own terminal."""
    st.subheader("🆘 Training Commands / Crash Resume")
    st.caption(
        "Copy-ready terminal commands for starting, resuming, and recovering a "
        "training run. Nothing on this page executes anything — copy a command "
        "into your own terminal to run it."
    )

    st.warning(
        "- To **resume**, rerun the *exact same* training command with the "
        "*same* `--logdir` below.\n"
        "- Use a **new** `--logdir` for clean thesis A/B validation runs.\n"
        "- Do **not** delete a logdir that is currently training.\n"
        "- Do **not** run `rm -rf` from inside the dashboard — copy the command "
        "and run it yourself in a terminal.\n"
        "- Do **not** commit `logdir/`, `csv_logs/`, `path_plots/`, `*.db`, "
        "`*.npz`, `*.pt`, `*.pth`, `*.tmp`, or `*.bak` files."
    )

    st.markdown("#### Run configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        logdir_input = st.text_input(
            "logdir", value="./logdir/stage4_360_full_imu_seed0_eff_tuned_fixedgoal_v2",
            key="tc_logdir",
            help="Same value passed to --logdir. Rerunning with this exact path "
                 "resumes; a different path starts a fresh run.",
        )
        stage = st.number_input("stage", min_value=1, max_value=8, value=4, step=1, key="tc_stage")
        odometry_mode = st.selectbox(
            "odometry_mode", ["none", "twist", "delta", "full", "full_imu"],
            index=4, key="tc_odometry_mode",
        )
    with col2:
        seed = st.number_input("seed", min_value=0, value=0, step=1, key="tc_seed")
        checkpoint_every = st.number_input(
            "checkpoint_every", min_value=0, value=10000, step=1000, key="tc_checkpoint_every",
            help="0 disables interval checkpointing (checkpoint only at the "
                 "eval_every boundary). See CLAUDE.md Crash-Safe Manual Resume Workflow.",
        )
        eval_episode_num = st.number_input(
            "eval_episode_num", min_value=1, value=100, step=10, key="tc_eval_episode_num",
        )
    with col3:
        steps = st.number_input("steps", min_value=1000, value=300000, step=10000, key="tc_steps")
        reward_preset = st.selectbox(
            "reward preset", list(_REWARD_PRESETS.keys()), key="tc_reward_preset",
        )
        fixed_goals = st.text_input(
            "fixed_goals", value="2.0,2.0;-2.0,-2.0;2.0,-2.0;-2.0,2.0",
            key="tc_fixed_goals",
            help="';'-separated 'x,y' pairs. Leave blank for normal random goals.",
        )

    csv_dir_input = st.text_input(
        "csv_dir (for CSV NUL repair below)", value=str(CSV_DIR), key="tc_csv_dir",
        help="Defaults to the dashboard's active CSV folder (sidebar picker). "
             "Override if the crashed run used a different --csv_dir.",
    )

    st.divider()

    # ── A. Start Gazebo ──────────────────────────────────────────────────
    st.markdown(f"#### A. Start Gazebo — stage {int(stage)}, headless")
    st.code(
        "export TURTLEBOT3_MODEL=burger\n"
        f"ros2 launch ~/turtlebot-dreamerv3/turtlebot3_gazebo/launch/turtle_stage{int(stage)}.py gui:=false",
        language="bash",
    )

    # ── B. Start / resume DreamerV3 ──────────────────────────────────────
    st.markdown("#### B. Start / resume DreamerV3")
    st.caption(
        "Rerunning this exact command (same --logdir) resumes automatically — "
        "check the resume history table at the bottom of this page to confirm."
    )
    lines = [
        "cd ~/turtlebot-dreamerv3/dreamerv3-torch",
        "python3 dreamer.py --configs turtle --task turtle \\",
        f"  --logdir {logdir_input} \\",
        f"  --stage {int(stage)} --lidar 360 --odometry_mode {odometry_mode} --seed {int(seed)} \\",
        f"  --device cuda --steps {int(steps)} --eval_episode_num {int(eval_episode_num)} \\",
        f"  --checkpoint_every {int(checkpoint_every)} \\",
    ]
    if reward_preset == "default":
        lines.append("  --reward_mode default \\")
    else:
        lines.append("  --reward_mode shaped \\")
        for flag, val in _REWARD_PRESETS[reward_preset].items():
            lines.append(f"  --{flag} {val} \\")
    lines.append(f'  --fixed_goals "{fixed_goals}"')
    st.code("\n".join(lines), language="bash")

    st.divider()

    # ── C/D. Corrupt episode scan ────────────────────────────────────────
    st.markdown("#### C. After a crash — scan for corrupt episode files (read-only)")
    st.code(f"python3 scripts/scan_corrupt_episodes.py --logdir {logdir_input}", language="bash")

    st.markdown("#### D. Optional — clean up the corrupt files found above")
    st.caption("Only run after reviewing C's output. Prints exactly what it removes before removing anything.")
    st.code(f"python3 scripts/scan_corrupt_episodes.py --logdir {logdir_input} --clean_corrupt_eps", language="bash")

    st.divider()

    # ── E/F. CSV NUL repair ──────────────────────────────────────────────
    st.markdown("#### E. CSV NUL-padding repair — dry run (reports only, no changes)")
    st.code(f"python3 scripts/repair_csv_nuls.py --csv-dir {csv_dir_input}", language="bash")

    st.markdown("#### F. CSV NUL-padding repair — apply (backs up each file to `.bak` first)")
    st.code(f"python3 scripts/repair_csv_nuls.py --csv-dir {csv_dir_input} --apply", language="bash")

    st.divider()

    # ── G. Safe delete ───────────────────────────────────────────────────
    st.markdown("#### G. Safely delete an old crashed logdir")
    st.caption(
        "Manual, three-step, terminal-only: list before, delete, list after — "
        "so you can visually confirm what's gone before and after. Never run "
        "`rm -rf` without checking the listing first."
    )
    if not logdir_input.strip():
        st.info("Enter a logdir above to generate the delete-safety commands.")
    else:
        logdir_path = Path(logdir_input)
        parent_dir = str(logdir_path.parent) if str(logdir_path.parent) not in ("", ".") else "./logdir"
        run_name = logdir_path.name
        st.code(
            f"ls -lh {parent_dir} | grep {run_name}\n"
            f"rm -rf {logdir_input}\n"
            f"ls -lh {parent_dir} | grep {run_name}",
            language="bash",
        )

    st.divider()

    # ── Resume history ───────────────────────────────────────────────────
    st.markdown("#### Resume history for this logdir")
    resume_df = load_resume_events(logdir_input)
    if resume_df.empty:
        st.info("No resume_events.csv found yet.")
    else:
        st.dataframe(resume_df, width="stretch", key=f"tc_resume_events_{logdir_input}")
        latest = resume_df.iloc[-1]
        st.success(
            f"**Latest resume event:** {latest.get('datetime', '?')} — "
            f"policy=`{latest.get('resume_policy', '?')}`, "
            f"checkpoint=`{latest.get('checkpoint_loaded', '?')}`, "
            f"step={latest.get('step', '?')}, "
            f"replay_episodes={latest.get('replay_episodes_loaded', '?')}, "
            f"corrupt_skipped={latest.get('corrupt_episodes_skipped', '?')}"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("DreamerV3 TurtleBot3 — Experiment Dashboard")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Experiment Runs")

        # Folder picker: csv_logs/ (main runs) or a BO-tune subfolder.
        global CSV_DIR, PLOTS_DIR
        folders = list_csv_folders()
        prev = st.session_state.get("_csv_folder", ".")
        if prev not in folders:
            prev = "."
        folder = st.selectbox(
            "CSV folder",
            folders,
            index=folders.index(prev),
            format_func=lambda f: "csv_logs/ (main runs)" if f == "." else f"tune: {f}/",
            help="Switch between the main csv_logs/ and the per-stage BO-tune subfolders.",
        )
        if folder != prev:
            st.session_state["_csv_folder"] = folder
            # Only clear the loaders that read the global CSV_DIR (their cache key
            # is run_name alone, so it can't tell folders apart). Folder-independent
            # caches (list_csv_folders, scan_all_runs, load_*_in) keep their TTL —
            # no need to discard them on every folder click.
            for _fn in (scan_runs, load_bb, load_wb, load_pl,
                       load_resource, load_resource_eval, load_tune_trials):
                _fn.clear()
        CSV_DIR   = CSV_BASE   if folder == "." else CSV_BASE   / folder
        PLOTS_DIR = PLOTS_BASE if folder == "." else PLOTS_BASE / folder

        all_runs = scan_runs()
        run_opts = []
        if not all_runs:
            # Non-fatal: keep the app alive so the 📖 Commands tab stays reachable
            # even when this folder has no CSVs (don't st.stop()).
            st.warning("No CSV files in this folder — pick another, "
                       "or open the 📖 Commands tab.")
        else:
            st.caption(f"{len(all_runs)} run(s) detected")
            # Phase filter — training runs vs their eval_* variants (eval CSVs surface
            # as eval_{run} run names). Filters the run list so training and eval don't
            # mix on one chart unless you choose Both. Applies to every tab.
            phase_view = st.radio(
                "Phase", ["Both", "Training", "Eval"],
                horizontal=True, key="sidebar_phase",
                help="Training = the blackbox_*/planning_* runs.  Eval = their "
                     "eval_* variants (evaluation checkpoints).  Both = show all "
                     "(train and eval mixed on the chart).",
            )
            if phase_view == "Training":
                run_opts = [r for r in all_runs if not r.startswith("eval_")]
            elif phase_view == "Eval":
                run_opts = [r for r in all_runs if r.startswith("eval_")]
            else:
                run_opts = all_runs
            if not run_opts:
                st.caption(f"*No {phase_view.lower()} runs in this folder.*")

        selected = st.multiselect(
            "Select runs",
            options=run_opts,
            default=run_opts[:min(2, len(run_opts))],
            key="run_multiselect",
            help="Tip: select multiple runs to compare them on the same chart. "
                 "Use the Phase filter above to show only training or only eval runs.",
        )

        st.divider()
        st.subheader("Chart Lines")
        show_cumulative = st.checkbox("Cumulative",  value=True)
        show_r100       = st.checkbox("Rolling 100", value=True)
        show_r500       = st.checkbox("Rolling 500", value=True)

        st.divider()
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        refresh_secs = st.slider(
            "Interval (s)", 10, 120, 30, step=10,
            disabled=not auto_refresh,
        )
        if st.button("⟳  Refresh now"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(f"CSVs: `{CSV_DIR}`\nPlots: `{PLOTS_DIR}`")

    # ── Page selector ────────────────────────────────────────────────────────
    # A radio (not st.tabs) is what makes this genuinely lazy: Streamlit reruns
    # the whole script every interaction and st.tabs() executes EVERY tab body
    # regardless of which one is visually open — with 9 sections (one, BO
    # Trials, independently touching 100+ trial CSVs) that meant every click
    # anywhere in the app re-ran every section. A radio's un-selected branches
    # are plain Python `elif`s that never execute at all.
    PAGES = [
        "📊 Overview", "📈 Blackbox Charts", "⚙️ Whitebox Charts",
        "📍 Path Efficiency", "⚡ Resource Cost", "🏁 BO Trials", "🔀 Compare",
        "🎯 Convergence", "📖 Commands", "🆘 Training Commands / Crash Resume",
    ]
    view = st.radio("View", PAGES, horizontal=True, key="main_view",
                    label_visibility="collapsed")
    st.divider()

    # No global st.stop() when nothing is selected — 📖 Commands must stay
    # reachable. Data-dependent pages below show a "select a run" hint instead.
    _NO_RUN_MSG = "← Select one or more runs from the sidebar to see this page."

    if view == "📊 Overview":
        if not selected:
            st.info(_NO_RUN_MSG)
        else:
            bb_data = {r: load_bb(r) for r in selected}
            wb_data = {r: load_wb(r) for r in selected}
            _section_summary(selected, bb_data, wb_data)
            st.divider()
            _section_comparison(selected, bb_data, wb_data)
            st.divider()
            n_per_ckpt = st.number_input(
                "Episodes per eval checkpoint (match --eval_episode_num)",
                min_value=1, max_value=500, value=100, step=10,
                key="ov_eval_ckpt_n",
                help="Set this to the --eval_episode_num used in your run (default 100). "
                     "Each block of this many eval episodes = one checkpoint.",
            )
            _section_eval_checkpoints(selected, int(n_per_ckpt))

    elif view == "📈 Blackbox Charts":
        if not selected:
            st.info(_NO_RUN_MSG)
        else:
            bb_data = {r: load_bb(r) for r in selected}
            _section_bb(selected, bb_data, show_cumulative, show_r100, show_r500)

    elif view == "⚙️ Whitebox Charts":
        if not selected:
            st.info(_NO_RUN_MSG)
        else:
            wb_data = {r: load_wb(r) for r in selected}
            _section_wb(selected, wb_data)

    elif view == "📍 Path Efficiency":
        if len(selected) == 0:
            st.info(_NO_RUN_MSG)
        elif len(selected) == 1:
            rn = selected[0]
            _section_planner(rn, load_pl(rn), load_bb(rn))
        else:
            for rn in selected:
                st.subheader(rn)
                _section_planner(rn, load_pl(rn), load_bb(rn))
                st.divider()

    elif view == "⚡ Resource Cost":
        if not selected:
            st.info(_NO_RUN_MSG)
        elif len(selected) == 1:
            rn = selected[0]
            _section_resource(rn, load_resource(rn), load_resource_eval(rn))
        else:
            for rn in selected:
                st.subheader(rn)
                _section_resource(rn, load_resource(rn), load_resource_eval(rn))
                st.divider()

    elif view == "🏁 BO Trials":
        _section_bo_trials(load_tune_trials())

    elif view == "🔀 Compare":
        _section_compare()

    elif view == "🎯 Convergence":
        _section_convergence()

    elif view == "📖 Commands":
        _section_commands()

    elif view == "🆘 Training Commands / Crash Resume":
        _section_training_commands()

    # ── Auto-refresh ─────────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(refresh_secs)
        # No cache_data.clear() here — CACHE_TTL (20s) already re-fetches stale
        # entries on next access; forcing a full clear every cycle re-triggers
        # every cached loader (including the BO-trial loop) from a cold cache
        # even when nothing changed. The manual "Refresh now" button below is
        # the explicit one-shot "I want it now" action and keeps its full clear.
        st.rerun()


if __name__ == "__main__":
    main()
