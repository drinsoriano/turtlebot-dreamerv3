"""
DreamerV3 TurtleBot3 — Experiment Dashboard
Run: streamlit run dashboard/app.py  (from dreamerv3-torch/)
"""
from __future__ import annotations

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

CSV_DIR   = Path(__file__).parent.parent / "csv_logs"
PLOTS_DIR = Path(__file__).parent.parent / "path_plots"

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


@st.cache_data(ttl=CACHE_TTL)
def load_bb(run_name: str) -> pd.DataFrame:
    p = CSV_DIR / f"blackbox_{run_name}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        df["_idx"] = range(len(df))
        if "path_efficiency" in df.columns and "path_directness" not in df.columns:
            df = df.rename(columns={"path_efficiency": "path_directness"})
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def load_wb(run_name: str) -> pd.DataFrame:
    p = CSV_DIR / f"whitebox_{run_name}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def load_pl(run_name: str) -> pd.DataFrame:
    p = CSV_DIR / f"planning_{run_name}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


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
    try:
        df = pd.read_csv(p, dtype=str)  # str to avoid mixed-type on blank GPU fields
        for col in RESOURCE_NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def load_resource_eval(run_name: str) -> pd.DataFrame:
    if run_name.startswith("eval_"):
        return pd.DataFrame()  # already an eval run; avoid resource_eval_eval_* lookup
    p = CSV_DIR / f"resource_eval_{run_name}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, dtype=str)
        for col in RESOURCE_NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


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


def _success_mask(df: pd.DataFrame):
    if df.empty or "steps_to_goal" not in df.columns:
        return None
    return df["steps_to_goal"] != -1


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


def _section_bb(selected: list, bb_data: dict, show_cumulative: bool, show_r100: bool, show_r500: bool):
    run_map = {r: bb_data[r] for r in selected}
    specs = [
        ("success_rate",      "Success Rate (%)",
         [("rolling_success_rate_100", "dash", "rolling-100"),
          ("rolling_success_rate_500", "dot",  "rolling-500")], False),
        ("collision_rate",    "Collision Rate (%)",
         [("rolling_collision_rate_100", "dash", "rolling-100"),
          ("rolling_collision_rate_500", "dot",  "rolling-500")], False),
        ("steps_to_goal",     "Steps to Goal  (successful episodes only)", None, True),
        ("path_directness",   "Path Directness  [0 – 1]",                  None, False),
        ("min_obstacle_dist", "Min Obstacle Distance (m)",                  None, False),
        ("near_collisions",   "Near Collisions",                            None, False),
    ]
    for col, title, extra, success_only in specs:
        fig = _bb_chart(run_map, col, title, extra, success_only,
                        show_cumulative=show_cumulative,
                        show_r100=show_r100,
                        show_r500=show_r500)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption(f"*{title} — column not available in selected runs*")


def _section_wb(selected: list, wb_data: dict):
    run_map = {r: wb_data[r] for r in selected}
    specs = [
        ("eval_return",                                "Eval Return"),
        (["eval_success_rate", "eval_collision_rate"], "Eval Success & Collision Rate (%)"),
        ("model_loss",                                 "Model Loss"),
        (["actor_loss", "value_loss"],                 "Actor & Value Loss"),
        (["kl", "prior_ent", "post_ent"],              "KL Divergence & Entropy"),
    ]
    for cols, title in specs:
        fig = _wb_chart(run_map, cols, title)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
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
        "**Process metrics** = DreamerV3 Python process only.  "
        "**Device metrics** = GPU device context (may include other GPU users)."
    )
    chart_specs = [
        ("episode_wall_time_sec",       "Episode Wall Time (s)  [process]"),
        ("ram_used_mb_process",         "RAM Used MB  [process]"),
        ("gpu_memory_used_mb_process",  "GPU Memory MB  [process — primary]"),
        ("gpu_memory_used_mb_device",   "GPU Memory MB  [device context]"),
        ("gpu_power_watts",             "GPU Power (W)  [device context]"),
        ("gpu_temperature_c",           "GPU Temperature (°C)  [device context]"),
    ]
    for y_col, title in chart_specs:
        fig = _resource_chart(df, x_col, y_col, title)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption(f"*{title} — not available or all values blank*")

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("Raw resource log", expanded=False):
        display_df = df.drop(columns=["_phase"], errors="ignore").copy()
        for col in display_df.select_dtypes(include="object").columns:
            display_df[col] = display_df[col].astype(str)
        st.dataframe(display_df, width="stretch")


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

    # ── Planner status breakdown ──────────────────────────────────────────────
    with st.expander("Planner status breakdown", expanded=False):
        status_counts = pl_df["planner_status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        st.dataframe(status_counts, width="stretch", hide_index=True)

    # ── Efficiency chart ──────────────────────────────────────────────────────
    st.subheader("Planner Path Efficiency vs Episode")
    roll_window = st.slider("Rolling average window", 5, 100, 25, key=f"roll_{run_name}")

    fig = go.Figure()
    if "episode" in pl_df.columns and "planner_path_efficiency" in pl_df.columns:
        fig.add_trace(go.Scatter(
            x=pl_df["episode"],
            y=pl_df["planner_path_efficiency"],
            mode="lines+markers",
            name="Efficiency (region)",
            marker=dict(size=3),
            line=dict(width=1, color=PALETTE[0]),
            connectgaps=False,
        ))
        rolled = pl_df["planner_path_efficiency"].rolling(roll_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=pl_df["episode"],
            y=rolled,
            mode="lines",
            name=f"Rolling {roll_window} (region)",
            line=dict(width=2, color=PALETTE[0], dash="dash"),
        ))
    if "planner_path_efficiency_center" in pl_df.columns:
        fig.add_trace(go.Scatter(
            x=pl_df["episode"],
            y=pl_df["planner_path_efficiency_center"],
            mode="lines+markers",
            name="Efficiency (centre)",
            marker=dict(size=3),
            line=dict(width=1, color="#9467bd"),
            connectgaps=False,
        ))
        rolled_cen = pl_df["planner_path_efficiency_center"].rolling(roll_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=pl_df["episode"],
            y=rolled_cen,
            mode="lines",
            name=f"Rolling {roll_window} (centre)",
            line=dict(width=2, color="#9467bd", dash="dot"),
        ))
    fig.update_layout(
        xaxis_title="Episode", yaxis_title="Efficiency",
        yaxis=dict(range=[0, 1.05]),
        height=360, margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(font_size=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Planning table ────────────────────────────────────────────────────────
    st.subheader("Planning Log")
    display_cols = [c for c in [
        "episode", "outcome",
        "planner_status", "planner_path_efficiency", "planner_path_efficiency_raw",
        "planner_status_center", "planner_path_efficiency_center", "planner_path_efficiency_center_raw",
        "initial_distance", "actual_path_length",
        "planned_path_length", "planned_path_length_center",
    ] if c in pl_df.columns]

    try:
        subset = ["outcome"] if "outcome" in display_cols else None
        base = pl_df[display_cols].style
        if subset:
            try:
                styled = base.map(_color_outcome_text, subset=subset)
            except AttributeError:
                styled = base.applymap(_color_outcome_text, subset=subset)
        else:
            styled = base
        st.dataframe(styled, width="stretch")
    except Exception:
        display_pl = pl_df[display_cols].copy()
        for col in display_pl.select_dtypes(include="object").columns:
            display_pl[col] = display_pl[col].astype(str)
        st.dataframe(display_pl, width="stretch")

    # ── Episode selector & PNG viewer ─────────────────────────────────────────
    st.subheader("Episode Path Plot")

    episodes = pl_df["episode"].tolist() if "episode" in pl_df.columns else []
    if not episodes:
        st.info("No episodes in planning CSV.")
        return

    selected_ep = st.selectbox(
        "Select episode",
        options=episodes,
        key=f"ep_sel_{run_name}",
    )

    ep_row = pl_df[pl_df["episode"] == selected_ep]
    outcome_val = str(ep_row["outcome"].iloc[0]) if not ep_row.empty and "outcome" in ep_row.columns else ""

    # PNG — full width
    _plot_dir = (run_name[len("eval_"):] + "_eval"
                 if run_name.startswith("eval_")
                 else run_name)
    png_path = _find_png(run_name, int(selected_ep), outcome_val)
    if png_path:
        st.image(str(png_path), width="stretch")
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("DreamerV3 TurtleBot3 — Experiment Dashboard")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Experiment Runs")
        all_runs = scan_runs()
        if not all_runs:
            st.error(f"No CSV files found in:\n`{CSV_DIR}`")
            st.stop()
        st.caption(f"{len(all_runs)} run(s) detected")

        selected = st.multiselect(
            "Select runs",
            options=all_runs,
            default=all_runs[:min(2, len(all_runs))],
            help="Tip: select multiple runs to compare them on the same chart",
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
        st.caption(f"Reading CSVs from:\n`{CSV_DIR}`")

    # ── Guard ────────────────────────────────────────────────────────────────
    if not selected:
        st.info("← Select one or more runs from the sidebar.")
        st.stop()

    # ── Load data ────────────────────────────────────────────────────────────
    bb_data = {r: load_bb(r) for r in selected}
    wb_data = {r: load_wb(r) for r in selected}

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_ov, tab_bb, tab_wb, tab_pl, tab_res = st.tabs([
        "📊 Overview", "📈 Blackbox Charts", "⚙️ Whitebox Charts",
        "📍 Path Efficiency", "⚡ Resource Cost",
    ])

    with tab_ov:
        _section_summary(selected, bb_data, wb_data)
        st.divider()
        _section_comparison(selected, bb_data, wb_data)

    with tab_bb:
        _section_bb(selected, bb_data, show_cumulative, show_r100, show_r500)

    with tab_wb:
        _section_wb(selected, wb_data)

    with tab_pl:
        if len(selected) == 0:
            st.info("← Select a run in the sidebar.")
        elif len(selected) == 1:
            rn = selected[0]
            _section_planner(rn, load_pl(rn), load_bb(rn))
        else:
            for rn in selected:
                st.subheader(rn)
                _section_planner(rn, load_pl(rn), load_bb(rn))
                st.divider()

    with tab_res:
        if len(selected) == 1:
            rn = selected[0]
            _section_resource(rn, load_resource(rn), load_resource_eval(rn))
        else:
            for rn in selected:
                st.subheader(rn)
                _section_resource(rn, load_resource(rn), load_resource_eval(rn))
                st.divider()

    # ── Auto-refresh ─────────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(refresh_secs)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
