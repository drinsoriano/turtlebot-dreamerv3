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

CSV_DIR = Path(__file__).parent.parent / "csv_logs"

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

CACHE_TTL = 20  # seconds between cache invalidations


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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
    tab_ov, tab_bb, tab_wb = st.tabs(
        ["Overview", "Blackbox Charts", "Whitebox Charts"]
    )

    with tab_ov:
        _section_summary(selected, bb_data, wb_data)
        st.divider()
        _section_comparison(selected, bb_data, wb_data)

    with tab_bb:
        _section_bb(selected, bb_data, show_cumulative, show_r100, show_r500)

    with tab_wb:
        _section_wb(selected, wb_data)

    # ── Auto-refresh ─────────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(refresh_secs)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
