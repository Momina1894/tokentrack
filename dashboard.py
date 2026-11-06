"""
dashboard.py — the tokentrack web dashboard (Streamlit).

    pip install streamlit plotly
    streamlit run dashboard.py                 # your real data
    streamlit run dashboard.py -- --db demo.db # the demo month

A browser tab opens at http://localhost:8501. Ctrl+C in the terminal stops it.
"""
import os
import sqlite3
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coach import RULES, Finding  # noqa: E402

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="tokentrack", page_icon="▣", layout="wide")

# "--db demo.db" after "--" on the command line
DB = os.environ.get("TOKENTRACK_DB", "tokentrack.db")
if "--db" in sys.argv:
    DB = sys.argv[sys.argv.index("--db") + 1]

COLORS = px.colors.qualitative.Set2


@st.cache_data(ttl=30)
def load(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    df = pd.read_sql("SELECT * FROM events ORDER BY ts", sqlite3.connect(db_path))
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df.ts, utc=True)
    df["day"] = df.ts.dt.floor("D")
    df["tag"] = df.tag.fillna("(untagged)")
    df["total_tokens"] = df.input_tokens + df.output_tokens
    return df


def money(x: float) -> str:
    """Pick precision based on scale so tiny bills don't all read $0.00."""
    if x >= 100:
        return f"${x:,.0f}"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.4f}"


# ----------------------------------------------------------------------------
# Sidebar: filters
# ----------------------------------------------------------------------------
df_all = load(DB)

st.sidebar.title("tokentrack")
st.sidebar.caption(f"reading `{DB}`")

if df_all.empty:
    st.title("No data yet")
    st.write(f"`{DB}` is empty or missing. Run an example script, or try the demo:")
    st.code("python coach.py --demo\nstreamlit run dashboard.py -- --db demo.db")
    st.stop()

min_day, max_day = df_all.day.min().date(), df_all.day.max().date()
date_range = st.sidebar.date_input("Date range", (min_day, max_day), min_value=min_day, max_value=max_day)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
else:
    start, end = pd.Timestamp(min_day, tz="UTC"), pd.Timestamp(max_day, tz="UTC") + pd.Timedelta(days=1)

models = sorted(df_all.model.unique())
tags = sorted(df_all.tag.unique())
sel_models = st.sidebar.multiselect("Models", models, default=models)
sel_tags = st.sidebar.multiselect("Tags", tags, default=tags)

df = df_all[(df_all.ts >= start) & (df_all.ts < end) & df_all.model.isin(sel_models) & df_all.tag.isin(sel_tags)]
if df.empty:
    st.warning("Nothing matches these filters.")
    st.stop()

# ----------------------------------------------------------------------------
# Header + KPI tiles
# ----------------------------------------------------------------------------
st.title("Usage & waste")

total_cost = df.cost_usd.sum()
findings: list[Finding] = [f for rule in RULES for f in rule(df)]
saveable = sum(f.est_savings_usd for f in findings)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Spend", money(total_cost))
k2.metric("Calls", f"{len(df):,}")
k3.metric("Input tokens", f"{df.input_tokens.sum():,}")
k4.metric("Output tokens", f"{df.output_tokens.sum():,}")
k5.metric("Avoidable (est.)", money(saveable), f"{saveable / total_cost:.0%} of spend" if total_cost else None, delta_color="inverse")

tab_overview, tab_convs, tab_waste, tab_raw = st.tabs(["Overview", "Conversations", "Waste report", "Raw events"])

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns([2, 1])

    daily = df.groupby(["day", "model"], as_index=False).cost_usd.sum()
    fig = px.bar(daily, x="day", y="cost_usd", color="model", color_discrete_sequence=COLORS,
                 title="Spend per day", labels={"cost_usd": "USD", "day": ""})
    fig.update_layout(legend_title_text="", margin=dict(t=40, b=0))
    c1.plotly_chart(fig, use_container_width=True)

    by_model = df.groupby("model", as_index=False).cost_usd.sum().sort_values("cost_usd")
    fig = px.bar(by_model, x="cost_usd", y="model", orientation="h", color_discrete_sequence=COLORS,
                 title="Spend by model", labels={"cost_usd": "USD", "model": ""})
    fig.update_layout(margin=dict(t=40, b=0))
    c2.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    by_tag = df.groupby("tag", as_index=False).agg(cost_usd=("cost_usd", "sum"), calls=("id", "count")).sort_values("cost_usd")
    fig = px.bar(by_tag, x="cost_usd", y="tag", orientation="h", color_discrete_sequence=COLORS,
                 title="Spend by tag (which part of your app)", labels={"cost_usd": "USD", "tag": ""})
    fig.update_layout(margin=dict(t=40, b=0))
    c3.plotly_chart(fig, use_container_width=True)

    io = df.groupby("model", as_index=False).agg(input=("input_tokens", "sum"), output=("output_tokens", "sum"))
    fig = go.Figure()
    fig.add_bar(name="input", x=io.model, y=io.input, marker_color=COLORS[0])
    fig.add_bar(name="output", x=io.model, y=io.output, marker_color=COLORS[1])
    fig.update_layout(barmode="stack", title="Tokens in vs out, by model", margin=dict(t=40, b=0), legend_title_text="")
    c4.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Conversations
# ----------------------------------------------------------------------------
with tab_convs:
    convs = df[df.conversation_id.notna()].copy()
    if convs.empty:
        st.info("No calls carry a conversation_id yet. Pass one to group multi-turn chats.")
    else:
        convs["turn"] = convs.groupby("conversation_id").cumcount() + 1
        summary = (convs.groupby("conversation_id")
                   .agg(turns=("id", "count"), model=("model", "first"), tag=("tag", "first"),
                        first_input=("input_tokens", "first"), last_input=("input_tokens", "last"),
                        cost_usd=("cost_usd", "sum"), started=("ts", "min"))
                   .sort_values("cost_usd", ascending=False))
        summary["growth"] = (summary.last_input / summary.first_input.clip(lower=1)).round(1)

        st.subheader("Most expensive conversations")
        st.dataframe(summary.style.format({"cost_usd": money}), use_container_width=True, height=300)

        pick = st.selectbox("Inspect a conversation", summary.index.tolist())
        one = convs[convs.conversation_id == pick]
        fig = go.Figure()
        fig.add_bar(name="input tokens", x=one.turn, y=one.input_tokens, marker_color=COLORS[0])
        fig.add_bar(name="output tokens", x=one.turn, y=one.output_tokens, marker_color=COLORS[1])
        fig.update_layout(barmode="stack", title=f"Tokens per turn — {pick}", xaxis_title="turn",
                          margin=dict(t=40, b=0), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Input climbing every turn means the whole history is being re-sent. That's context bloat.")

# ----------------------------------------------------------------------------
# Waste report
# ----------------------------------------------------------------------------
with tab_waste:
    if not findings:
        st.success("Nothing to flag in this range. Either you're efficient or there isn't enough data yet.")
    else:
        order = {"high": 0, "medium": 1, "low": 2}
        findings.sort(key=lambda f: (order[f.severity], -f.est_savings_usd))
        st.caption("Dollar figures are estimates built on stated assumptions — see each item.")
        for f in findings:
            icon = {"high": "🔴", "medium": "🟠", "low": "⚪"}[f.severity]
            label = f"{icon} {f.title}" + (f"  —  ~{money(f.est_savings_usd)}" if f.est_savings_usd else "")
            with st.expander(label, expanded=(f.severity == "high")):
                st.markdown(f.detail.replace("$", r"\$"))  # stop markdown treating $...$ as maths
                if not f.evidence.empty:
                    st.dataframe(f.evidence, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# Raw events
# ----------------------------------------------------------------------------
with tab_raw:
    cols = ["ts", "provider", "model", "tag", "conversation_id", "input_tokens", "output_tokens",
            "cached_tokens", "cost_usd", "latency_ms"]
    st.dataframe(df[cols].sort_values("ts", ascending=False), use_container_width=True, height=500)
    st.download_button("Download as CSV", df[cols].to_csv(index=False).encode(), "tokentrack_events.csv", "text/csv")
