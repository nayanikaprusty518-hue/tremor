"""
Tremor — community meltdown early-warning radar (Streamlit dashboard).

Run locally:   streamlit run app.py
Data:          reads comments.parquet (set TREMOR_DATA to override).
Acceleration:  if benchmark_results.json is present it shows the cuDF-vs-pandas
               speedup banner (produced by the Colab GPU notebook).
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import tremor_core as tc

DATA_PATH = os.environ.get("TREMOR_DATA", "comments.parquet")
BENCH_PATH = os.environ.get("TREMOR_BENCH", "benchmark_results.json")

STATUS_COLORS = {"CRITICAL": "#e5484d", "WATCH": "#f5a623", "CALM": "#30a46c"}
COMMUNITY_NAMES = [
    "r/gaming", "r/politics", "r/technology", "discord: crypto-hub", "r/sports",
    "r/movies", "discord: study-group", "r/parenting", "r/localcity",
    "r/investing", "discord: fandom", "r/science",
]

st.set_page_config(page_title="Tremor — meltdown early-warning",
                   page_icon="📡", layout="wide")


def cname(i):
    return COMMUNITY_NAMES[int(i) % len(COMMUNITY_NAMES)]


@st.cache_data(show_spinner=False)
def load_comments(path):
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Scoring comments and computing risk…")
def run_pipeline(path, freq, roll):
    df = load_comments(path).copy()
    risk, wl, health = tc.full_pipeline(df, freq=freq, roll=roll)
    meta = {
        "rows": int(len(df)),
        "communities": int(df["community"].nunique()),
        "threads": int(df["thread_id"].nunique()),
        "start": pd.to_datetime(df["timestamp"].min()),
        "end": pd.to_datetime(df["timestamp"].max()),
    }
    return risk, wl, health, meta


def load_bench():
    if os.path.exists(BENCH_PATH):
        try:
            return json.load(open(BENCH_PATH))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------- header
st.markdown(
    """
    <div style="padding:18px 22px;border-radius:14px;
         background:linear-gradient(100deg,#1f6feb22,#e5484d22);border:1px solid #8884;">
      <span style="font-size:30px;font-weight:800;">📡 Tremor</span>
      <span style="font-size:17px;opacity:.8;"> &nbsp;— community meltdown early-warning radar</span>
      <div style="opacity:.75;margin-top:4px;">
        Catches the tremors before the quake: ranks which threads are about to melt down,
        so a volunteer moderator knows <b>where</b> and <b>when</b> to step in — before the damage.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.path.exists(DATA_PATH):
    # self-heal on a fresh deploy: generate a demo dataset once
    with st.spinner("First run — generating demo dataset (≈200k comments)…"):
        import generate_data
        _df0 = generate_data.generate(rows=200_000, n_communities=12, n_threads=350,
                                      n_authors=8_000, days=14, meltdown_frac=0.16, seed=7)
        _df0.to_parquet(DATA_PATH, index=False)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    freq = st.select_slider("Time-bucket size",
                            options=["10min", "15min", "30min", "1h", "2h"], value="30min")
    roll = st.slider("Baseline window (buckets)", 3, 12, 6)
    st.divider()

    st.subheader("⚡ NVIDIA acceleration")
    bench = load_bench()
    if bench and "cpu" in bench and "gpu" in bench:
        sp = bench.get("speedup", "?")
        st.metric("cuDF vs pandas", f"{sp}× faster",
                  help="Scoring + aggregating the full comment set")
        st.caption(
            f"{bench['cpu']['rows']:,} comments · "
            f"CPU {bench['cpu']['process_s']}s → GPU {bench['gpu']['process_s']}s"
        )
    else:
        st.info("Run `notebooks/tremor_gpu_benchmark.ipynb` on a Colab GPU, "
                "commit `benchmark_results.json`, and the speedup shows here.")
    st.caption("Same code both ways — `cudf.pandas` is a drop-in.")

risk, wl, health, meta = run_pipeline(DATA_PATH, freq, roll)

# ---------------------------------------------------------------- KPI row
n_crit = int((wl["status"] == "CRITICAL").sum())
n_watch = int((wl["status"] == "WATCH").sum())
n_fore = int(wl["was_forecast"].sum())
k = st.columns(5)
k[0].metric("Comments analysed", f"{meta['rows']:,}")
k[1].metric("Communities", meta["communities"])
k[2].metric("Threads monitored", f"{meta['threads']:,}")
k[3].metric("🔴 Critical now", n_crit)
k[4].metric("⚠️ Early warnings", n_fore)

st.divider()

# ---------------------------------------------------------------- community health
left, right = st.columns([1.05, 1])
with left:
    st.subheader("Community health")
    hh = health.copy()
    hh["name"] = hh["community"].map(cname)
    hh = hh.sort_values("health")
    fig = go.Figure(go.Bar(
        x=hh["health"], y=hh["name"], orientation="h",
        marker=dict(color=hh["health"], colorscale="RdYlGn", cmin=0, cmax=100),
        text=[f"{v:.0f}" for v in hh["health"]], textposition="outside",
        hovertemplate="%{y}<br>health %{x:.0f}/100<extra></extra>",
    ))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="health score (higher = calmer)",
                      xaxis_range=[0, 108], plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Where to look first")
    worst = health.sort_values("health").iloc[0]
    st.markdown(
        f"**{cname(worst['community'])}** needs attention most — "
        f"health **{worst['health']:.0f}/100**, "
        f"**{int(worst['critical'])} critical** threads, "
        f"**{int(worst['forecasts'])}** early warnings issued."
    )
    st.dataframe(
        health.assign(community=health["community"].map(cname))
              .rename(columns={"community": "Community", "threads": "Threads",
                               "critical": "🔴", "watch": "⚠️",
                               "forecasts": "Forecasts", "health": "Health"})
              [["Community", "Threads", "🔴", "⚠️", "Forecasts", "Health"]]
              .style.format({"Health": "{:.0f}"})
              .background_gradient(subset=["Health"], cmap="RdYlGn", vmin=0, vmax=100),
        width="stretch", hide_index=True, height=300,
    )

st.divider()

# ---------------------------------------------------------------- watchlist
st.subheader("🚨 Live watchlist — threads ranked by meltdown risk")
fc1, fc2 = st.columns([1, 3])
with fc1:
    status_filter = st.multiselect("Show", ["CRITICAL", "WATCH", "CALM"],
                                   default=["CRITICAL", "WATCH"])
view = wl[wl["status"].isin(status_filter)] if status_filter else wl
disp = view.copy()
disp["Community"] = disp["community"].map(cname)
disp["Thread"] = "#" + disp["thread_id"].astype(str)
disp["Forecast"] = np.where(disp["was_forecast"], "⚠️ early warning", "—")
disp = disp.rename(columns={"peak_risk": "Peak risk", "risk_now": "Now",
                            "total_comments": "Comments", "buckets": "Buckets",
                            "status": "Status"})
disp = disp[["Community", "Thread", "Status", "Peak risk", "Now",
             "Forecast", "Comments", "Buckets"]]


def _status_style(v):
    return (f"color:white;background-color:{STATUS_COLORS.get(v, '#888')};"
            "font-weight:600;text-align:center;")


st.dataframe(
    disp.style
        .map(_status_style, subset=["Status"])
        .format({"Peak risk": "{:.0f}", "Now": "{:.0f}", "Comments": "{:,}"})
        .background_gradient(subset=["Peak risk"], cmap="OrRd", vmin=0, vmax=100),
    width="stretch", hide_index=True, height=340,
)

st.divider()

# ---------------------------------------------------------------- drill-down / replay
st.subheader("🔎 Thread replay — watch a meltdown build (and see it forecast early)")

# default to the riskiest thread that had an early warning, else the riskiest
cand = wl[wl["was_forecast"]] if wl["was_forecast"].any() else wl
options = cand.head(40)
labels = [f"{cname(r.community)} · thread #{int(r.thread_id)} · peak {r.peak_risk:.0f}"
          f"{'  ⚠️ forecast' if r.was_forecast else ''}"
          for r in options.itertuples()]
choice = st.selectbox("Pick a thread", range(len(labels)),
                      format_func=lambda i: labels[i])
sel = options.iloc[choice]
comm, tid = int(sel["community"]), int(sel["thread_id"])

ts = risk[(risk["community"] == comm) & (risk["thread_id"] == tid)].sort_values("bucket")
ts = ts.reset_index(drop=True)

# replay cursor
cur = st.slider("Replay up to", 0, len(ts) - 1, len(ts) - 1,
                help="Drag left to rewind time and watch the risk build.")
shown = ts.iloc[: cur + 1]
now_row = shown.iloc[-1]

# lead-time stat: first forecast vs peak
peak_idx = ts["risk_smooth"].idxmax()
peak_time = ts.loc[peak_idx, "bucket"]
fore_rows = ts[ts["predicted_meltdown"]]
lead_txt = "—"
if len(fore_rows):
    first_fore = fore_rows.iloc[0]["bucket"]
    lead = pd.to_datetime(peak_time) - pd.to_datetime(first_fore)
    mins = max(int(lead.total_seconds() // 60), 0)
    lead_txt = f"{mins} min early" if mins < 120 else f"{mins/60:.1f} h early"

m = st.columns(4)
cur_status = tc._status(now_row["risk_smooth"])
m[0].markdown(f"### :{'red' if cur_status=='CRITICAL' else 'orange' if cur_status=='WATCH' else 'green'}[{cur_status}]")
m[1].metric("Risk (at cursor)", f"{now_row['risk_smooth']:.0f}/100")
m[2].metric("Comments / bucket", int(now_row["n_comments"]))
m[3].metric("Forecast lead time", lead_txt)

if len(fore_rows) and cur >= ts.index.get_loc(fore_rows.index[0]):
    st.warning(f"⚠️ **Tremor early-warning fired** at "
               f"{pd.to_datetime(fore_rows.iloc[0]['bucket']):%b %d %H:%M} — "
               f"toxicity and volume rising fast. Step in now, before it peaks.")

fig2 = go.Figure()
fig2.add_bar(x=shown["bucket"], y=shown["n_comments"], name="comments / bucket",
             marker_color="#8ab4f8", opacity=0.55, yaxis="y2")
fig2.add_trace(go.Scatter(x=shown["bucket"], y=shown["tox_mean"] * 100,
                          name="toxicity %", mode="lines",
                          line=dict(color="#f5a623", width=2)))
fig2.add_trace(go.Scatter(x=shown["bucket"], y=shown["risk_smooth"],
                          name="meltdown risk", mode="lines",
                          line=dict(color="#e5484d", width=3)))
# markers for forecast buckets within the shown window
fw = shown[shown["predicted_meltdown"]]
if len(fw):
    fig2.add_trace(go.Scatter(x=fw["bucket"], y=fw["risk_smooth"], mode="markers",
                              name="⚠️ early warning",
                              marker=dict(color="#e5484d", size=11, symbol="triangle-up",
                                          line=dict(color="white", width=1))))
if pd.to_datetime(peak_time) <= pd.to_datetime(now_row["bucket"]):
    fig2.add_vline(x=peak_time, line_dash="dot", line_color="#888",
                   annotation_text="peak", annotation_position="top")
fig2.update_layout(
    height=380, margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(title="risk / toxicity", range=[0, 105]),
    yaxis2=dict(title="comments", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", y=1.12), plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig2, width="stretch")

st.caption(
    "Toxicity + volume climb together; the ⚠️ marker is where Tremor forecast the "
    "meltdown — ahead of the dotted **peak** line. That lead time is the moderator's "
    "window to intervene. Scoring every comment fast enough to give that warning is "
    "what the NVIDIA GPU acceleration buys."
)
