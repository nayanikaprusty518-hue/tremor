"""
Tremor core — community meltdown early-warning engine.

Design note (why this is the heart of the acceleration story):
The heavy stage — `score_and_aggregate` — runs a vectorized regex string op and a
groupby over EVERY comment (millions of rows). That is exactly the workload where
NVIDIA cuDF crushes pandas. Because we use ONLY the pandas API here, the identical
code runs:
    * on CPU  ->  plain `python`          (imports the real pandas)
    * on GPU  ->  `python -m cudf.pandas` (pandas calls dispatch to cuDF on the GPU)
No separate code paths. That is the whole point of cudf.pandas.

The downstream stage — `compute_risk` / `thread_watchlist` — runs on the small
aggregated table (threads x time-buckets), so its cost is negligible either way.
"""

import re
import numpy as np
import pandas as pd  # under `python -m cudf.pandas ...` this transparently becomes cuDF

# ---------------------------------------------------------------------------
# Escalation lexicon — genuinely hostile markers only.
# Deliberately PG / synthetic (heat markers, not slurs). We intentionally EXCLUDE
# common conversational words (lol, actually, clearly, ...) because they are not
# hostility signals and would pollute the score.
# ---------------------------------------------------------------------------
LEXICON = {
    "medium": ["idiot", "stupid", "dumb", "trash", "garbage", "clown", "pathetic",
               "loser", "losers", "delusional", "shameful", "embarrassing",
               "toxic", "clueless", "nonsense", "ridiculous"],
    "hot":    ["shut up", "hate", "worst", "disgusting", "insufferable", "get lost",
               "grow up", "coping", "cope", "seethe", "rage", "reported", "brigade",
               "brigading", "gtfo", "cringe", "troll"],
}
TOX_WORDS = [w for tier in LEXICON.values() for w in tier]
# longest phrases first so multi-word markers ("shut up") match before parts
_ESCAPED = sorted((re.escape(w) for w in TOX_WORDS), key=len, reverse=True)
TOX_PATTERN = r"\b(?:" + "|".join(_ESCAPED) + r")\b"

# tuning knobs
HITS_NORMALIZER = 3.0   # a comment with >=3 hostile markers is treated as fully toxic
VOL_K = 6.0             # volume-confidence: a bucket needs volume to be trusted
STATUS_WATCH = 28      # above the calm-thread ceiling (~26) -> no false alarms
STATUS_CRITICAL = 50   # ~median meltdown -> the worst threads go red


# ---------------------------------------------------------------------------
# STAGE 1 — the heavy, GPU-accelerated stage (scales with number of comments)
# ---------------------------------------------------------------------------
def score_and_aggregate(df, freq="30min"):
    """Score every comment for toxicity and roll comments up into
    (community, thread, time-bucket) buckets. This is the benchmarked stage."""
    # vectorized regex count over the WHOLE comment column  <-- cuDF sweet spot
    df["tox_hits"] = df["text"].str.count(TOX_PATTERN).fillna(0)
    df["tox"] = (df["tox_hits"] / HITS_NORMALIZER).clip(upper=1.0)
    # integer-arithmetic time bucketing — GPU-native, avoids a .dt.floor CPU fallback.
    # Floors each timestamp to the freq grid measured from the epoch (identical to
    # .dt.floor for fixed frequencies like 10min/30min/1h). Requires ns-resolution.
    bucket_ns = pd.Timedelta(freq).value
    df["bucket"] = ((df["timestamp"].astype("int64") // bucket_ns) * bucket_ns).astype("datetime64[ns]")

    agg = (
        df.groupby(["community", "thread_id", "bucket"])
        .agg({"comment_id": "count", "tox": "mean",
              "tox_hits": "sum", "author_id": "nunique"})
        .reset_index()
        .rename(columns={"comment_id": "n_comments", "tox": "tox_mean",
                         "author_id": "n_authors"})
    )
    return agg


# ---------------------------------------------------------------------------
# STAGE 2 — the light stage: turn bucket aggregates into a risk + early warning
# (runs on the small aggregated table, so engine choice barely matters here)
# ---------------------------------------------------------------------------
def _roll_mean(g, col, roll):
    return g[col].transform(lambda s: s.rolling(roll, min_periods=1).mean())


def compute_risk(agg, roll=6):
    """Per-thread rolling baselines -> anomaly -> 0..100 meltdown risk + forecast."""
    agg = agg.sort_values(["community", "thread_id", "bucket"]).reset_index(drop=True)
    g = agg.groupby(["community", "thread_id"])

    agg["tox_base"] = _roll_mean(g, "tox_mean", roll)
    agg["vol_base"] = _roll_mean(g, "n_comments", roll)
    agg["tox_vel"] = g["tox_mean"].diff().fillna(0)                    # toxicity acceleration
    agg["vol_ratio"] = agg["n_comments"] / (agg["vol_base"] + 1e-6)    # volume surge

    # volume-confidence: a lone angry comment must NOT read as a meltdown
    conf = agg["n_comments"] / (agg["n_comments"] + VOL_K)
    agg["level"] = (agg["tox_mean"] * conf).clip(0, 1)                 # trustworthy toxicity

    risk = (
        60.0 * agg["level"]
        + 22.0 * (agg["tox_vel"].clip(lower=0) * 6.0).clip(0, 1)       # rising fast
        + 18.0 * ((agg["vol_ratio"] - 1.0) / 3.0).clip(0, 1)          # volume surge
    )
    agg["risk"] = risk.clip(0, 100)

    # smooth per thread so one noisy bucket does not dominate the ranking
    agg["risk_smooth"] = (
        agg.groupby(["community", "thread_id"])["risk"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # EARLY WARNING: toxicity + volume rising fast, enough evidence, level not yet
    # extreme => "this thread is about to melt down" BEFORE it peaks.
    agg["predicted_meltdown"] = (
        (agg["tox_vel"] > 0.03)
        & (agg["vol_ratio"] > 1.5)
        & (agg["n_comments"] >= 5)
        & (agg["level"] < 0.50)
        & (agg["risk"] > 30)
    )
    return agg


def _status(r):
    if r >= STATUS_CRITICAL:
        return "CRITICAL"
    if r >= STATUS_WATCH:
        return "WATCH"
    return "CALM"


def thread_watchlist(risk):
    """Collapse the time-series into one ranked row per thread (ranked by the
    thread's WORST smoothed moment — that is what a meltdown radar cares about)."""
    latest = (
        risk.sort_values("bucket")
        .drop_duplicates(["community", "thread_id"], keep="last")
        .loc[:, ["community", "thread_id", "bucket", "risk_smooth", "tox_mean", "n_comments"]]
        .rename(columns={"risk_smooth": "risk_now", "tox_mean": "tox_now"})
    )
    stats = (
        risk.groupby(["community", "thread_id"])
        .agg(peak_risk=("risk_smooth", "max"),
             was_forecast=("predicted_meltdown", "max"),
             total_comments=("n_comments", "sum"),
             buckets=("bucket", "count"))
        .reset_index()
    )
    wl = stats.merge(latest, on=["community", "thread_id"])
    wl["status"] = wl["peak_risk"].map(_status)
    return wl.sort_values("peak_risk", ascending=False).reset_index(drop=True)


def community_health(wl):
    """One health row per community for the top-level scorecards."""
    h = (
        wl.groupby("community")
        .agg(threads=("thread_id", "count"),
             avg_peak=("peak_risk", "mean"),
             max_peak=("peak_risk", "max"),
             critical=("status", lambda s: (s == "CRITICAL").sum()),
             watch=("status", lambda s: (s == "WATCH").sum()),
             forecasts=("was_forecast", "sum"))
        .reset_index()
    )
    # health = inverse of how bad threads get, penalised by each critical thread
    h["health"] = (100 - 0.7 * h["avg_peak"] - 5 * h["critical"]).clip(0, 100)
    return h.sort_values("health").reset_index(drop=True)


def full_pipeline(df, freq="30min", roll=6):
    """Convenience: raw comments -> (risk timeseries, watchlist, community health)."""
    agg = score_and_aggregate(df, freq=freq)
    risk = compute_risk(agg, roll=roll)
    wl = thread_watchlist(risk)
    health = community_health(wl)
    return risk, wl, health
