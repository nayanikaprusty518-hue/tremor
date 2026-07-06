"""
Deterministic synthetic comment-stream generator for Tremor.

Why synthetic: it removes any data-sourcing / API / scraping dependency, it is
fully reproducible (fixed seed), and it lets us dial the row count up to millions
so the cuDF-vs-pandas benchmark is honest and dramatic. The generator plants a
realistic signal: a fraction of threads have a "meltdown" window where both the
comment *volume* and the *toxicity* rise — first gradually (the early-warning
signal Tremor detects), then peak.

Usage:
    python generate_data.py --rows 300000                 # quick local/demo file
    python generate_data.py --rows 5000000 --out big.parquet   # benchmark file
"""

import argparse
import numpy as np
import pandas as pd

# --- text pools ------------------------------------------------------------
# Neutral comments: normal community chatter (occasional mild word only).
NEUTRAL = [
    "thanks for sharing this, really helpful",
    "does anyone know how to fix this setup",
    "i tried that yesterday and it worked fine",
    "great point, i had not thought about it that way",
    "here is the link to the docs i mentioned",
    "can we move this to the pinned thread",
    "honestly not sure, let me check and get back",
    "that update improved things a lot for me",
    "appreciate the detailed write up here",
    "i think both approaches are reasonable actually",
    "welcome to the community, glad you joined",
    "the event this weekend looks fun, who is going",
    "posting my results below for anyone curious",
    "makes sense, i will give it a try tonight",
    "good catch, i missed that in the guide",
    "seriously though this feature is underrated lol",
    "clearly a lot of effort went into this, nice",
    "obviously depends on your use case i guess",
    "whatever works for you is fine by me",
    "actually the newer version handles this better",
]

# Toxic comments: escalation. Multiple markers per line raise the hit-density.
TOXIC = [
    "this is the most stupid take i have ever seen honestly",
    "shut up you clueless clown, nobody asked",
    "what a pathetic garbage post, delete this trash",
    "you are an idiot and clearly delusional, cope harder",
    "worst community ever, everyone here is insufferable",
    "hate how dumb this thread is getting, embarrassing",
    "get lost troll, this is disgusting behaviour",
    "ridiculous nonsense, you are all coping and raging",
    "reported you, absolute clown behaviour, grow up",
    "so toxic and dumb, this whole thread is cringe garbage",
    "stupid losers brigading again, gtfo of here",
    "clearly the worst, most pathetic idiots online",
    "shameful and disgusting, hate every one of you",
    "what a joke, delusional trash take as always",
    "cope and seethe, you clueless embarrassing loser",
]


def generate(rows, n_communities, n_threads, n_authors, days, meltdown_frac, seed):
    rng = np.random.default_rng(seed)
    base_ts = np.datetime64("2026-07-01T00:00:00")

    # --- thread-level attributes ------------------------------------------
    thr_comm = rng.integers(0, n_communities, n_threads)
    thr_start = rng.uniform(0, days * 86400.0, n_threads)               # seconds from base
    thr_dur = rng.uniform(3 * 3600.0, 2 * 86400.0, n_threads)           # 3h .. 2d lifetimes
    thr_meltdown = rng.random(n_threads) < meltdown_frac
    thr_m_start = rng.uniform(0.30, 0.60, n_threads)                    # window start (frac of life)
    thr_m_len = rng.uniform(0.12, 0.28, n_threads)                      # window length (frac)

    melt_threads = np.flatnonzero(thr_meltdown)
    if melt_threads.size == 0:                                          # guarantee >=1
        melt_threads = np.array([0])
        thr_meltdown[0] = True

    # --- split rows: baseline chatter + a concentrated burst in meltdowns --
    n_burst = int(rows * 0.28)
    n_base = rows - n_burst

    # baseline comments: uniform across threads and across each thread's life
    b_thread = rng.integers(0, n_threads, n_base)
    b_tfrac = rng.random(n_base)

    # burst comments: only in meltdown threads, only inside their meltdown window
    u_thread = melt_threads[rng.integers(0, melt_threads.size, n_burst)]
    u_ms = thr_m_start[u_thread]
    u_ml = thr_m_len[u_thread]
    # cluster toward the LATER part of the window so risk builds then peaks
    u_tfrac = u_ms + (rng.random(n_burst) ** 0.7) * u_ml

    c_thread = np.concatenate([b_thread, u_thread])
    c_tfrac = np.concatenate([b_tfrac, u_tfrac])

    # --- timestamps --------------------------------------------------------
    ts_sec = thr_start[c_thread] + c_tfrac * thr_dur[c_thread]
    timestamp = base_ts + (ts_sec * 1000.0).astype("timedelta64[ms]")

    # --- is this comment inside its thread's meltdown window? --------------
    ms = thr_m_start[c_thread]
    ml = thr_m_len[c_thread]
    in_window = thr_meltdown[c_thread] & (c_tfrac >= ms) & (c_tfrac <= ms + ml)
    # ramp: toxicity probability rises across the window (early-warning signal)
    pos_in_window = np.clip((c_tfrac - ms) / np.maximum(ml, 1e-6), 0, 1)
    p_tox = np.where(in_window, 0.25 + 0.65 * pos_in_window, 0.04)
    is_tox = rng.random(c_thread.size) < p_tox

    # --- text (fancy-index into small pools => millions of refs, low memory)
    neutral_arr = np.array(NEUTRAL, dtype=object)
    toxic_arr = np.array(TOXIC, dtype=object)
    neutral_pick = neutral_arr[rng.integers(0, neutral_arr.size, c_thread.size)]
    toxic_pick = toxic_arr[rng.integers(0, toxic_arr.size, c_thread.size)]
    text = np.where(is_tox, toxic_pick, neutral_pick)

    df = pd.DataFrame(
        {
            "comment_id": np.arange(c_thread.size, dtype=np.int64),
            "community": thr_comm[c_thread].astype(np.int32),
            "thread_id": c_thread.astype(np.int32),
            "timestamp": timestamp,
            "author_id": rng.integers(0, n_authors, c_thread.size).astype(np.int32),
            "text": text,
            "is_meltdown_thread": thr_meltdown[c_thread],  # ground truth for eval
        }
    )
    # shuffle so rows are not grouped by thread (realistic stream order)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["comment_id"] = np.arange(len(df), dtype=np.int64)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300_000)
    ap.add_argument("--communities", type=int, default=12)
    ap.add_argument("--threads", type=int, default=None,
                    help="default ~ rows/120")
    ap.add_argument("--authors", type=int, default=None,
                    help="default ~ rows/25")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--meltdown-frac", type=float, default=0.16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="comments.parquet")
    args = ap.parse_args()

    n_threads = args.threads or max(50, args.rows // 600)
    n_authors = args.authors or max(200, args.rows // 25)

    df = generate(
        rows=args.rows,
        n_communities=args.communities,
        n_threads=n_threads,
        n_authors=n_authors,
        days=args.days,
        meltdown_frac=args.meltdown_frac,
        seed=args.seed,
    )
    df.to_parquet(args.out, index=False)
    print(
        f"wrote {len(df):,} comments -> {args.out}  "
        f"({args.communities} communities, {n_threads:,} threads, "
        f"{df['is_meltdown_thread'].mean()*100:.1f}% meltdown-thread rows)"
    )


if __name__ == "__main__":
    main()
