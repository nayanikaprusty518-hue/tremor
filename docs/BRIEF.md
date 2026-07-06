# Tremor — brief description (for the submission form)

## One-liner
**Tremor** is a community meltdown early-warning radar: it scores an online
community's comment stream, ranks which threads are about to melt down, and warns
a moderator *before* the blow-up.

## Short (≈100 words)
Volunteer moderators of subreddits, Discord servers and group chats have no Trust
& Safety team — they find out about pile-ons and toxic spirals only after the
damage is done. Tremor ingests a comment stream, scores every comment for
hostility, and rolls it up per thread and time window to produce a 0–100 meltdown
risk, a ranked watchlist, a per-community health score, and — the key output — an
**early-warning forecast that fires while a thread is still climbing, ~60 minutes
before it peaks.** The heavy scoring stage runs on the identical code on CPU or an
NVIDIA GPU via `cudf.pandas`; on 5M comments the GPU is **33× faster** (97.6s → 2.95s on a
Colab T4), which is what makes the warning arrive in time to act.

## What was built
- A reproducible synthetic comment-stream generator (scales to millions of rows).
- A scoring + risk + early-warning engine using the pandas API (so it GPU-accelerates unchanged with `cudf.pandas`).
- A CPU-vs-GPU benchmark + a Colab GPU notebook.
- An interactive Streamlit dashboard: community health, ranked watchlist, and a thread "replay" that shows the warning firing before the meltdown peak.

## Stack (2+ layers)
Google Cloud (Cloud Storage / BigQuery) for the data layer + NVIDIA (cuDF /
`cudf.pandas` on a GPU) for the acceleration layer; deployed as a Streamlit app.

## Measured results
100% precision and 100% recall against planted ground-truth meltdowns; 66 of 68
meltdowns forecast before they peaked; **33× GPU speedup** on the heavy stage (5M comments: 97.6s → 2.95s).

## Links
- **Deployment:** _[paste your public URL]_
- **GitHub:** _[paste your repo URL]_
- **Demo video:** _[paste your ≤3-min video URL]_
