# 📡 Tremor — community meltdown early-warning radar

**Catches the tremors before the quake.** Tremor scores a community's comment
stream in real time, ranks which threads are **about to melt down**, and warns a
moderator *before* the blow-up — turning "we found out after everyone left" into
"we stepped in with minutes to spare."

Built for the **long tail of volunteer moderators** — subreddit mods, Discord/
Telegram/WhatsApp group admins, small-brand community managers — who have **no
Trust & Safety team** and no enterprise tooling.

---

## The problem (real, and documented)

Volunteer moderators burn out from daily exposure to toxic behaviour and from
finding out about pile-ons *after* the damage is done ([U-Michigan study](https://news.umich.edu/online-content-moderators-likely-to-experience-burnout-u-m-study-suggests/)).
Enterprise platforms get AI moderation; the people running a 40k-member subreddit
by hand get nothing. Tremor is that missing radar.

## The decision it powers

> **"Which thread do I open right now, and is it about to get worse?"**

Tremor answers with a ranked watchlist, a per-community health score, and — the
part that matters — an **early-warning flag that fires while a thread is still
climbing, before it peaks.**

---

## How it works (the pipeline)

```
raw comments ─▶ [1] score toxicity      (vectorized lexicon matching, EVERY comment)◀── GPU-accelerated
             ─▶     bucket by time        (community × thread × 30-min window)   ◀── GPU-accelerated
             ─▶     aggregate             (volume, mean toxicity, unique authors)◀── GPU-accelerated
             ─▶ [2] rolling baselines      per thread
             ─▶     anomaly + velocity     (toxicity rising? volume surging?)
             ─▶     meltdown risk 0–100    (volume-confidence gated)
             ─▶     EARLY-WARNING forecast (rising fast + not-yet-extreme)
             ─▶ ranked watchlist · community health · thread replay
```

**Stage 1 is the heavy stage** — it touches every one of millions of comments, so
it is where GPU acceleration pays off. **Stage 2** runs on the small aggregated
table (threads × time-buckets), so it is cheap on any engine.

### Validated quality (on the synthetic benchmark set)
Ground-truth meltdown threads are planted in the data, so we can measure:

| metric | result |
|---|---|
| Precision of flagged threads | **100%** (no false alarms) |
| Recall of true meltdowns | **100%** (68/68 caught) |
| Meltdowns forecast **before** they peaked | **66 / 68** |
| Typical early-warning lead time | **~60 min** |

---

## ⚡ The acceleration story (Google Cloud + NVIDIA)

The core stage uses **only the pandas API**, so the *identical code* runs on two
engines with **zero changes**:

```bash
# CPU (pandas)
python benchmark.py --data big.parquet

# NVIDIA GPU (cuDF) — same file, one prefix
python -m cudf.pandas benchmark.py --data big.parquet
```

That is the whole point of **`cudf.pandas`**: a drop-in that dispatches pandas
calls to the GPU. Run `notebooks/tremor_gpu_benchmark.ipynb` on a Colab/GCP GPU
to reproduce the CPU-vs-GPU numbers; the dashboard then shows the speedup live.

> **Measured (Colab T4, 5,000,000 comments):** heavy stage CPU **97.6s** → GPU
> **2.95s** = **33× faster**. Profiler-confirmed: scoring, bucketing and the
> groupby all run on the GPU.

**Why it matters operationally:** to warn *early* you must score comments faster
than they arrive. On CPU a busy day's backlog takes too long — you learn about the
fire afterwards. On a GPU it is seconds → the warning arrives in time to act. That
is *lower time-to-insight* and *better operational responsiveness*, which is what
this challenge asks for.

### Tech stack (2+ layers, as required)
| Layer | Used for |
|---|---|
| **Google Cloud — Cloud Storage** | store the raw comment corpus (parquet) |
| **Google Cloud — BigQuery** | (optional) query scored comments at scale |
| **NVIDIA — cuDF / `cudf.pandas`** | GPU-accelerate the heavy scoring + aggregation stage |
| **NVIDIA — GPU on Google Cloud / Colab** | the hardware the benchmark runs on |
| Deployment | Streamlit on Cloud Run / Streamlit Community Cloud / HF Spaces |

---

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python generate_data.py --rows 300000     # make a demo dataset
streamlit run app.py                        # open the dashboard
```

The dashboard has three parts:
1. **Community health** — which community is worst, at a glance.
2. **Live watchlist** — every thread ranked by meltdown risk (🔴 CRITICAL / ⚠️ WATCH / 🟢 CALM).
3. **Thread replay** — drag the time cursor and watch a real meltdown build, with
   the ⚠️ early-warning marker firing *ahead of* the dotted peak line.

## Reproduce the GPU benchmark
Open `notebooks/tremor_gpu_benchmark.ipynb` in Google Colab → set runtime to GPU →
Run all → download `benchmark_results.json` → commit it. Done.

---

## Repo layout
```
tremor_core.py     the engine: lexicon + scoring + risk + early-warning
generate_data.py   deterministic synthetic comment-stream generator (scales to millions)
benchmark.py       times the heavy stage on whatever engine pandas resolves to
app.py             Streamlit dashboard (health · watchlist · replay)
notebooks/         Colab GPU benchmark notebook
requirements.txt   CPU/app deps (cuDF is installed only in the GPU notebook)
```

See [docs/DEPLOY.md](docs/DEPLOY.md) for one-click deployment and
[docs/](docs/) for the slide deck outline, demo script, and brief.

> Data note: comments are **synthetic** (deterministic, seeded) so the project is
> fully reproducible with no scraping or private data. The lexicon uses PG "heat"
> markers, not slurs. The pipeline is platform-agnostic — point it at real
> Reddit/Discord exports and it works unchanged.
