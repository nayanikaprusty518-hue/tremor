# Tremor — Presentation deck (slide-by-slide content)

Drop this into the provided PPT template. ~10 slides, one idea each. Speaker
notes in _italics_.

---

## Slide 1 — Title
# 📡 Tremor
### Community meltdown early-warning radar
**Catches the tremors before the quake.**
_Your name · Google Cloud × NVIDIA Data Intelligence Challenge · July 2026_

---

## Slide 2 — The problem
- Online communities melt down: pile-ons, brigades, threads that turn toxic fast.
- **Volunteer moderators** (subreddits, Discord/Telegram/WhatsApp groups) have **no Trust & Safety team.**
- They find out **after** the damage — members leave, mods burn out.
- Documented: moderator burnout is real (U-Michigan, 2022) and worsening.

_Enterprise platforms get AI moderation. The person running a 40k-member subreddit by hand gets nothing._

---

## Slide 3 — Who uses this & the decision
**User:** a volunteer moderator / community manager.
**The decision, every hour:**
> "Which thread do I open right now — and is it about to get worse?"

Today that decision is a guess. Tremor makes it data-driven.

---

## Slide 4 — The solution
Tremor scores the whole comment stream and delivers:
- 🔴 **Ranked watchlist** — threads by meltdown risk (CRITICAL / WATCH / CALM)
- 🩺 **Community health score** — where to look first
- ⚠️ **Early-warning forecast** — fires *while a thread is still climbing, before it peaks*

_The forecast is the product. Detecting bad comments after the fact is easy; predicting the meltdown early is the value._

---

## Slide 5 — How it works (pipeline)
```
comments → score toxicity → bucket by time → aggregate      [HEAVY, GPU-accelerated]
         → rolling baseline → anomaly + velocity → risk 0–100
         → EARLY-WARNING forecast → watchlist · health · replay
```
- Toxicity = vectorized lexicon match over **every** comment.
- Risk gated by **volume-confidence** (one angry comment ≠ a meltdown).
- Forecast = toxicity + volume rising fast while level not yet extreme.

---

## Slide 6 — ⚡ Acceleration (the core of the challenge)
Same code, two engines — `cudf.pandas` is a **drop-in**:
```
python                     benchmark.py   →  CPU (pandas)
python -m cudf.pandas      benchmark.py   →  NVIDIA GPU (cuDF)
```
> **5,000,000 comments — CPU 97.6s → GPU 2.95s = 33× faster** *(measured on a Colab T4)*

**Why it matters:** to warn *early* you must score faster than comments arrive.
CPU = you learn about the fire afterwards. GPU = the warning arrives in time to act.

---

## Slide 7 — Results (measured vs planted ground truth)
| Metric | Result |
|---|---|
| Precision of flagged threads | **100%** |
| Recall of true meltdowns | **100% (68/68)** |
| Meltdowns forecast **before** peak | **66 / 68** |
| Typical lead time | **~60 minutes** |
| GPU speedup (heavy stage, 5M comments) | **33×** (97.6s → 2.95s) |

_Ground-truth meltdowns are planted in the synthetic data, so accuracy is measurable, not asserted._

---

## Slide 8 — Demo
_(Live dashboard / 3-min video)_
1. Community health → worst community.
2. Watchlist → the red CRITICAL threads.
3. **Thread replay** → drag the time cursor: toxicity + volume climb, the ⚠️ marker fires **before** the peak line. That gap is the moderator's window.
4. Sidebar → the cuDF-vs-pandas speedup banner.

---

## Slide 9 — Architecture & stack
- **Google Cloud:** Cloud Storage (raw corpus) · BigQuery (scale-out queries)
- **NVIDIA:** cuDF / `cudf.pandas` on a GPU (Colab / GCP GPU VM)
- **App:** Streamlit + Plotly, deployed on Cloud Run / Streamlit Cloud / HF Spaces
- **Data:** deterministic synthetic generator (no scraping); swaps 1:1 for real exports

_Two-plus layers as required: Google Cloud data layer + NVIDIA acceleration layer._

---

## Slide 10 — Impact & next steps
**Impact:** gives the powerless volunteer moderator enterprise-grade early warning — less burnout, healthier communities, intervention *before* harm.
**Next:** live streaming ingestion · per-community model tuning · Slack/Discord alert bot · Looker dashboard for multi-community orgs.

**📡 Tremor — catch the tremors before the quake.**
