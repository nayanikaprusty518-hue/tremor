# Tremor — 3-minute demo video script

Target: **≤ 3:00**. Record screen (the dashboard) + voiceover. Timestamps are cues.
Keep the energy up; the money shot is the replay at 1:50.

---

**[0:00–0:20] Hook + problem**
> "Every online community has meltdowns — a thread turns toxic, people pile on,
> and by the time the moderator notices, the damage is done. Big platforms have AI
> for this. The volunteer running a subreddit or a Discord server has nothing.
> This is **Tremor** — a meltdown early-warning radar that tells them where to look,
> *before* the blow-up."

_On screen: the Tremor header + the header tagline._

---

**[0:20–0:45] What it does — health + KPIs**
> "Tremor scores the entire comment stream. Up top: how many comments we analysed,
> and how many threads are critical right now. Here's **community health** — one
> glance tells a moderator which community needs attention first."

_On screen: KPI row, then the community-health bar chart. Hover the worst community._

---

**[0:45–1:15] The watchlist**
> "This is the live watchlist — every thread ranked by meltdown risk. Red is
> critical, amber is watch. And this column is the important one: **early warning** —
> these threads were flagged while they were still heating up, not after."

_On screen: scroll the watchlist; point at a CRITICAL row and an ⚠️ early-warning tag._

---

**[1:15–2:10] The money shot — thread replay**
> "Let's replay one. I'll drag the time cursor and we watch the meltdown build in
> real time. Toxicity climbing… comment volume surging… and — right here — Tremor
> fires the early warning. Notice the dotted line: that's when the thread actually
> peaked. The warning came **about an hour earlier.** *That* gap is the moderator's
> window to step in and cool it down before it explodes."

_On screen: pick a forecast thread, drag the "Replay up to" slider left→right slowly.
Let the ⚠️ warning banner appear, then reach the dotted "peak" line. Pause on the gap._

---

**[2:10–2:45] Why acceleration matters**
> "To give that warning in time, you have to score comments faster than they arrive.
> Tremor's heavy stage runs on the exact same code on CPU or GPU — `cudf.pandas` is a
> drop-in. On five million comments, the NVIDIA GPU did in about three seconds what
> took the CPU a minute and a half — **thirty-three times faster.** That speed is the
> difference between a warning that arrives in time and one that arrives too late."

_On screen: the sidebar speedup banner, then flash the Colab notebook / benchmark output._

---

**[2:45–3:00] Close**
> "Google Cloud for the data, NVIDIA for the acceleration, real early warnings for
> the people who need them most. **Tremor — catch the tremors before the quake.**"

_On screen: back to the dashboard header. End card with repo + deployment link._

---

### Recording tips
- Generate data first (`python generate_data.py --rows 300000`) so it loads instantly.
- Pre-select a good thread before recording: sort watchlist, pick a CRITICAL one with an ⚠️ flag and many buckets (smoother replay).
- Do the slider drag slowly — it's the emotional beat.
- If the speedup banner isn't populated yet, run the Colab notebook and commit `benchmark_results.json` first.
