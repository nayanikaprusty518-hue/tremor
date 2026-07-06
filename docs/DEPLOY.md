# Tremor — deployment & submission guide

Everything here needs **your** accounts (GitHub, a host, Colab). Total time
< 30 min. Do them in this order.

---

## Step 1 — Push to GitHub  (~3 min)

From the `tremor/` folder:

```bash
git init
git add .
git commit -m "Tremor: community meltdown early-warning radar"
git branch -M main
# create an empty repo named 'tremor' on github.com first, then:
git remote add origin https://github.com/YOUR_USERNAME/tremor.git
git push -u origin main
```

✅ **Deliverable: GitHub repository link** = `https://github.com/YOUR_USERNAME/tremor`

> `comments.parquet` (4.7 MB) is committed so the deployed app has data instantly.
> `.venv/` and `big.parquet` are git-ignored.

---

## Step 2 — Get the real GPU numbers  (~3 min)

1. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → `notebooks/tremor_gpu_benchmark.ipynb`.
2. **Runtime → Change runtime type → GPU (T4)** → Save.
3. In cell 3, set `REPO_URL` to your repo URL from Step 1.
4. **Runtime → Run all.** Wait ~3 min.
5. Download `benchmark_results.json` (left files panel) and commit it:
   ```bash
   git add benchmark_results.json && git commit -m "Add GPU benchmark" && git push
   ```

Now the dashboard's sidebar shows the real cuDF-vs-pandas speedup, and you have the
number for slide 6 / the demo.

---

## Step 3 — Deploy for a public URL  (pick ONE)

### Option A — Streamlit Community Cloud  ⭐ easiest, free
1. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
2. **New app** → pick your `tremor` repo, branch `main`, main file `app.py`.
3. **Deploy.** First load auto-generates data if needed (self-healing).
4. Copy the `https://<something>.streamlit.app` URL.

### Option B — Hugging Face Spaces  (free)
1. [huggingface.co/new-space](https://huggingface.co/new-space) → SDK = **Streamlit**.
2. Push these files to the Space repo (or link the GitHub repo). Ensure the Space
   `README.md` starts with this frontmatter:
   ```yaml
   ---
   title: Tremor
   sdk: streamlit
   app_file: app.py
   pinned: false
   ---
   ```
3. It builds and gives you `https://huggingface.co/spaces/YOUR_USERNAME/tremor`.

### Option C — Google Cloud Run  (uses the Dockerfile; on-brand for the challenge)
```bash
gcloud run deploy tremor \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 1Gi
```
Copy the `https://tremor-....run.app` URL it prints.

✅ **Deliverable: Deployment link** = the URL from whichever option you chose.

---

## Step 4 — Record the demo video  (~15 min)
Follow `docs/DEMO_SCRIPT.md` (≤ 3 min). Screen-record the deployed app, upload to
YouTube/Drive (set to public/anyone-with-link).

✅ **Deliverable: Demo video link**

---

## Step 5 — Slides + brief
- Fill the template using `docs/SLIDES.md` (paste your GPU number into slide 6).
- Paste `docs/BRIEF.md`'s short paragraph into the "brief description" field.

✅ **Deliverables: Final Project PPT + Brief description**

---

## Final submission checklist
- [ ] Deployment link (public, opens the dashboard)
- [ ] Final Project PPT (uploaded)
- [ ] GitHub repository link (public)
- [ ] Demo video link (≤ 3 min, public)
- [ ] Brief description
- [ ] All links confirmed working in an incognito window
