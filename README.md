# PDF/JPG → Clean PDF Converter

Self-hosted, single-user web app that converts **image-based PDF files and JPG images** to clean, re-typeset PDF files using **Google Gemini** for OCR.

- ✅ Upload **PDF or JPG/JPEG** files — a JPG is auto-wrapped into a one-page PDF and OCR'd the same way
- ✅ OCR via Google Gemini (`gemini-3.5-flash-lite` by default, switchable with the `GEMINI_MODEL` variable)
- ✅ Languages: Traditional Chinese, Simplified Chinese, Japanese, Korean, English (and 100+ others)
- ✅ Auto-detects horizontal / vertical text layout per page
- ✅ Clean PDF with correct CJK font/CMap per detected language
- ✅ Re-embeds images, preserves hyperlinks, headings, TOC, footnotes, page numbers
- ✅ Async job queue with Start / Pause / Stop / Delete / Retry controls
- ✅ Google OAuth2 authentication (single-user, allowlist by email)
- ✅ One Gemini API call per page (efficient, low cost / quota)
- ✅ Per-page OCR caching — pause/resume without spending extra quota

---

## Why Gemini?

The previous PaddleOCR / Surya implementations needed too much RAM for a small cloud instance. Gemini moves OCR off-server entirely — the worker just sends each page image to Google's API and receives structured JSON back. The worker now uses **under 1 GB RAM** and needs no GPU or PyTorch.

Trade-off: each page = 1 Gemini API call, so **daily free-tier quota matters**. If you pause or a job fails partway through, the OCR results for completed pages are cached — resuming costs zero extra quota for those pages.

---

## Architecture

```
Browser → FastAPI (auth + UI + queue control)
                  ↕
         In-process store (fakeredis, or external Redis if REDIS_URL is set)
                  ↕
         Worker thread (Gemini API client, runs inside the same container)
                  ↕
    Google Gemini API  (https://generativelanguage.googleapis.com)
```

The API and Worker run as a **single process** — the worker is a background daemon thread started automatically when the app boots. No separate Redis service is needed; state is kept in-process via [fakeredis](https://github.com/cunla/fakeredis-py). If you set `REDIS_URL`, an external Redis is used instead (useful if you want persistent state across restarts).

---

## Prerequisites

- A [Railway](https://railway.com) account — the app is small enough for the Hobby plan (see [Memory Budget](#memory-budget))
- A **Google account** (Workspace or personal Gmail) for OAuth2 login
- A **Gemini API key** from Google AI Studio

> Any Docker host works — the app is a single container. The instructions below use Railway; `docker compose up` still works locally and unchanged.

---

## Step 1 — Get a Gemini API Key

1. Go to [https://aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Click **Get API key** in the left sidebar
4. Click **Create API key → Create API key in new project**
5. Copy the key — it looks like `AIzaSy...` (~39 characters)

The `config.yaml` ships with **paid-plan** rate limits (`rpm_limit: 2000`, `rpd_limit: 10000`).
If you are on the **free tier**, lower these values to stay within quota. Free-tier RPM/RPD limits
change over time and by account — check your current limits at
[Google AI Studio → Rate limits](https://aistudio.google.com) before deploying, and set
`rpm_limit` / `rpd_limit` in `config.yaml` accordingly.

The quota resets at midnight Pacific Time. Each PDF page (or each uploaded JPG) = 1 request.

---

## Step 2 — Google OAuth2 Setup (for app login)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Use the same project Gemini created (or any project)
3. **APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID**
4. Application type: **Web application**
5. Authorised redirect URI: `https://YOUR-DOMAIN/auth/callback`
   (on Railway this is `https://<service>.up.railway.app/auth/callback` — you can come back and fill this in after Step 4, once Railway has generated the domain)
6. Copy the **Client ID** and **Client Secret**

---

## Step 3 — Environment Variables (Railway → service → Variables)

**Required:**

| Variable | Value |
|---|---|
| `GEMINI_API_KEY` | API key from Step 1 |
| `GOOGLE_CLIENT_ID` | OAuth client ID from Step 2 |
| `GOOGLE_CLIENT_SECRET` | OAuth client secret from Step 2 |
| `ALLOWED_EMAIL` | Your Gmail address |
| `SECRET_KEY` | Random 32+ char string (`openssl rand -hex 32`) |

If any of these is missing the container exits at boot with a message naming exactly which ones — check the Deploy Logs.

**Optional:**

| Variable | Value |
|---|---|
| `APP_BASE_URL` / `BASE_URL` | Public origin, no trailing slash. **On Railway you can omit this** — the app falls back to `RAILWAY_PUBLIC_DOMAIN`. Set it explicitly once you attach a custom domain. |
| `GEMINI_MODEL` | Gemini model to OCR with, e.g. `gemini-3.5-flash`. Overrides `ocr.model_name` in `config.yaml`. Leave unset to use the config default (`gemini-3.5-flash-lite`). |
| `DATA_DIR` | Parent of `uploads/`, `outputs/`, `tmp-work/`. Set to a volume mount path (e.g. `/data`) to survive redeploys — see Step 4. |
| `REDIS_URL` | External Redis. On Railway: add a Redis database to the project and set this to `${{Redis.REDIS_URL}}`. |
| `PORT` | Injected by Railway automatically. Do not set it. |

> **No Redis service is required.** The app uses in-process storage ([fakeredis](https://github.com/cunla/fakeredis-py)) by default. Job state then lives only in the container, so a redeploy or restart clears the job list — add Redis if you want it to persist.

---

## Step 4 — Deploy to Railway

1. Push this repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo**, and pick it.
3. Railway reads `railway.json` and builds the root `Dockerfile` as **one service** — no Redis or Worker service needed.
4. **Settings → Networking → Generate Domain** to get a public URL, then add `https://<that-domain>/auth/callback` as the authorised redirect URI in Google Cloud Console (Step 2).
5. Add the variables from Step 3 and let it redeploy.

`railway.json` pins the deploy settings that matter here:

- `numReplicas: 1` — **required.** The worker runs as a thread inside the API process and (without `REDIS_URL`) keeps job state in memory. A second replica would have its own queue and its own copy of the output files, so downloads would intermittently 404.
- `healthcheckPath: /health` — Railway waits for this before switching traffic to a new deploy.
- `restartPolicyType: ON_FAILURE`.

### Persisting uploads and outputs (recommended)

Railway container filesystems are ephemeral: every redeploy starts from a fresh image, so converted PDFs from earlier deploys are gone. To keep them:

1. In the project canvas, right-click the service (or press `⌘K`) → **Attach Volume**, and set the mount path to `/data`.
2. Add the variable `DATA_DIR=/data`.

The app then writes `/data/uploads`, `/data/outputs` and `/data/tmp-work`. Retention still applies (`upload_retention_hours`, `output_retention_days` in `config.yaml`), so the volume does not grow without bound. Note that a volume also pins the service to one instance, which matches the single-replica requirement above.

### Local development

```bash
git clone https://github.com/YOUR-USERNAME/pdf2epub.git
cd pdf2epub

cp .env.example .env      # then fill in the values
docker compose up -d
docker compose logs -f
```

---

## Step 5 — Verify

1. `https://YOUR-DOMAIN/health` → `{"status":"ok","redis":true,"worker":true}`
   (`"status":"degraded"` with `"worker":false` for the first second or two after a deploy is normal — the worker is still initialising the Gemini client.)
2. `https://YOUR-DOMAIN` → login page
3. Sign in with the allowlisted Gmail
4. Upload a PDF or JPG, click **Start**, watch progress
5. When done, click **↓ Clean PDF** to download

---

## Configuration (`config.yaml`)

```yaml
ocr:
  engine: gemini
  model_name: "gemini-3.5-flash-lite"   # default; overridden by the GEMINI_MODEL env var when set
  rpm_limit: 2000                        # paid plan; lower to your free-tier RPM
  rpd_limit: 10000                       # paid plan; lower to your free-tier RPD
  max_retries: 5
  request_timeout_s: 180
  confidence_threshold: 0.7
  dpi: 300                               # rasterization DPI (300 balances quality vs memory)

pipeline:
  max_pdf_size_mb: 100
  page_batch_size: 10
  upload_retention_hours: 24
  output_retention_days: 7
  tmp_cleanup_on_complete: true

server:
  max_concurrent_jobs: 1
  port: 8080
  job_history_limit: 10
```

### Free tier overrides

If you are on the free Gemini tier, lower `rpm_limit` / `rpd_limit` in `config.yaml` to match the
current free-tier limits shown for your account at
[Google AI Studio → Rate limits](https://aistudio.google.com), e.g.:

```yaml
ocr:
  rpm_limit: 15
  rpd_limit: 1500
```

### Switching the model (e.g. to `gemini-3.5-flash` for higher accuracy)

The model is resolved in this order, most specific first:

1. the **`GEMINI_MODEL`** environment variable,
2. `ocr.model_name` in `config.yaml`,
3. the built-in default `gemini-3.5-flash-lite`.

**Preferred on Railway — no code change, no commit:** service → **Variables** →
**New Variable**, `GEMINI_MODEL` = `gemini-3.5-flash`. Railway redeploys the
service and the worker picks the new model up on the next job. Delete the
variable to fall back to `config.yaml`.

Confirm which model a running deploy is using with `GET /health`:

```json
{ "status": "ok", "redis": true, "worker": true, "gemini_model": "gemini-3.5-flash" }
```

**Or in `config.yaml`** (for local / committed defaults):

```yaml
ocr:
  model_name: gemini-3.5-flash
  rpm_limit: 2000    # paid plan; lower to your free-tier RPM
  rpd_limit: 10000   # paid plan; lower to your free-tier RPD
```

Then `docker compose restart app` to apply. Locally you can also set
`GEMINI_MODEL=gemini-3.5-flash` in `.env` — `docker-compose.yml` passes it through.

Note that the rate limits stay in `config.yaml`: a model switch does not change
`rpm_limit` / `rpd_limit`, so lower them there if the new model's quota is tighter.

---

## Memory Budget

| Component | Idle | Peak |
|---|---|---|
| OS + existing services | ~1.2 GB | ~1.2 GB |
| FastAPI + in-process store | ~200 MB | ~200 MB |
| Worker thread (Gemini client) | ~150 MB | ~400 MB (during page rasterization at 300 DPI) |
| **Total** | **~1.55 GB** | **~1.8 GB** |

Comfortably within Railway's default service size now that PaddleOCR/Surya are gone.

---

## Project Structure

```
ocr-pdf/
├── Dockerfile              # single-container build (API + Worker merged)
├── railway.json            # Railway build/deploy config (Dockerfile, healthcheck, 1 replica)
├── docker-compose.yml      # local dev — single service, no Redis
├── requirements.txt        # merged deps for API + Worker
├── config.yaml
├── settings.py             # env-driven paths + public URL (shared by API + Worker)
├── store.py                # Redis / fakeredis provider (shared by API + Worker)
├── .env.example
├── .dockerignore
│
├── Api/
│   ├── main.py             # /api/upload, /api/start, …
│   └── static/
│       ├── index.html      # main UI
│       └── login.html
│
└── Worker/
    ├── worker.py           # job loop + per-page OCR caching
    ├── ocr_engine.py       # abstract OCREngine interface
    ├── engine_factory.py   # only "gemini" registered
    ├── gemini_engine.py    # ⭐ the Gemini API integration
    ├── pdf_ingestion.py    # PyMuPDF + JPG→1-page-PDF conversion
    ├── structure_analysis.py # text → headings / paragraphs / footnotes / …
    └── pdf_assembly.py     # ReportLab / PyMuPDF: clean PDF output
```

---

## Troubleshooting

**Container exits at boot with `Missing required environment variable(s): …`:**
Add the named variables in Railway → service → **Variables**. The deploy will restart on its own once you save.

**Worker says `GEMINI_API_KEY environment variable is not set`:**
The variable is missing or empty. Check Railway → service → **Variables** (this one is read by the worker at job time, so the app boots fine without it and only fails when you start a conversion).

**Job fails with `Daily Gemini quota reached`:**
You've used all your free calls today. Wait until midnight Pacific Time (~UTC-7), or pause the job and resume tomorrow — cached pages will not be re-spent.

**429 errors in worker logs:**
The rate limiter should normally prevent this. If you see persistent 429s, your account might be on a more restrictive tier than the docs suggest — lower `rpm_limit` to 5 or 8 in `config.yaml`.

**Google OAuth callback error / `redirect_uri_mismatch`:**
The redirect URI the app sends is `APP_BASE_URL + /auth/callback`, and it must match Google Cloud Console character for character. If `APP_BASE_URL` is unset, the app derives it from Railway's `RAILWAY_PUBLIC_DOMAIN` — so after attaching a custom domain, set `APP_BASE_URL` explicitly (no trailing slash) and register the matching redirect URI.

**Downloads 404, or the job list empties after a deploy:**
Expected without a volume — the filesystem is ephemeral and (without `REDIS_URL`) job state is in memory. See "Persisting uploads and outputs" in Step 4. The same symptom appears if the service is scaled past one replica; keep `numReplicas: 1`.

---

## Cost Estimate

`gemini-3.5-flash-lite` on the **free tier** is 0¢ as long as you stay under your account's daily request quota. A JPG upload costs exactly 1 request (it is OCR'd as a single-page PDF).

If you exceed the free tier and enable billing, check current per-token pricing for `gemini-3.5-flash-lite` / `gemini-3.5-flash` at [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) — pricing and quotas are updated by Google independently of this project.

---

## License

MIT
