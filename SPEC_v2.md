# System Specification Document
## PDF to Clean PDF Converter — v2.0

---

### 1. Project Overview

A self-hosted, single-user web application that converts image-based PDF files into clean, re-typeset PDF output. Protected by Google OAuth2 authentication with email allowlist. The system performs OCR via Google Gemini, preserves document structure, re-embeds images, retains hyperlinks, and correctly handles horizontal and vertical CJK (Traditional Chinese, Simplified Chinese, Japanese, Korean) and English text. Deployed on Zeabur as a **single container** — no Redis service required.

---

### 2. Infrastructure

| Item | Spec |
|---|---|
| Platform | Zeabur |
| Server Plan | $3/mo minimum — 2 Core CPU, 4 GB RAM (no local OCR, minimal RAM needed) |
| Region | Tokyo, Japan (Tencent Cloud) |
| Deployment | Single Docker container (API + Worker in one process) |
| Zeabur Config | `zbpack.json` pointing to root `Dockerfile` |
| Public Domain | Assigned by Zeabur (HTTPS enforced by Zeabur reverse proxy) |
| Max PDF Upload Size | 100 MB |

---

### 3. System Architecture

**Design Pattern: Single-Process, Multi-Output Pipeline**

The entire application runs as **one container, one process**. The FastAPI API and the Worker are launched together — the Worker runs as a background daemon thread started automatically via FastAPI's lifespan hook. They share state through a store abstraction layer (`store.py`) that uses in-process `fakeredis` by default, or an external Redis if `REDIS_URL` is provided.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Single Container                         │
│                                                                 │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐ │
│  │   FastAPI (API Thread)   │  │  Worker (Daemon Thread)       │ │
│  │                          │  │                              │ │
│  │  /auth/login             │  │  Started on app boot via     │ │
│  │  /auth/callback          │  │  FastAPI lifespan hook       │ │
│  │  /auth/logout            │  │                              │ │
│  │  /api/upload             │  │  Pipeline:                   │ │
│  │  /api/start/{id}         │  │  1. Ingest PDF               │ │
│  │  /api/pause/{id}         │  │  2. Rasterize pages          │ │
│  │  /api/stop/{id}          │  │  3. Gemini OCR (1 call/page) │ │
│  │  /api/delete/{id}        │  │  4. Structure analysis       │ │
│  │  /api/status/{id}        │  │  5. Assemble outputs         │ │
│  │  /api/history            │  │  5. Assemble clean PDF       │ │
│  │  /api/download/{id}/     │  │                              │ │
│  │    clean                 │  │  Reads/writes job state      │ │
│  │  /health                 │  │  via get_sync_redis()        │ │
│  └─────────────┬────────────┘  └──────────────┬───────────────┘ │
│                │                              │                  │
│                └──────────┬───────────────────┘                  │
│                           │                                      │
│              ┌────────────▼──────────────┐                       │
│              │       store.py            │                       │
│              │  ┌────────────────────┐   │                       │
│              │  │  REDIS_URL set?    │   │                       │
│              │  │  Yes → real Redis  │   │                       │
│              │  │  No  → fakeredis   │   │                       │
│              │  │       (in-process) │   │                       │
│              │  └────────────────────┘   │                       │
│              │  get_sync_redis()  ← Worker thread                │
│              │  get_async_redis() ← FastAPI async                │
│              │  Both share same FakeServer instance              │
│              └───────────────────────────┘                       │
│                                                                 │
│  Volumes: /app/uploads · /app/outputs · /app/tmp-work           │
└─────────────────────────────────────────────────────────────────┘
         ↕ HTTPS
┌─────────────────────────────────────────────────────────────────┐
│                       User Browser                              │
└─────────────────────────────────────────────────────────────────┘
         ↕ API calls (1 per page)
┌─────────────────────────────────────────────────────────────────┐
│               Google Gemini API                                 │
│         (gemini-3.5-flash-lite — OCR + layout JSON)             │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4. Store Abstraction (`store.py`)

The store layer provides both sync and async Redis-compatible interfaces. This allows the Worker thread (sync) and FastAPI (async) to share state without coupling to a specific Redis backend.

```python
# No REDIS_URL set (default for Zeabur single-container):
fakeredis.FakeServer shared between both interfaces
→ get_sync_redis()  — used by Worker thread
→ get_async_redis() — used by FastAPI routes

# REDIS_URL set (optional, for persistence across restarts):
Real Redis via redis-py (sync) and redis.asyncio (async)
→ get_sync_redis()  — redis.from_url(REDIS_URL)
→ get_async_redis() — aioredis.from_url(REDIS_URL)
```

**State stored in Redis / fakeredis:**

| Key pattern | Content |
|---|---|
| `job:{job_id}` | Full job record (JSON) |
| `job_history` | List of recent job IDs (capped at `job_history_limit`) |
| `job_queue` | Queue of job IDs awaiting processing (BRPOP) |
| `session:{token}` | Email address for authenticated session |
| `ocr:{job_id}:{page}` | Per-page OCR result (JSON), for pause/resume |

---

### 5. Authentication

**Method:** Google OAuth2 via Authlib (FastAPI-native)

**Auth Flow:**

```
User visits app
      │
      ▼
Session token in cookie? ──No──► Redirect to /auth/login
      │                                    │
     Yes                                   ▼
      │                          Google OAuth2 consent screen
Token valid in store? ──No──►              │
      │                                    ▼
     Yes                        /auth/callback
      │                                    │
      ▼                         Extract email from Google userinfo
 Allow access                              │
                                 email == ALLOWED_EMAIL?
                                    │              │
                                   Yes             No
                                    │              │
                                    ▼              ▼
                             Generate UUID4     Return 403
                             session token    Access Denied
                                    │
                             Store in Redis/fakeredis
                             (no expiry — persistent until logout)
                                    │
                             Set HttpOnly + Secure cookie
                                    │
                             Redirect to /
```

**Note on callback URL:** The OAuth callback URL is built dynamically via `request.url_for("auth_callback")` rather than hardcoded from `BASE_URL`. This means Zeabur's automatic HTTPS reverse proxy domain is used correctly without extra configuration.

**Logout:** `GET /auth/logout` — deletes session token from store, clears cookie.

**Security Details:**

| Item | Decision |
|---|---|
| Session token | UUID4, stored server-side in store |
| Cookie flags | `HttpOnly`, `Secure`, `SameSite=Lax` |
| Session expiry | None — persistent until explicit logout |
| Allowlist | `ALLOWED_EMAIL` environment variable |
| 403 page | Generic "Access Denied" — reveals nothing |
| HTTPS | Enforced by Zeabur reverse proxy |

---

### 6. Output Format

The single output format is a clean, re-typeset PDF produced from one OCR pass.

| Format | Internal key | Output file | Description |
|---|---|---|---|
| Clean PDF | `clean` | `{job_id}_clean.pdf` | OCR text re-typeset into a fresh PDF using ReportLab with proper CJK typography |

**Download endpoint:**
- `GET /api/download/{job_id}/clean` — Clean PDF

---

### 7. OCR Engine: Google Gemini

**Model:** `gemini-3.5-flash-lite` — resolved from the `GEMINI_MODEL` environment variable if set, otherwise `ocr.model_name` in `config.yaml`, otherwise this default

**Strategy:** One API call per PDF page. The call returns OCR text, layout classification, and text direction in a single structured JSON response — no separate layout analysis pass needed.

**Prompt output schema:**
```json
{
  "direction": "horizontal" | "vertical",
  "blocks": [
    {
      "text": "recognised text",
      "type": "heading" | "paragraph" | "list-item" | "footnote" | "page-number" | "caption",
      "bbox": [x0, y0, x1, y1]
    }
  ]
}
```

**Caching:** The Gemini result for each page is cached by `id(page_image)` and cleared after the page is processed (`engine.reset_page_cache()`). This ensures the three interface methods (`detect_direction`, `recognize`, `get_layout`) share exactly one API call per page.

**Rate limiting:** Built-in sliding-window rate limiter (thread-safe). Blocks the Worker thread until a request can be issued without exceeding `rpm_limit` per 60-second window.

**Retry logic:** Up to `max_retries` attempts with exponential backoff. HTTP 429/503 → back off; other API errors → raise immediately.

**Image preparation:** Pages rasterized at `dpi` (default 400), then downscaled to max 2048px on the longest side before sending to Gemini to control token usage.

**Free tier limits for `gemini-3.5-flash-lite`:**
- 15 requests per minute (RPM)
- 1,000 requests per day (RPD)
- Quota resets at midnight Pacific Time

**Daily quota tracking:** Per-page OCR results are cached in Redis under `ocr:{job_id}:{page}`. On resume, cached pages are replayed without spending API quota.

---

### 8. Processing Pipeline (Per Job)

**Step 1 — PDF Ingestion (`pdf_ingestion.py`)**
- Open PDF with PyMuPDF
- Extract embedded images (with bounding boxes and format)
- Extract hyperlink annotations (URL + clickable bbox)
- Extract PDF metadata (title, author)
- Record total page count

**Step 2 — Page Rasterization**
- Rasterize each page at configured DPI using PyMuPDF
- Convert to BGR numpy array (for OpenCV compatibility)
- Process pages sequentially — never load all pages simultaneously

**Step 3 — Gemini OCR (one call per page)**
- Convert page image to JPEG (downsized to ≤2048px max side, quality 85)
- Send to Gemini with structured OCR prompt
- Receive: direction, text blocks with type classifications and bboxes
- Cache result in Redis for pause/resume; also cache in-memory for steps 4–5

**Step 4 — Structure Analysis (`structure_analysis.py`)**
- Map Gemini layout types to internal `LayoutType` classifications
- Detect headings by font-size ratio relative to page median
- Detect page numbers by position heuristics (near top/bottom center)
- Detect footnotes by small font size near bottom of page
- Detect list items by bullet/numbering patterns (CJK and Latin)
- Match hyperlinks to text blocks by bounding box overlap
- Build `StructuredPage` list with `StructuredElement` and `StructuredImage` objects

**Step 5 — Assemble Clean PDF (`pdf_assembly.py → assemble_clean_pdf`)**
- Re-typesets all OCR text into a new A4 PDF using ReportLab
- Registers CJK font (`STSong-Light` or `MSung-Light`, falls back to `Helvetica`)
- Maps element types to ReportLab styles: headings, body, footnotes, captions, list items
- Images embedded inline where detected
- Full title page generated from PDF metadata

**Step 6 — Cleanup**
- `/app/tmp-work/{job_id}/` deleted immediately after assembly
- `/app/uploads/*.pdf` deleted after `upload_retention_hours`
- `/app/outputs/*.pdf` deleted after `output_retention_days`

---

### 9. Job Lifecycle

```
Upload (PDF saved, page count read, job record created)
      │
      ▼
  [pending] ── user clicks Start ──► [queued] ── Worker picks up ──► [processing]
      │                                  │                                 │
      │                                  │ user stops/pauses               │ user stops/pauses
      ▼                                  ▼                                 ▼
  [stopped]                     [stopped/paused]                  [stopped/paused]*
                                                             (flag checked per page;
                                                              OCR cache preserved)
                                                                         │
                                            ┌────────────────────────────┤
                                            │                            │
                                            ▼                            ▼
                                         [done]                       [failed]
                                   (download available)        (retry available;
                                                               OCR cache preserved)
```

**Job record fields:**

| Field | Description |
|---|---|
| `job_id` | UUID4 |
| `filename` | Original uploaded filename |
| `status` | `pending` / `queued` / `processing` / `done` / `failed` / `stopped` / `paused` |
| `progress` | 0–100 integer |
| `message` | Human-readable status message |
| `created_at` | Unix timestamp |
| `pdf_path` | Path to uploaded PDF |
| `clean_pdf_path` | Path to clean PDF output |
| `error` | Error message if `status == failed` |
| `stop_requested` | Boolean flag — checked by Worker per page |
| `pause_requested` | Boolean flag — checked by Worker per page |
| `page_count` | Total PDF pages (set on upload) |
| `output_formats` | Always `["clean"]` |

---

### 10. Frontend

**Single production UI served from `Api/static/`:**

| File | Description |
|---|---|
| `Api/static/index.html` | Main app — format checkboxes, progress, multi-download, history |
| `Api/static/login.html` | Login page — Google OAuth button only |

**UI features:**
- Drag-and-drop or file picker (max 100 MB, PDF only)
- Upload → receive Job ID → show Start / Delete buttons
- Click **Start** → job queued → progress bar with 5-second polling
- Progress message: `OCR page X / Y…` then `Building clean PDF…`
- Pause / Resume support with progress preserved
- Download button when done: `↓ Clean PDF`
- Stop button during processing
- Retry + Delete on failure
- Job history cards: filename, pages, status, created date, actions
- Sign out link

---

### 11. API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | — | Login page or main app (based on session) |
| `GET` | `/auth/login` | — | Redirect to Google OAuth2 |
| `GET` | `/auth/callback` | — | Handle Google OAuth2 response |
| `GET` | `/auth/logout` | ✓ | Destroy session, redirect to login |
| `POST` | `/api/upload` | ✓ | Upload PDF, select formats, create job |
| `GET` | `/api/status/{id}` | ✓ | Get job status and progress |
| `GET` | `/api/history` | ✓ | Get last N jobs |
| `POST` | `/api/start/{id}` | ✓ | Queue a pending/stopped/failed/paused job |
| `POST` | `/api/pause/{id}` | ✓ | Request pause of queued/processing job |
| `POST` | `/api/stop/{id}` | ✓ | Request stop of queued/processing job |
| `DELETE` | `/api/delete/{id}` | ✓ | Delete job record and associated files |
| `GET` | `/api/download/{id}/clean` | ✓ | Download clean PDF |
| `GET` | `/health` | — | Health check (Redis ping) |

---

### 12. Configuration (`config.yaml`)

```yaml
ocr:
  engine: gemini
  model_name: "gemini-3.5-flash-lite"  # overridden by the GEMINI_MODEL env var when set
  rpm_limit: 15                         # match your Gemini tier
  rpd_limit: 1000
  max_retries: 3
  request_timeout_s: 120
  confidence_threshold: 0.7             # kept for OCREngine interface compatibility
  dpi: 400                              # rasterization DPI

pipeline:
  max_pdf_size_mb: 100
  page_batch_size: 5
  upload_retention_hours: 24
  output_retention_days: 7
  tmp_cleanup_on_complete: true

server:
  max_concurrent_jobs: 1
  port: 8080
  job_history_limit: 10
```

---

### 13. Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| API & Web Server | FastAPI (Python 3.11) | Single process with Worker thread |
| OAuth2 Client | Authlib | Google OAuth2, dynamic callback URL |
| OCR Engine | Google Gemini (`gemini-3.5-flash-lite`) | Via `google-genai` SDK |
| PDF Processing | PyMuPDF (fitz) | Ingestion, rasterization |
| Image Processing | OpenCV (`opencv-python-headless`) | BGR conversion |
| PDF Re-typesetting | ReportLab | Clean PDF output with CJK font support |
| Store | fakeredis (default) / Redis | In-process by default, external optional |
| Session Storage | fakeredis / Redis | Persistent until logout |
| Containerization | Docker (single container) | Root `Dockerfile` |
| Zeabur Config | `zbpack.json` | Points to root `Dockerfile` |
| Frontend | Plain HTML + Vanilla JS | No framework |

---

### 14. Environment Variables

Set in Zeabur UI (never in code):

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_CLIENT_ID` | ✅ | Google OAuth2 client ID |
| `GOOGLE_CLIENT_SECRET` | ✅ | Google OAuth2 client secret |
| `ALLOWED_EMAIL` | ✅ | Only this Gmail address can sign in |
| `SECRET_KEY` | ✅ | Cookie signing key (`openssl rand -hex 32`) |
| `GEMINI_API_KEY` | ✅ | Gemini API key from AI Studio |
| `APP_BASE_URL` or `BASE_URL` | ✅ | Public domain (`https://your-domain.zeabur.app`) |
| `GEMINI_MODEL` | ☐ | OCR model override (e.g. `gemini-3.5-flash`) — takes precedence over `ocr.model_name` |
| `REDIS_URL` | ☐ | External Redis URL — omit to use in-process fakeredis |

---

### 15. Deployment

**Zeabur (production):**
1. Push repository to GitHub
2. Create Zeabur project → connect GitHub repository
3. Zeabur detects `zbpack.json` → builds using root `Dockerfile`
4. Single service deployed — no Redis service needed
5. Set environment variables in Zeabur UI

**Local development:**
```bash
cp .env.example .env   # fill in values
docker compose up -d   # builds from root Dockerfile, single container
```

**Google OAuth2 setup (one-time):**
1. [console.cloud.google.com](https://console.cloud.google.com) → Credentials → Create OAuth 2.0 Client ID
2. Application type: Web application
3. Authorised redirect URI: `https://YOUR-ZEABUR-DOMAIN/auth/callback`
4. Copy Client ID and Client Secret to Zeabur env vars

**Gemini API key (one-time):**
1. [aistudio.google.com](https://aistudio.google.com) → Get API key → Create API key
2. Copy to `GEMINI_API_KEY` env var in Zeabur UI

---

### 16. Memory Budget (4 GB Server)

| Component | Idle | Peak (OCR processing) |
|---|---|---|
| OS + existing services | ~1.2 GB | ~1.2 GB |
| FastAPI + fakeredis | ~200 MB | ~200 MB |
| Worker thread (Gemini client) | ~100 MB | ~600 MB (page rasterization) |
| ReportLab (clean PDF) | — | ~200 MB |
| **Total** | **~1.5 GB** | **~2.2 GB** |
| **Free headroom** | **~2.5 GB** | **~1.8 GB** |

Comfortably fits on the **$3/mo (4 GB)** Zeabur plan.

---

### 17. Project Structure

```
ocr-pdf/
├── Dockerfile              # Single-container build (API + Worker)
├── docker-compose.yml      # Local dev only
├── zbpack.json             # Zeabur build config
├── requirements.txt        # Merged API + Worker dependencies
├── config.yaml             # All tuneable parameters
├── store.py                # Redis / fakeredis abstraction (shared by API + Worker)
├── .env.example            # Template for environment variables
├── .dockerignore
│
├── Api/
│   ├── main.py             # FastAPI app — routes, auth, job control
│   └── static/
│       ├── index.html      # Production UI: upload, progress, download, history cards
│       └── login.html      # Login: Google OAuth only
│
└── Worker/
    ├── worker.py           # Job queue consumer + pipeline orchestrator
    ├── ocr_engine.py       # Abstract OCREngine interface
    ├── engine_factory.py   # Engine instantiation by config (gemini only)
    ├── gemini_engine.py    # Gemini API integration, rate limiting, caching
    ├── pdf_ingestion.py    # PyMuPDF: images, links, metadata, rasterization
    ├── structure_analysis.py  # OCR blocks → headings/paragraphs/etc.
    └── pdf_assembly.py     # ReportLab: clean PDF output
```

---

### 18. Out of Scope (v2.0)

- Multi-user support / per-user quotas
- Cloud storage for output files (S3, GCS)
- Real-time WebSocket progress (currently polling)
- Table structure recognition (tables treated as images)
- RTL language support (Arabic, Hebrew)
- Translation between languages
- Batch upload (multiple PDFs at once)
