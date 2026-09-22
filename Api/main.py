import os
import uuid
import json
import time
import shutil
import asyncio
import aiofiles
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import (
    HTMLResponse, RedirectResponse, JSONResponse, FileResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from store import get_async_redis
from settings import (
    CONFIG_PATH, UPLOAD_DIR, OUTPUT_DIR, TMPWORK_DIR,
    ensure_dirs, public_base_url, require_env, gemini_model, poe_model,
)

# ── Config ────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MAX_UPLOAD_BYTES = CFG["pipeline"]["max_pdf_size_mb"] * 1024 * 1024
OCR_ENGINE = CFG["ocr"].get("engine", "gemini").lower()  # config.yaml's default, overridable per-job — see _effective_ocr_engine
GEMINI_MODEL = gemini_model(CFG["ocr"].get("model_name"))
POE_MODEL = poe_model(CFG["ocr"].get("poe_model_name"))
IMAGE_EXTENSIONS = (".jpg", ".jpeg")
# Kept in sync by hand with Worker/engine_factory.py's ENGINES dict — not
# imported directly because that module (and the ones it imports) rely on
# Worker/ being on sys.path, which only happens once Worker.worker is first
# imported (inside lifespan(), i.e. after this module has already loaded).
VALID_OCR_ENGINES = ("gemini", "poe")
ensure_dirs()

require_env("SECRET_KEY", "ALLOWED_EMAIL", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET")

SECRET_KEY    = os.environ["SECRET_KEY"]
ALLOWED_EMAIL = os.environ["ALLOWED_EMAIL"].strip().lower()

BASE_URL = public_base_url()

_HTTPS_ONLY = BASE_URL.startswith("https://")

JOB_HISTORY = CFG["server"]["job_history_limit"]

MAX_WATCH_RETRIES = 5
WATCH_RETRY_DELAY = 0.01

SESSION_TTL_SECONDS = 30 * 24 * 3600


@asynccontextmanager
async def lifespan(app: FastAPI):
    from Worker.worker import main as worker_main
    t = threading.Thread(target=worker_main, daemon=True, name="pdf-worker")
    t.start()
    yield

app = FastAPI(title="PDF→Clean PDF Converter", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, https_only=_HTTPS_ONLY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[BASE_URL], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_static_dir = Path(__file__).parent / "static"

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.environ["GOOGLE_CLIENT_ID"],
    client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# ── OCR cache cleanup helper ──────────────────────────────────────────────────
async def _clear_ocr_cache(r, job_id: str) -> int:
    deleted = 0
    pattern = f"ocr:{job_id}:*"
    try:
        async for key in r.scan_iter(match=pattern, count=200):
            try:
                await r.delete(key)
                deleted += 1
            except Exception:
                pass
    except Exception:
        pass
    return deleted


def _clear_tmp_work(job_id: str) -> None:
    try:
        tmp_dir = TMPWORK_DIR / job_id
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass


async def _purge_job_record(r, job_id: str) -> None:
    """
    Delete a job's redis record, output files, source PDF, OCR cache and
    scratch directory. Only purges terminal-state jobs.
    """
    if not job_id:
        return
    raw = None
    try:
        raw = await r.get(f"job:{job_id}")
    except Exception:
        return
    if raw:
        try:
            job = json.loads(raw)
            status = job.get("status", "")
            if status in ("queued", "processing", "paused", "pending"):
                return
            for key in ("pdf_path", "clean_pdf_path", "searchable_pdf_path"):
                try:
                    p = Path(job.get(key, ""))
                    if p.exists():
                        p.unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception:
            pass
    _clear_tmp_work(job_id)
    await _clear_ocr_cache(r, job_id)
    try:
        await r.delete(f"job:{job_id}")
    except Exception:
        pass


async def create_session(request: Request, email: str) -> None:
    session_token = str(uuid.uuid4())
    r = await get_async_redis()
    try:
        await r.set(f"session:{session_token}", email, ex=SESSION_TTL_SECONDS)
    finally:
        await r.aclose()
    request.session["session_token"] = session_token

async def get_current_user(request: Request) -> Optional[str]:
    token = request.session.get("session_token")
    if not token:
        return None
    r = await get_async_redis()
    try:
        email = await r.get(f"session:{token}")
    finally:
        await r.aclose()
    return email

async def require_auth(request: Request) -> str:
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.get("/auth/login")
async def auth_login(request: Request):
    redirect_uri = f"{BASE_URL}/auth/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        return HTMLResponse("<h1>OAuth error. Please try again.</h1>", status_code=400)
    userinfo = token.get("userinfo") or {}
    email = (userinfo.get("email") or "").strip().lower()
    if email != ALLOWED_EMAIL:
        error_path = _static_dir / "login_error.html"
        try:
            async with aiofiles.open(error_path) as f:
                html = (await f.read()).replace(
                    "{{ error }}",
                    "This Google account is not authorized to use this app.",
                )
            return HTMLResponse(html, status_code=403)
        except Exception:
            return HTMLResponse("<h1>403 Access Denied</h1>", status_code=403)
    await create_session(request, email)
    return RedirectResponse(url="/", status_code=302)

@app.get("/auth/logout")
async def auth_logout(request: Request):
    token = request.session.pop("session_token", None)
    if token:
        r = await get_async_redis()
        try:
            await r.delete(f"session:{token}")
        finally:
            await r.aclose()
    return RedirectResponse(url="/", status_code=302)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = await get_current_user(request)
    static_path = _static_dir / "index.html"
    login_path  = _static_dir / "login.html"
    if user:
        async with aiofiles.open(static_path) as f:
            return HTMLResponse(await f.read())
    async with aiofiles.open(login_path) as f:
        return HTMLResponse(await f.read())

@app.get("/library", response_class=HTMLResponse)
async def library(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    library_path = _static_dir / "library.html"
    async with aiofiles.open(library_path) as f:
        return HTMLResponse(await f.read())

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    settings_path = _static_dir / "settings.html"
    async with aiofiles.open(settings_path) as f:
        return HTMLResponse(await f.read())

# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    user: str = Depends(require_auth),
):
    fname = (file.filename or "").strip()
    fname_lower = fname.lower()
    is_image = fname_lower.endswith(IMAGE_EXTENSIONS)
    if not fname or not (fname_lower.endswith(".pdf") or is_image):
        raise HTTPException(400, "Only PDF or JPG files are accepted.")

    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File exceeds {CFG['pipeline']['max_pdf_size_mb']}MB limit."
        )

    job_id     = str(uuid.uuid4())
    pdf_path   = UPLOAD_DIR / f"{job_id}.pdf"
    stage_path = (UPLOAD_DIR / f"{job_id}_src.jpg") if is_image else pdf_path
    total      = 0
    CHUNK      = 1024 * 1024

    try:
        async with aiofiles.open(stage_path, "wb") as out:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"File exceeds {CFG['pipeline']['max_pdf_size_mb']}MB limit.",
                    )
                await out.write(chunk)
    except HTTPException:
        try:
            stage_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except Exception:
        try:
            stage_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    if total == 0:
        try:
            stage_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(400, "Uploaded file is empty.")

    if is_image:
        try:
            from Worker.pdf_ingestion import convert_image_to_pdf
            convert_image_to_pdf(stage_path, pdf_path, dpi=CFG["ocr"]["dpi"])
        except Exception:
            raise HTTPException(400, "Uploaded file is not a valid JPG image.")
        finally:
            try:
                stage_path.unlink(missing_ok=True)
            except OSError:
                pass

    formats = ["clean"]

    page_count = 0
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()
    except Exception:
        pass

    job = {
        "job_id":              job_id,
        "filename":            fname,
        "status":              "pending",
        "progress":            0,
        "message":             "Waiting to start",
        "created_at":          int(time.time()),
        "pdf_path":            str(pdf_path),
        "clean_pdf_path":      "",
        "searchable_pdf_path": "",
        "error":               "",
        "stop_requested":      False,
        "pause_requested":     False,
        "page_count":          page_count,
        "output_formats":      formats,
    }

    r = await get_async_redis()
    try:
        await r.set(f"job:{job_id}", json.dumps(job))
        await r.lpush("job_history", job_id)
        try:
            orphan_ids = await r.lrange("job_history", JOB_HISTORY, -1)
        except Exception:
            orphan_ids = []
        await r.ltrim("job_history", 0, JOB_HISTORY - 1)
        for oid in (orphan_ids or []):
            await _purge_job_record(r, oid)
    finally:
        await r.aclose()

    return JSONResponse({
        "job_id": job_id, "filename": fname,
        "page_count": page_count, "output_formats": formats,
    })

# ── Status / History ──────────────────────────────────────────────────────────
@app.get("/api/status/{job_id}")
async def job_status(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        raw = await r.get(f"job:{job_id}")
    finally:
        await r.aclose()
    if not raw:
        raise HTTPException(404, "Job not found.")
    return JSONResponse(json.loads(raw))

@app.get("/api/history")
async def job_history(user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        ids = await r.lrange("job_history", 0, JOB_HISTORY - 1)
        jobs = []
        if ids:
            try:
                raws = await r.mget([f"job:{jid}" for jid in ids])
            except Exception:
                raws = None
            if raws is None:
                raws = []
                for jid in ids:
                    try:
                        raws.append(await r.get(f"job:{jid}"))
                    except Exception:
                        raws.append(None)
            for raw in raws:
                if raw:
                    try:
                        jobs.append(json.loads(raw))
                    except Exception:
                        pass
    finally:
        await r.aclose()
    return JSONResponse(jobs)

# ── Download: Clean PDF ───────────────────────────────────────────────────────
@app.get("/api/download/{job_id}/clean")
async def download_clean_pdf(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        raw = await r.get(f"job:{job_id}")
    finally:
        await r.aclose()
    if not raw:
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    if job["status"] != "done":
        raise HTTPException(400, "Job not complete.")
    p = Path(job.get("clean_pdf_path", ""))
    if not p.exists():
        try:
            job["clean_pdf_path"] = ""
            r2 = await get_async_redis()
            try:
                await r2.set(f"job:{job_id}", json.dumps(job))
            finally:
                await r2.aclose()
        except Exception:
            pass
        raise HTTPException(
            410,
            "Clean PDF is no longer available (output retention window expired). "
            "Please re-upload and reconvert."
        )
    return FileResponse(str(p), media_type="application/pdf",
                        filename=f"{Path(job['filename']).stem}_clean.pdf")

# ── Download: Searchable PDF ──────────────────────────────────────────────────
@app.get("/api/download/{job_id}/searchable")
async def download_searchable_pdf(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        raw = await r.get(f"job:{job_id}")
    finally:
        await r.aclose()
    if not raw:
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    if job["status"] != "done":
        raise HTTPException(400, "Job not complete.")
    p = Path(job.get("searchable_pdf_path", ""))
    if not p.exists():
        try:
            job["searchable_pdf_path"] = ""
            r2 = await get_async_redis()
            try:
                await r2.set(f"job:{job_id}", json.dumps(job))
            finally:
                await r2.aclose()
        except Exception:
            pass
        raise HTTPException(
            410,
            "Searchable PDF is no longer available (output retention window "
            "expired). Please re-upload and reconvert."
        )
    return FileResponse(str(p), media_type="application/pdf",
                        filename=f"{Path(job['filename']).stem}_searchable.pdf")

# ── Start / Pause / Stop / Delete ────────────────────────────────────────────
@app.post("/api/start/{job_id}")
async def start_job(job_id: str, request: Request, user: str = Depends(require_auth)):
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass
    if not isinstance(body, dict):
        body = {}

    language_hints = body.get("language_hints")
    if not isinstance(language_hints, list):
        language_hints = []

    requested_formats = body.get("output_formats")
    valid_formats = {"clean", "searchable"}
    if isinstance(requested_formats, list):
        formats = [f for f in requested_formats if f in valid_formats]
    else:
        formats = []
    if not formats:
        formats = ["clean"]

    r = await get_async_redis()
    try:
        for _attempt in range(MAX_WATCH_RETRIES):
            try:
                async with r.pipeline() as pipe:
                    await pipe.watch(f"job:{job_id}")
                    raw = await pipe.get(f"job:{job_id}")
                    if not raw:
                        raise HTTPException(404, "Job not found.")
                    job = json.loads(raw)
                    if job["status"] not in ("pending", "stopped", "failed", "paused"):
                        raise HTTPException(400, f"Cannot start from status: {job['status']}.")

                    pdf_path = job.get("pdf_path", "")
                    if not pdf_path or not Path(pdf_path).exists():
                        raise HTTPException(
                            410,
                            "Source PDF has been removed from the server "
                            "(retention window expired). Please re-upload."
                        )

                    # Clear any previous output files.
                    for old_key in ("clean_pdf_path", "searchable_pdf_path"):
                        old = job.get(old_key, "")
                        if old:
                            try:
                                p = Path(old)
                                if p.exists():
                                    p.unlink(missing_ok=True)
                            except OSError:
                                pass

                    job["output_formats"]      = formats
                    job["clean_pdf_path"]      = ""
                    job["searchable_pdf_path"] = ""
                    job["language_hints"]      = language_hints
                    job.update(status="queued", message="Queued", progress=0, error="",
                               stop_requested=False, pause_requested=False)

                    pipe.multi()
                    pipe.set(f"job:{job_id}", json.dumps(job))
                    pipe.lpush("job_queue", job_id)
                    await pipe.execute()
                break
            except HTTPException:
                raise
            except Exception:
                await asyncio.sleep(WATCH_RETRY_DELAY)
        else:
            raise HTTPException(500, "Concurrent update conflict, please retry.")
    finally:
        await r.aclose()
    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "output_formats": job["output_formats"],
    })

@app.post("/api/pause/{job_id}")
async def pause_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        for _attempt in range(MAX_WATCH_RETRIES):
            try:
                async with r.pipeline() as pipe:
                    await pipe.watch(f"job:{job_id}")
                    raw = await pipe.get(f"job:{job_id}")
                    if not raw:
                        raise HTTPException(404, "Job not found.")
                    job = json.loads(raw)
                    s = job["status"]

                    if s in ("done", "failed", "stopped", "paused"):
                        raise HTTPException(400, f"Cannot pause from status: {s}.")

                    if s == "pending":
                        job.update(status="paused", message="Paused by user.")
                    elif s == "queued":
                        job.update(status="paused", message="Paused by user.",
                                   pause_requested=True)
                    elif s == "processing":
                        job.update(pause_requested=True, message="Pausing…")

                    pipe.multi()
                    pipe.set(f"job:{job_id}", json.dumps(job))
                    if s == "queued":
                        pipe.lrem("job_queue", 0, job_id)
                    await pipe.execute()
                break
            except HTTPException:
                raise
            except Exception:
                await asyncio.sleep(WATCH_RETRY_DELAY)
        else:
            raise HTTPException(500, "Concurrent update conflict, please retry.")
    finally:
        await r.aclose()
    return JSONResponse({"job_id": job_id, "status": job["status"]})

@app.post("/api/stop/{job_id}")
async def stop_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        for _attempt in range(MAX_WATCH_RETRIES):
            try:
                async with r.pipeline() as pipe:
                    await pipe.watch(f"job:{job_id}")
                    raw = await pipe.get(f"job:{job_id}")
                    if not raw:
                        raise HTTPException(404, "Job not found.")
                    job = json.loads(raw)
                    s = job["status"]
                    if s in ("done", "failed", "stopped"):
                        raise HTTPException(400, f"Already terminal: {s}.")
                    if s == "pending":
                        job.update(status="stopped", message="Stopped by user.")
                    elif s == "queued":
                        job.update(status="stopped", message="Stopped by user.",
                                   stop_requested=True)
                    elif s in ("processing", "paused"):
                        job.update(stop_requested=True, message="Stopping…")

                    pipe.multi()
                    pipe.set(f"job:{job_id}", json.dumps(job))
                    if s == "queued":
                        pipe.lrem("job_queue", 0, job_id)
                    await pipe.execute()
                break
            except HTTPException:
                raise
            except Exception:
                await asyncio.sleep(WATCH_RETRY_DELAY)
        else:
            raise HTTPException(500, "Concurrent update conflict, please retry.")
    finally:
        await r.aclose()
    return JSONResponse({"job_id": job_id, "status": job["status"]})

@app.delete("/api/delete/{job_id}")
async def delete_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        raw = await r.get(f"job:{job_id}")
        if not raw:
            raise HTTPException(404, "Job not found.")
        job = json.loads(raw)
        if job["status"] == "processing":
            raise HTTPException(400, "Stop it first.")
        if job["status"] == "queued":
            await r.lrem("job_queue", 0, job_id)
        for key in ("pdf_path", "clean_pdf_path", "searchable_pdf_path"):
            try:
                p = Path(job.get(key, ""))
                if p.exists():
                    p.unlink(missing_ok=True)
            except OSError:
                pass
        _clear_tmp_work(job_id)
        await _clear_ocr_cache(r, job_id)
        await r.delete(f"job:{job_id}")
        await r.lrem("job_history", 0, job_id)
    finally:
        await r.aclose()
    return JSONResponse({"job_id": job_id, "deleted": True})

@app.get("/health")
async def health():
    r = await get_async_redis()
    redis_ok = False
    effective_engine = OCR_ENGINE
    try:
        await r.ping()
        redis_ok = True
        effective_engine = await _effective_ocr_engine(r)
    except Exception:
        pass
    finally:
        await r.aclose()
    from Worker.worker import get_worker_health
    worker_ok, worker_err = get_worker_health()
    return {
        "status": "ok" if (redis_ok and worker_ok) else "degraded",
        "redis": redis_ok,
        "worker": worker_ok,
        "worker_error": worker_err if not worker_ok else "",
        # The engine actually in effect right now — Settings-page choice if
        # one is saved, else config.yaml's ocr.engine. Falls back to the
        # static config.yaml value if Redis is unreachable.
        "ocr_engine": effective_engine,
        # Which model/bot this deploy is actually using for OCR, so a
        # GEMINI_MODEL / POE_MODEL change can be confirmed without reading
        # the logs. Only the field matching ocr_engine is actually in use.
        "gemini_model": GEMINI_MODEL,
        "poe_model": POE_MODEL,
    }

@app.get("/api/config")
async def get_config():
    return JSONResponse({
        "max_pdf_size_mb": CFG["pipeline"]["max_pdf_size_mb"],
    })

# ── User-provided OCR engine choice + Poe settings ───────────────────────────
# Lets the logged-in user pick which OCR engine runs (instead of only
# config.yaml's ocr.engine) and supply their own POE_API_KEY / POE_MODEL at
# runtime (Settings page) instead of only via environment variables. Stored
# in Redis so both the API and the worker thread see the same values
# (store.py shares one connection/FakeServer between them). The worker
# re-resolves the engine choice at the start of each job
# (Worker.worker._resolve_engine_name) and refreshes Poe credentials via
# PoeOCREngine.refresh_runtime_settings() — see Worker/poe_engine.py.
#
# NOTE: without REDIS_URL (the default), this lives in the in-process
# fakeredis store and is lost on restart/redeploy, same as job history —
# see "Persisting uploads and outputs" in the README.
async def _effective_ocr_engine(r) -> str:
    saved = (await r.get("settings:ocr_engine") or "").strip().lower()
    return saved if saved in VALID_OCR_ENGINES else OCR_ENGINE

async def _settings_payload(r) -> dict:
    api_key = (await r.get("settings:poe_api_key") or "").strip()
    model   = (await r.get("settings:poe_model") or "").strip()
    saved_engine = (await r.get("settings:ocr_engine") or "").strip().lower()
    if saved_engine not in VALID_OCR_ENGINES:
        saved_engine = ""
    return {
        # What's actually in effect right now (saved_engine, or the
        # config.yaml default when nothing is saved).
        "ocr_engine": saved_engine or OCR_ENGINE,
        # "" means "using the server default" — lets the UI show a
        # distinct "Server default" option rather than pre-selecting
        # whichever engine that happens to resolve to today.
        "ocr_engine_saved": saved_engine,
        "ocr_engine_default": OCR_ENGINE,
        "ocr_engines_available": list(VALID_OCR_ENGINES),
        "poe_api_key_set": bool(api_key),
        "poe_api_key_preview": (f"····{api_key[-4:]}" if len(api_key) >= 4 else ("····" if api_key else "")),
        "poe_model": model,
        # What PoeOCREngine falls back to when no value is saved here.
        "poe_model_env_default": POE_MODEL or None,
    }

@app.get("/api/settings")
async def get_settings(user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        return JSONResponse(await _settings_payload(r))
    finally:
        await r.aclose()

@app.post("/api/settings")
async def save_settings(request: Request, user: str = Depends(require_auth)):
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass
    if not isinstance(body, dict):
        body = {}

    r = await get_async_redis()
    try:
        # A field only changes the stored value when the client explicitly
        # includes it in the request body: an empty string (or, for
        # ocr_engine, any value that isn't a known engine name) clears it,
        # and omitting the key entirely leaves whatever is already saved
        # untouched (so the API key doesn't need to be retyped just to
        # change the model name or engine choice).
        if "ocr_engine" in body:
            choice = str(body.get("ocr_engine") or "").strip().lower()
            if choice in VALID_OCR_ENGINES:
                await r.set("settings:ocr_engine", choice)
            else:
                await r.delete("settings:ocr_engine")

        if "poe_model" in body:
            model = str(body.get("poe_model") or "").strip()
            if model:
                await r.set("settings:poe_model", model)
            else:
                await r.delete("settings:poe_model")

        if "poe_api_key" in body:
            key = str(body.get("poe_api_key") or "").strip()
            if key:
                await r.set("settings:poe_api_key", key)
            else:
                await r.delete("settings:poe_api_key")

        return JSONResponse(await _settings_payload(r))
    finally:
        await r.aclose()
