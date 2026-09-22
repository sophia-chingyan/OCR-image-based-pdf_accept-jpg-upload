"""
Worker — supports two output formats: clean PDF and searchable PDF.

Per-page OCR caching + Pause/Resume Support
============================================
After each page is OCR'd successfully, the page result is saved to Redis
under `ocr:{job_id}:{page_num}`. If the job is paused, stopped, or fails,
the next run skips pages that already have a cached result.

Pause mechanism:
- User clicks Pause → API sets pause_requested=True
- Worker checks pause_requested after each page
- If True → worker sets status="paused", saves progress, exits cleanly
- User clicks Resume/Start → job goes back to "queued" → worker picks up
  where it left off using cached OCR results

Skip-and-continue on OCR failure:
- If a page exhausts all Gemini retries, that page is SKIPPED (not the whole
  job). Its 0-based index is appended to `failed_pages` and the loop moves on.
- After all pages are attempted, every successfully OCR'd page is assembled
  into a downloadable PDF (skipped pages simply don't appear in the output).
- If at least one page was skipped the job is marked status="done" with
  partial=True. The job record carries `failed_pages` (1-based list) and a
  human-readable message listing all skipped page numbers / ranges.
- Only if EVERY page fails is the job marked status="failed".
- The OCR cache for completed pages is preserved so a future retry
  re-attempts only the skipped pages without re-spending quota.

Output formats:
- "clean"      → assemble_clean_pdf() — re-typeset fresh PDF
- "searchable" → assemble_searchable_pdf() — original scan + invisible text layer
Both can be requested simultaneously; each produces its own output file.
"""
from __future__ import annotations
import os, sys, json, time, shutil, logging, traceback, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Repo root, so `store` / `settings` resolve even when the process was not
# started with PYTHONPATH pointing at it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from store import get_sync_redis
from settings import (
    CONFIG_PATH, UPLOAD_DIR, OUTPUT_DIR, TMPWORK_DIR, ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("worker")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

ensure_dirs()

DPI        = CFG["ocr"]["dpi"]
BATCH_SIZE = CFG["pipeline"]["page_batch_size"]
CLEANUP    = CFG["pipeline"].get("tmp_cleanup_on_complete", True)

_worker_healthy: bool = False
_worker_error: str = ""


def get_worker_health() -> tuple[bool, str]:
    return _worker_healthy, _worker_error


# ── Job state helpers ────────────────────────────────────────────────────────

def update_job(r, job_id: str, **kw):
    """
    Atomic field-merge update of a job record. Retries up to 5 times if
    a concurrent writer (e.g. the API setting stop/pause flags) touches
    the same key during our read-modify-write window.
    """
    key = f"job:{job_id}"
    for _attempt in range(5):
        try:
            with r.pipeline() as pipe:
                pipe.watch(key)
                raw = pipe.get(key)
                if not raw:
                    pipe.unwatch()
                    return
                job = json.loads(raw)
                job.update(kw)
                pipe.multi()
                pipe.set(key, json.dumps(job))
                pipe.execute()
                return
        except Exception:
            time.sleep(0.01)
    raw = r.get(key)
    if not raw:
        return
    job = json.loads(raw)
    job.update(kw)
    r.set(key, json.dumps(job))


# ── OCR page cache helpers ───────────────────────────────────────────────────

def _ocr_cache_key(job_id: str, page_num: int) -> str:
    return f"ocr:{job_id}:{page_num}"


def _save_ocr_page(r, job_id: str, page_num: int, page_result: dict) -> None:
    try:
        r.set(_ocr_cache_key(job_id, page_num), json.dumps(page_result))
    except Exception as e:
        logger.warning(f"Failed to cache OCR for {job_id} page {page_num}: {e}")


def _load_ocr_page(r, job_id: str, page_num: int) -> dict | None:
    try:
        raw = r.get(_ocr_cache_key(job_id, page_num))
        if not raw:
            return None
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        return data
    except Exception as e:
        logger.warning(f"Failed to load cached OCR for {job_id} page {page_num}: {e}")
        return None


def _count_cached_pages(r, job_id: str, total: int) -> int:
    if total <= 0:
        return 0
    try:
        with r.pipeline(transaction=False) as pipe:
            for i in range(total):
                pipe.exists(_ocr_cache_key(job_id, i))
            results = pipe.execute()
        return sum(1 for x in results if x)
    except Exception as e:
        logger.warning(f"_count_cached_pages pipeline failed ({e}), falling back")
        n = 0
        for i in range(total):
            try:
                if r.exists(_ocr_cache_key(job_id, i)):
                    n += 1
            except Exception:
                pass
        return n


def _clear_ocr_cache(r, job_id: str) -> int:
    deleted = 0
    pattern = f"ocr:{job_id}:*"
    try:
        for key in r.scan_iter(match=pattern, count=200):
            try:
                r.delete(key)
                deleted += 1
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Failed to clear OCR cache for {job_id}: {e}")
    return deleted


def _format_page_list(page_indexes: list[int]) -> str:
    """
    Turn a list of 0-based page indexes into a compact, human-readable,
    1-based range string. e.g. [0, 1, 2, 5, 7, 8, 9] → "1–3, 6, 8–10".
    """
    if not page_indexes:
        return ""
    pages = sorted(set(p + 1 for p in page_indexes))
    parts: list[str] = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        parts.append(str(start) if start == prev else f"{start}–{prev}")
        start = prev = p
    parts.append(str(start) if start == prev else f"{start}–{prev}")
    return ", ".join(parts)



def run_pipeline(r, job: dict, engine) -> None:
    from pdf_ingestion import ingest_pdf, rasterize_page
    from structure_analysis import (
        analyse_page, build_toc, DocumentStructure, detect_dominant_language
    )
    from pdf_assembly import assemble_clean_pdf, assemble_searchable_pdf

    job_id   = job["job_id"]
    pdf_path = Path(job["pdf_path"])
    tmp_dir  = TMPWORK_DIR / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    language_hints = job.get("language_hints") or []
    engine.set_language_hints(language_hints)
    engine.reset_page_cache()
    # Pick up any POE_API_KEY / POE_MODEL saved via the Settings page since
    # this engine was loaded. Not part of the OCREngine ABC (only PoeOCREngine
    # implements it), hence the hasattr guard.
    if hasattr(engine, "refresh_runtime_settings"):
        engine.refresh_runtime_settings(r)

    ingested = None

    def check_stop_or_pause() -> str | None:
        raw = r.get(f"job:{job_id}")
        if raw:
            cur = json.loads(raw)
            if cur.get("stop_requested") or cur.get("status") == "stopped":
                update_job(r, job_id, status="stopped", message="Stopped by user.",
                           stop_requested=False, pause_requested=False)
                return "stop"
            if cur.get("pause_requested"):
                update_job(r, job_id, status="paused", message="Paused by user.",
                           pause_requested=False, stop_requested=False)
                return "pause"
        return None

    try:
        if check_stop_or_pause() is not None:
            return

        update_job(r, job_id, status="processing", message="Ingesting PDF…", progress=2)
        ingested = ingest_pdf(pdf_path, original_filename=job.get("filename", ""))
        total_pages = ingested.meta.total_pages
        structured_pages = []
        image_id_counter = [0]

        cached_n = _count_cached_pages(r, job_id, total_pages)
        if cached_n > 0:
            logger.info(f"Job {job_id}: resuming with {cached_n}/{total_pages} pages cached")
            update_job(r, job_id,
                       message=f"Resuming · {cached_n}/{total_pages} pages cached",
                       progress=2)

        # Pages that exhausted all OCR retries and were skipped (0-based indexes).
        failed_pages: list[int] = []
        last_error = ""

        for page_num in range(total_pages):
            action = check_stop_or_pause()
            if action == "stop":
                return
            elif action == "pause":
                logger.info(f"Job {job_id}: paused at page {page_num}/{total_pages}")
                return

            progress = int(5 + (page_num / total_pages) * 80)
            cached = _load_ocr_page(r, job_id, page_num)

            if cached is not None:
                update_job(r, job_id,
                           message=f"Page {page_num+1} / {total_pages} (cached)…",
                           progress=progress)
                engine.prime_page_cache_from_dict(cached)
                page_img = _CachedPageSentinel(page_num)
            else:
                update_job(r, job_id,
                           message=f"OCR page {page_num+1} / {total_pages}…",
                           progress=progress)
                page_img = rasterize_page(ingested.doc, page_num, dpi=DPI)

            # Per-page OCR is isolated: a page that exhausts all retries is
            # SKIPPED (recorded in failed_pages) so the loop continues to the
            # next page rather than aborting the whole job.
            try:
                direction     = engine.detect_direction(page_img)
                text_blocks   = engine.recognize(page_img, direction)
                layout_blocks = engine.get_layout(page_img)
            except Exception as e:
                engine.reset_page_cache()
                try:
                    del page_img
                except Exception:
                    pass
                failed_pages.append(page_num)
                last_error = str(e)
                logger.error(
                    f"Job {job_id}: OCR failed on page {page_num+1}/{total_pages}: {e} "
                    f"— skipping and continuing"
                )
                update_job(r, job_id,
                           message=f"Skipped page {page_num+1} / {total_pages} "
                                   f"(OCR failed) — continuing…",
                           progress=progress)
                if total_pages > 50 or (page_num + 1) % BATCH_SIZE == 0:
                    gc.collect()
                continue

            if cached is None:
                page_result = engine.export_last_page_result()
                if page_result is not None:
                    _save_ocr_page(r, job_id, page_num, page_result)

            page_info = ingested.pages[page_num]
            sp = analyse_page(page_number=page_num, text_blocks=text_blocks,
                              layout_blocks=layout_blocks, page_info=page_info,
                              direction=direction, image_id_counter=image_id_counter,
                              dpi=DPI)
            structured_pages.append(sp)
            engine.reset_page_cache()
            del page_img, text_blocks, layout_blocks
            if total_pages > 50 or (page_num + 1) % BATCH_SIZE == 0:
                gc.collect()

        # Every page failed — nothing to assemble.
        if failed_pages and not structured_pages:
            update_job(r, job_id, status="failed", message="Conversion failed.",
                       error=f"OCR failed on all {total_pages} page(s). "
                             f"Last error: {last_error}")
            logger.error(f"Job {job_id}: all pages failed OCR — {last_error}")
            return

        is_partial = bool(failed_pages)
        pages_done = len(structured_pages)

        toc = build_toc(structured_pages)
        dominant_lang = detect_dominant_language(structured_pages)
        logger.info(f"Job {job_id}: detected dominant language: {dominant_lang}")
        structure = DocumentStructure(
            title=ingested.meta.title, author=ingested.meta.author,
            pages=structured_pages, toc=toc, dominant_language=dominant_lang)

        # ── Assemble requested output formats ────────────────────────────────
        formats = job.get("output_formats") or ["clean"]
        formats = [f for f in formats if f in ("clean", "searchable")] or ["clean"]

        produced: dict = {}
        assembly_errors: list[str] = []

        if "clean" in formats:
            update_job(r, job_id,
                       message=("Building clean PDF (partial)…" if is_partial else "Building clean PDF…"),
                       progress=87)
            try:
                p = OUTPUT_DIR / f"{job_id}_clean.pdf"
                assemble_clean_pdf(structure, p, source_pdf_path=pdf_path)
                produced["clean_pdf_path"] = str(p)
            except Exception as e:
                logger.error(f"Clean PDF assembly failed for {job_id}: {e}\n{traceback.format_exc()}")
                assembly_errors.append(f"clean: {e}")

        if "searchable" in formats:
            update_job(r, job_id,
                       message=("Building searchable PDF (partial)…" if is_partial else "Building searchable PDF…"),
                       progress=93)
            try:
                p = OUTPUT_DIR / f"{job_id}_searchable.pdf"
                assemble_searchable_pdf(structure, pdf_path, p, dpi=DPI)
                produced["searchable_pdf_path"] = str(p)
            except Exception as e:
                logger.error(f"Searchable PDF assembly failed for {job_id}: {e}\n{traceback.format_exc()}")
                assembly_errors.append(f"searchable: {e}")

        # ── Determine final status ───────────────────────────────────────────
        if produced:
            done_fields = {"status": "done", "progress": 100}
            done_fields.update(produced)

            if is_partial:
                # Keep OCR cache: completed pages are preserved so a future
                # retry re-attempts only the skipped pages without re-spending
                # quota on the ones already done.
                failed_1based = [p + 1 for p in sorted(failed_pages)]
                failed_list = _format_page_list(failed_pages)
                note = (f"Partial — {pages_done}/{total_pages} pages OCR'd. "
                        f"Skipped {len(failed_pages)} page(s) that could not be "
                        f"OCR'd: {failed_list}")
                if assembly_errors:
                    note += " | assembly: " + "; ".join(assembly_errors)
                done_fields["message"] = note
                done_fields["error"] = note
                done_fields["partial"] = True
                done_fields["pages_completed"] = pages_done
                done_fields["pages_total"] = total_pages
                done_fields["failed_pages"] = failed_1based           # full list
                done_fields["failed_page"] = failed_1based[0]         # back-compat
                logger.warning(f"Job {job_id} partial success: {note}")
            else:
                removed = _clear_ocr_cache(r, job_id)
                if removed:
                    logger.info(f"Job {job_id}: cleared {removed} cached OCR pages")
                done_fields["partial"] = False
                if assembly_errors:
                    err_detail = "; ".join(assembly_errors)
                    done_fields["message"] = f"Partial success (errors: {err_detail})"
                    done_fields["error"] = err_detail
                    logger.warning(f"Job {job_id} partial success: {err_detail}")
                else:
                    done_fields["message"] = "Complete"

            update_job(r, job_id, **done_fields)
            logger.info(f"Job {job_id} done: {produced}")
        else:
            if is_partial:
                err = (f"{pages_done} page(s) OCR'd but none could be assembled; "
                       f"skipped pages: {_format_page_list(failed_pages)}.")
            else:
                err = "; ".join(assembly_errors) or "No output produced"
            update_job(r, job_id, status="failed", message="Conversion failed.", error=err)
            logger.error(f"Job {job_id} failed: {err}")

    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}\n{traceback.format_exc()}")
        update_job(r, job_id, status="failed", message="Conversion failed.", error=str(exc))
    finally:
        if ingested is not None:
            try:
                ingested.doc.close()
            except Exception:
                pass
        if CLEANUP and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


class _CachedPageSentinel:
    """
    Lightweight stand-in for a rasterized page used only when a page's
    OCR result has been replayed from Redis.
    """
    __slots__ = ("page_num",)

    def __init__(self, page_num: int):
        self.page_num = page_num


def cleanup_expired_files(r):
    """
    Periodic disk cleanup:
    - delete uploads older than upload_retention_hours, unless the job
      is in a state the user may still want to resume/retry;
    - delete outputs older than output_retention_days;
    - delete stale tmp-work scratch directories.
    """
    ur  = CFG["pipeline"]["upload_retention_hours"] * 3600
    orr = CFG["pipeline"]["output_retention_days"] * 86400
    now = time.time()

    protected_job_ids: set[str] = set()
    try:
        for key in r.scan_iter(match="job:*", count=200):
            try:
                raw = r.get(key)
                if not raw:
                    continue
                job = json.loads(raw)
                status = job.get("status")
                if status not in ("done",):
                    protected_job_ids.add(job.get("job_id", ""))
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"cleanup: could not enumerate jobs ({e}); "
                       f"skipping upload pruning to be safe")
        protected_job_ids = None

    if protected_job_ids is not None:
        for f in UPLOAD_DIR.glob("*.pdf"):
            if now - f.stat().st_mtime > ur:
                jid = f.stem
                if jid in protected_job_ids:
                    continue
                f.unlink(missing_ok=True)

    for f in OUTPUT_DIR.glob("*.pdf"):
        if now - f.stat().st_mtime > orr:
            f.unlink(missing_ok=True)

    if TMPWORK_DIR.exists():
        for d in TMPWORK_DIR.iterdir():
            if not d.is_dir():
                continue
            try:
                if now - d.stat().st_mtime > ur:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass


# OCR engine instances, keyed by engine name, created + loaded lazily on
# first use and cached for the worker thread's lifetime so each engine's
# internal state (rate limiter, Gemini's daily-quota counter, Poe's page
# cache) persists across jobs — same as the single module-level engine used
# to, just now there can be one entry per engine name instead of exactly one.
# Only ever touched from this single worker thread, so no locking needed
# (max_concurrent_jobs: 1 / numReplicas: 1 — see README).
_ENGINE_CACHE: dict = {}


def _resolve_engine_name(r) -> str:
    """
    Which OCR engine this job should use: the Settings page's choice
    (settings:ocr_engine in Redis) if one is saved, else ocr.engine in
    config.yaml. Falls back to the config.yaml default on any Redis error
    or an unrecognised saved value, so a stale/bad setting can't wedge the
    worker.
    """
    from engine_factory import ENGINES
    try:
        saved = (r.get("settings:ocr_engine") or "").strip().lower()
    except Exception as e:
        logger.warning(f"Could not read settings:ocr_engine from Redis: {e}")
        saved = ""
    if saved in ENGINES:
        return saved
    return CFG["ocr"].get("engine", "gemini").lower()


def _get_or_load_engine(name: str):
    """
    Return a loaded engine for `name`, building + loading it on first use.
    Propagates whatever engine.load() raises — the caller treats that as
    this JOB failing, not a reason to stop the worker loop, since switching
    the Settings-page engine choice (or fixing the other engine's
    credentials) may make the next job succeed.
    """
    if name in _ENGINE_CACHE:
        return _ENGINE_CACHE[name]
    from engine_factory import get_engine
    cfg = dict(CFG["ocr"])
    cfg["engine"] = name
    engine = get_engine(cfg)
    engine.load()
    _ENGINE_CACHE[name] = engine
    return engine


def main():
    logger.info("Worker starting…")

    # Which OCR engine to use is now resolved per job (Settings page, or
    # config.yaml as the fallback default) rather than fixed at startup, so
    # there's nothing engine-specific left to validate before the loop
    # starts — a bad/missing credential for one engine no longer prevents
    # the worker thread (or the OTHER engine) from running at all.
    global _worker_healthy, _worker_error
    _worker_healthy = True
    _worker_error = ""
    logger.info("Worker ready.")

    r = get_sync_redis()
    last_cleanup = time.time()

    while True:
        try:
            if time.time() - last_cleanup > 3600:
                try:
                    cleanup_expired_files(r)
                except Exception as e:
                    logger.warning(f"cleanup_expired_files raised: {e}")
                last_cleanup = time.time()

            result = r.brpop("job_queue", timeout=30)
            if result is None:
                continue
            _, job_id = result
            raw = r.get(f"job:{job_id}")
            if not raw:
                continue
            try:
                job = json.loads(raw)
            except Exception as e:
                logger.warning(f"Corrupt job record for {job_id}: {e}")
                continue
            if job.get("status") != "queued":
                continue

            engine_name = _resolve_engine_name(r)
            try:
                engine = _get_or_load_engine(engine_name)
            except Exception as exc:
                logger.error(f"Job {job_id}: OCR engine '{engine_name}' failed to initialise: {exc}")
                update_job(r, job_id, status="failed", message="Conversion failed.",
                           error=f"OCR engine '{engine_name}' is not ready: {exc}")
                continue

            logger.info(f"Processing {job_id}: {job.get('filename')} (engine={engine_name})")
            update_job(r, job_id, status="processing", message="Starting…", progress=1)
            run_pipeline(r, job, engine)
        except Exception as exc:
            logger.error(
                f"Worker main loop caught unexpected error: {exc}\n"
                f"{traceback.format_exc()}"
            )
            time.sleep(2.0)


if __name__ == "__main__":
    main()
