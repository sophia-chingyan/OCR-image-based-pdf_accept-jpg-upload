"""
Runtime settings resolved from the environment
===============================================
Centralises the handful of things that differ between a local `docker compose`
run and a managed host such as Railway:

- where the config file lives,
- where writable data (uploads / outputs / scratch) lives,
- what the app's public URL is,
- which environment variables must be present before the app can boot.

Defaults reproduce the previous hardcoded `/app/...` layout, because the
Dockerfile still puts the source at `/app`.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

CONFIG_PATH = Path(os.getenv("CONFIG_PATH") or (REPO_ROOT / "config.yaml"))

# Parent of every writable directory. Container filesystems are ephemeral on
# Railway, so point DATA_DIR at the mount path of an attached volume (e.g.
# /data) to keep uploads and outputs across redeploys.
DATA_DIR = Path(os.getenv("DATA_DIR") or REPO_ROOT)

UPLOAD_DIR  = Path(os.getenv("UPLOAD_DIR")  or DATA_DIR / "uploads")
OUTPUT_DIR  = Path(os.getenv("OUTPUT_DIR")  or DATA_DIR / "outputs")
TMPWORK_DIR = Path(os.getenv("TMPWORK_DIR") or DATA_DIR / "tmp-work")


def ensure_dirs() -> None:
    """Create the writable directories; safe to call from both API and worker."""
    for d in (UPLOAD_DIR, OUTPUT_DIR, TMPWORK_DIR):
        d.mkdir(parents=True, exist_ok=True)


def public_base_url() -> str:
    """
    The origin the browser reaches this app on, without a trailing slash.

    An explicit APP_BASE_URL / BASE_URL always wins. Otherwise fall back to the
    domain Railway injects for the service, so a fresh deploy has a working
    OAuth redirect URI without any manual variable.
    """
    explicit = (os.environ.get("APP_BASE_URL") or os.environ.get("BASE_URL") or "").strip()
    if explicit:
        return explicit.rstrip("/")

    domain = (os.environ.get("RAILWAY_PUBLIC_DOMAIN") or "").strip()
    if domain:
        return "https://" + domain.split("://")[-1].rstrip("/")

    static_url = (os.environ.get("RAILWAY_STATIC_URL") or "").strip()
    if static_url:
        if not static_url.startswith(("http://", "https://")):
            static_url = "https://" + static_url
        return static_url.rstrip("/")

    port = (os.environ.get("PORT") or "8080").strip()
    return f"http://localhost:{port}"


def require_env(*names: str) -> None:
    """
    Fail fast with one actionable message instead of a bare KeyError buried in
    a traceback — deploy logs are the only debugging surface on a PaaS.
    """
    missing = [n for n in names if not (os.environ.get(n) or "").strip()]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them in your host's variables UI (Railway: service → "
              "Variables) or in your local .env — see .env.example."
        )


# Default when neither the environment nor config.yaml names a model.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def gemini_model(config_value: str | None = None) -> str:
    """
    Which Gemini model to use for OCR.

    Resolution order, most specific first:

    1. `GEMINI_MODEL` environment variable — lets the model be swapped from the
       host's variables UI (Railway: service → Variables) without a code change
       or a redeploy of config.yaml,
    2. `ocr.model_name` in config.yaml (passed in as `config_value`),
    3. DEFAULT_GEMINI_MODEL.
    """
    env_value = (os.environ.get("GEMINI_MODEL") or "").strip()
    if env_value:
        return env_value

    cfg_value = (config_value or "").strip() if isinstance(config_value, str) else ""
    return cfg_value or DEFAULT_GEMINI_MODEL


def poe_model(config_value: str | None = None) -> str:
    """
    Which Poe bot to use for OCR when ocr.engine is "poe".

    Unlike Gemini, Poe has no platform-wide default model — every request
    must name a specific bot, and which bots you can call depends on your
    Poe account. Resolution order, most specific first:

    1. `POE_MODEL` environment variable — lets the bot be swapped from the
       host's variables UI (Railway: service → Variables) without a code
       change or a redeploy of config.yaml,
    2. `ocr.poe_model_name` in config.yaml (passed in as `config_value`).

    Returns "" if neither is set, so PoeOCREngine.load() can fail fast with
    an actionable error rather than silently guessing a bot name that might
    not exist on your account.
    """
    env_value = (os.environ.get("POE_MODEL") or "").strip()
    if env_value:
        return env_value

    return (config_value or "").strip() if isinstance(config_value, str) else ""
