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
