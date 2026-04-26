import os
import os.path
from pathlib import Path


def _first_writable_dir(*candidates: str) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        parent = path if path.exists() else path.parent
        if parent.exists() and os.access(parent, os.W_OK):
            return str(path)
    return candidates[-1]


PLATFORM = os.getenv("EEGFM_PLATFORM", "local")
DEFAULT_DATA_ROOT = os.getenv(
    "EEGFM_DATA_ROOT", os.path.join("/share/tmp/mahir", "eegfmbench")
)

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
PROJECT_ROOT = os.getenv("EEGFM_PROJECT_ROOT", _REPO_ROOT)
ASSETS_ROOT = os.path.join(PROJECT_ROOT, "assets")

DEFAULT_WRITE_ROOT = os.getenv(
    "EEGFM_WRITE_ROOT",
    _first_writable_dir(
        "/mnt/eegfmbench",
        os.path.join(str(Path.home()), "eegfmbench"),
        os.path.join(PROJECT_ROOT, ".eegfm"),
    ),
)

RUN_ROOT = os.getenv("EEGFM_RUN_ROOT", os.path.join(DEFAULT_WRITE_ROOT, "run"))
LOG_ROOT = os.getenv("EEGFM_LOG_ROOT", os.path.join(RUN_ROOT, "log"))
CONF_ROOT = os.getenv("EEGFM_CONF_ROOT", os.path.join(ASSETS_ROOT, "conf"))

if os.path.abspath(RUN_ROOT).startswith(os.path.abspath(DEFAULT_DATA_ROOT)):
    raise ValueError(
        "RUN_ROOT must not be inside EEGFM_DATA_ROOT. "
        "Set EEGFM_RUN_ROOT or EEGFM_WRITE_ROOT to a separate writable location."
    )

if os.path.abspath(LOG_ROOT).startswith(os.path.abspath(DEFAULT_DATA_ROOT)):
    raise ValueError(
        "LOG_ROOT must not be inside EEGFM_DATA_ROOT. "
        "Set EEGFM_LOG_ROOT or EEGFM_WRITE_ROOT to a separate writable location."
    )

DATABASE_RAW_ROOT = os.getenv(
    "EEGFM_DATABASE_RAW_ROOT", os.path.join(DEFAULT_DATA_ROOT, "raw")
)
DATABASE_PROC_ROOT = os.getenv(
    "EEGFM_DATABASE_PROC_ROOT", os.path.join(DEFAULT_DATA_ROOT, "proc")
)
DATABASE_CACHE_ROOT = os.getenv(
    "EEGFM_DATABASE_CACHE_ROOT", os.path.join(DEFAULT_DATA_ROOT, "cache")
)


def get_conf_file_path(path):
    if os.path.isabs(path):
        return path
    elif os.path.exists(path):
        return path
    elif os.path.exists(os.path.normpath(path)):
        return os.path.normpath(path)
    else:
        return os.path.join(CONF_ROOT, os.path.normpath(path))


def create_parent_dir(path):
    par_dir = os.path.dirname(path)
    if not os.path.exists(par_dir):
        os.makedirs(par_dir, exist_ok=True)
