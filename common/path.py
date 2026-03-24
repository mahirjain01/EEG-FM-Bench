import os
import os.path


PLATFORM = os.getenv("EEGFM_PLATFORM", "local")
DEFAULT_DATA_ROOT = os.getenv('EEGFM_DATA_ROOT', os.path.join('/data/nvme0', 'eegfmbench'))

PROJECT_ROOT = os.getenv("EEGFM_PROJECT_ROOT", "/data/nvme0/eegfmbench")
ASSETS_ROOT = os.path.join(PROJECT_ROOT, "assets")
RUN_ROOT = os.getenv("EEGFM_RUN_ROOT", os.path.join(DEFAULT_DATA_ROOT, "run"))
LOG_ROOT = os.getenv("EEGFM_LOG_ROOT", os.path.join(RUN_ROOT, "log"))
CONF_ROOT = os.getenv("EEGFM_CONF_ROOT", os.path.join(ASSETS_ROOT, "conf"))

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
