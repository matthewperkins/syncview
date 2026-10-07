"""Where syncview keeps its caches."""
import os
import sys
from pathlib import Path


def platform_cache_root():
    """The platform's per-user cache folder for syncview."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "syncview"


def default_cache_root():
    """$SYNCVIEW_CACHE if set, else platform_cache_root()."""
    env = os.environ.get("SYNCVIEW_CACHE")
    return Path(env).expanduser() if env else platform_cache_root()
