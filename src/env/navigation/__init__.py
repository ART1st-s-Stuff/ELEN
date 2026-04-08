"""Navigation environment utilities."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["convert_eb_nav_json_to_hdf5", "download_eb_nav_repo"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = importlib.import_module(".dataset", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
