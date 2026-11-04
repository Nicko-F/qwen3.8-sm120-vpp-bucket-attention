"""Standard, configurable per-user build cache. No experiment directories."""

import os
from pathlib import Path


def cache_root():
    explicit = os.environ.get("QWEN_INFRA_CACHE_DIR")
    base = (
        Path(explicit).expanduser()
        if explicit
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "qwen-infra"
    )
    return base.resolve()


# Existing kernel builders derive content-addressed children from this root.
ROOT = cache_root()


def guard():
    """Prepare only package-owned cache storage; server admission is external."""
    ROOT.mkdir(parents=True, exist_ok=True)


def build_directory(component):
    path = cache_root() / component
    path.mkdir(parents=True, exist_ok=True)
    return path
