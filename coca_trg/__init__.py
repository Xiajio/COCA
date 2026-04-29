"""COCA-inspired TRG binary classification utilities."""

from .metadata import ManifestRow, build_manifest
from .models import COCAForTRG

__all__ = ["COCAForTRG", "ManifestRow", "build_manifest"]
