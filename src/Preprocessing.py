"""Compatibility package path for old notebooks; prefer lowercase preprocessing.

A module shim avoids two case-only directory names on Windows.
"""

from preprocessing import __path__ as _canonical_path

__path__ = _canonical_path
