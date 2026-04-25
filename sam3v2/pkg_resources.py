"""Minimal pkg_resources compatibility shim.

This project only needs `resource_filename` to resolve packaged assets.
"""

from importlib import resources as importlib_resources
from pathlib import Path
from typing import Union


def resource_filename(package_or_requirement: Union[str, object], resource_name: str) -> str:
    """Return an absolute filesystem path for a package resource.

    Supports the subset used by SAM3 model builder.
    """
    package = (
        package_or_requirement
        if isinstance(package_or_requirement, str)
        else str(package_or_requirement)
    )
    return str(Path(importlib_resources.files(package)).joinpath(resource_name))


__all__ = ["resource_filename"]
