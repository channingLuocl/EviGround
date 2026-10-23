"""Package source identities and repository resources."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = next(
    (p for p in PACKAGE_ROOT.parents if (p / "pyproject.toml").is_file()), Path.cwd()
)
ROOT = Path(os.getenv("EVIGROUND_ROOT", ROOT)).expanduser().resolve()


def source_files():
    return sorted(
        (p for p in PACKAGE_ROOT.rglob("*.py") if not p.name.startswith("._"))
    )
