"""Atomic result writes and protection for frozen experiment directories."""

import json
from pathlib import Path


def ensure_run_writable(output):
    """Reject a frozen run, including nested attempts and symlink aliases."""
    output = Path(output).resolve()
    for directory in (output, *output.parents):
        marker = directory / "FROZEN.json"
        if marker.is_file():
            raise ValueError(
                f"Frozen run is read-only: {directory}. Choose a fresh --output / EVIGROUND_RUN_DIR; the frozen run is read-only."
            )


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
