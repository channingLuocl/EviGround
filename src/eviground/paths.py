"""Repository resources and source identities after the src-layout migration."""
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = next((p for p in PACKAGE_ROOT.parents if (p / 'pyproject.toml').is_file()), Path.cwd())
ROOT = Path(os.environ.get('EVIGROUND_ROOT', os.environ.get('SEEGROUND_ROOT', ROOT))).expanduser().resolve()
VALIDATION_ROOT = ROOT / 'data' / 'validation'


def source_file(name):
    if name == 'parts_v83.json':
        return ROOT / 'data' / name
    if name.startswith('test_'):
        return ROOT / 'tests' / 'legacy' / name
    if name.startswith('category_v10/'):
        return PACKAGE_ROOT / 'classifiers' / name
    layout = json.loads((PACKAGE_ROOT / '_module_layout.json').read_text())
    return PACKAGE_ROOT / layout[name]


def source_files():
    return sorted(p for p in PACKAGE_ROOT.rglob('*.py') if not p.name.startswith('._'))
