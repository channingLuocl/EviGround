"""Load EviGround settings while accepting existing local configuration files."""
import os

from dotenv import load_dotenv

from eviground.paths import ROOT


def load_environment():
    load_dotenv(ROOT / '.env', override=False)
    for name, value in tuple(os.environ.items()):
        if name.startswith('SEEGROUND_'):
            os.environ.setdefault('EVIGROUND_' + name[len('SEEGROUND_'):], value)
