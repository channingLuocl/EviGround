"""Load the current EviGround environment configuration."""

from dotenv import load_dotenv
from eviground.paths import ROOT


def load_environment():
    load_dotenv(ROOT / ".env", override=False)
