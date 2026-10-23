"""Default command for the frozen predicted-category Top-5 r18 pipeline."""
import os

from dotenv import load_dotenv

from seeground.paths import ROOT


def configure_r18():
    load_dotenv(ROOT / '.env', override=False)
    os.environ.update({
        'SEEGROUND_V4': '1', 'SEEGROUND_V7': '1', 'SEEGROUND_V7_4': '1',
        'SEEGROUND_V9': '1', 'SEEGROUND_V7_2': '0', 'SEEGROUND_V7_3': '0',
        'SEEGROUND_V9_SOURCE': 'r8', 'SEEGROUND_R8_GROUNDING_REVISION': 'r18',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
    })
    defaults = {
        'SEEGROUND_SYNTAX_BACKEND': 'spacy',
        'SEEGROUND_API_PROVIDER': 'openai', 'SEEGROUND_REACT_BACKEND': 'openai',
        'SEEGROUND_MODEL': 'qwen3.7-plus', 'SEEGROUND_JEV_MODEL': '',
        'SEEGROUND_BASE_URL': 'https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1',
        'SEEGROUND_ENABLE_THINKING': '1', 'SEEGROUND_THINKING_TOKEN_FLOOR': '8192',
        'SEEGROUND_OPENAI_TOKEN_PARAMETER': 'max_completion_tokens',
        'SEEGROUND_OPENAI_TEMPERATURE': '0',
        'SEEGROUND_ALLOW_HTTP': '1', 'SEEGROUND_CONTEXT_LIMIT': '131072',
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main():
    configure_r18()
    from seeground.cli.run_grounding_v9 import main as run
    run()


if __name__ == '__main__':
    main()
