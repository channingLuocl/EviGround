"""Default command for the frozen predicted-category Top-5 r18 pipeline."""
import os

from eviground.config import load_environment


def configure_r18():
    load_environment()
    os.environ.update({
        'EVIGROUND_V4': '1', 'EVIGROUND_V7': '1', 'EVIGROUND_V7_4': '1',
        'EVIGROUND_V9': '1', 'EVIGROUND_V7_2': '0', 'EVIGROUND_V7_3': '0',
        'EVIGROUND_V9_SOURCE': 'r8', 'EVIGROUND_R8_GROUNDING_REVISION': 'r18',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
    })
    defaults = {
        'EVIGROUND_SYNTAX_BACKEND': 'spacy',
        'EVIGROUND_API_PROVIDER': 'openai', 'EVIGROUND_REACT_BACKEND': 'openai',
        'EVIGROUND_MODEL': 'qwen3.7-plus', 'EVIGROUND_JEV_MODEL': '',
        'EVIGROUND_BASE_URL': 'https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1',
        'EVIGROUND_ENABLE_THINKING': '1', 'EVIGROUND_THINKING_TOKEN_FLOOR': '8192',
        'EVIGROUND_OPENAI_TOKEN_PARAMETER': 'max_completion_tokens',
        'EVIGROUND_OPENAI_TEMPERATURE': '0',
        'EVIGROUND_ALLOW_HTTP': '1', 'EVIGROUND_CONTEXT_LIMIT': '131072',
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main():
    configure_r18()
    from eviground.cli.run_grounding_v9 import main as run
    run()


if __name__ == '__main__':
    main()
