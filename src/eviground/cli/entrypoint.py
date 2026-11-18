"""Default command for the predicted-category visual ReAct pipeline."""

import os
from eviground.config import load_environment


def configure():
    load_environment()
    os.environ.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    defaults = {
        "EVIGROUND_SYNTAX_BACKEND": "spacy",
        "EVIGROUND_API_PROVIDER": "openai",
        "EVIGROUND_MODEL": "qwen2-vl-72b",
        "EVIGROUND_BASE_URL": "http://127.0.0.1:18000/v1",
        "EVIGROUND_API_KEY": "EMPTY",
        "EVIGROUND_ENABLE_THINKING": "0",
        "EVIGROUND_THINKING_TOKEN_FLOOR": "8192",
        "EVIGROUND_OPENAI_TOKEN_PARAMETER": "max_tokens",
        "EVIGROUND_OPENAI_TEMPERATURE": "0",
        "EVIGROUND_ALLOW_HTTP": "1",
        "EVIGROUND_CONTEXT_LIMIT": "131072",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main():
    configure()
    from eviground.cli.ground import main as run

    run()


if __name__ == "__main__":
    main()
