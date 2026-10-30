import re

import llm_finetune_evaluator


def test_version_is_semantic() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", llm_finetune_evaluator.__version__)
