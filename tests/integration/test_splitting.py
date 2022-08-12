import sys
from pathlib import Path
import pytest 

from git import Repo

sys.path.append(
    Path(__file__).parents[2]
    / "orchestration_pipeline"
    / "script_processing_container"
    / "src"
)

from parsers import create_parsers
from split_code import (
    GoSplitter,
    JavascriptSplitter,
    PythonSplitter,
    TypescriptSplitter,
)


@pytest.mark.parametrize(
    "language,codebase,splitter",
    (
        ("python", "", PythonSplitter),
        ("javascript", "", JavascriptSplitter),
        ("javascript", "", TypescriptSplitter),
        ("go", "", GoSplitter),
    ),
)
def test_splits_single_language_codebase(tmp_path, language, codebase, splitter):
    Repo.clone_from(
        codebase,
        lang_path,
        multi_options=["--depth=1"],
    )

    parser_path = str(tmp_path / "language-grammar.so")
    create_parsers(parser_path)

    lang_repo = ""  # TODO
    lang_path = str(tmp_path / "testcodebase")

    splitter = splitter(lang_path, parser_path)
    codebase_funcs = splitter.split_codebase()

    assert len(codebase_funcs) > 20

    for func in codebase_funcs:
        assert func["language"] == language
        assert func["start_line"] >= 0
        assert func["end_line"] >= 0
        assert func["start_char"] >= 0
        assert func["end_char"] >= 0
