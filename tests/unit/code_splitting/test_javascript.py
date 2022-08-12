import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).parents[3]
        / "orchestration_pipeline"
        / "script_processing_container"
        / "src"
    ),
)

import parsers
import split_code

parser_path = Path(__file__).parent / "language-grammar.so"
if not parser_path.exists():
    parsers.create_parsers(parser_path)


def test_splits_functions_in_a_file(tmp_path):
    file_path = str(tmp_path / "file1.js")
    file_contents = """
function testfunc() {
    console.log("nothing");
}


function func2(a, b, c) {
    console.log(a, b, c);
}


""".lstrip()

    splitter = split_code.JavascriptSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert len(objects) == 2

    assert objects[0]["language"] == "javascript"
    assert objects[0]["name"] == "testfunc"
    assert objects[0]["start_line"] == 1
    assert objects[0]["end_line"] == 3
    assert objects[0]["start_char"] == 0
    assert objects[0]["end_char"] == 1
    assert objects[0]["filepath"] == file_path

    assert objects[1]["language"] == "javascript"
    assert objects[1]["name"] == "func2"
    assert objects[1]["start_line"] == 6
    assert objects[1]["end_line"] == 8
    assert objects[1]["start_char"] == 0
    assert objects[1]["end_char"] == 1
    assert objects[1]["filepath"] == file_path


def test_splits_single_line_function(tmp_path):
    file_path = str(tmp_path / "file1.js")
    file_contents = "function somefunc() {}"

    splitter = split_code.JavascriptSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert len(objects) == 1

    assert objects[0]["language"] == "javascript"
    assert objects[0]["name"] == "somefunc"
    assert objects[0]["start_line"] == 1
    assert objects[0]["end_line"] == 1
    assert objects[0]["start_char"] == 0
    assert objects[0]["end_char"] == 22


def test_splits_es6_funcs(tmp_path):
    file_path = str(tmp_path / "file1.js")
    file_contents = "const somefunc = () => {}"

    splitter = split_code.JavascriptSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert len(objects) == 1

    assert objects[0]["language"] == "javascript"
    assert objects[0]["name"] == "somefunc"
    assert objects[0]["start_line"] == 1
    assert objects[0]["end_line"] == 1
    assert objects[0]["start_char"] == 0


def test_splits_es6_func_with_a_single_variable(tmp_path):
    file_path = str(tmp_path / "file1.js")
    file_contents = """
const somefunc = a => {
    console.log(a);
}
"""

    splitter = split_code.JavascriptSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert len(objects) == 1

    assert objects[0]["language"] == "javascript"
    assert objects[0]["name"] == "somefunc"
    assert objects[0]["start_line"] == 2
    assert objects[0]["end_line"] == 4


def test_splits_es6_func_with_multiple_variables(tmp_path):
    file_path = str(tmp_path / "file1.js")
    file_contents = """
const somefunc = (a, b, c) => {
    console.log(a, b, c);
}
"""

    splitter = split_code.JavascriptSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert len(objects) == 1

    assert objects[0]["language"] == "javascript"
    assert objects[0]["name"] == "somefunc"
    assert objects[0]["start_line"] == 2
    assert objects[0]["end_line"] == 4


def test_splits_es6_func_with_deconstructed_variables(tmp_path):
    file_path = str(tmp_path / "file1.js")
    file_contents = """
const somefunc = ({a, b}) => {
    console.log(a, b);
}
"""

    splitter = split_code.JavascriptSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert len(objects) == 1

    assert objects[0]["language"] == "javascript"
    assert objects[0]["name"] == "somefunc"
    assert objects[0]["start_line"] == 2
    assert objects[0]["end_line"] == 4


def test_does_not_split_free_code(tmp_path):
    file_path = str(tmp_path / "file1.js")
    file_contents = """
const x = 1
var y = 2
console.log(x + y)
const z = x ** y
"""

    splitter = split_code.JavascriptSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert len(objects) == 0


@pytest.mark.parametrize("file_path", ("test.py", "test.js", "some/path/to/file.js"))
def test_adds_filepath_to_object_dict(tmp_path, file_path):
    file_contents = """
const somefunc = ({a, b}) => {
    console.log(a, b);
}
"""

    splitter = split_code.JavascriptSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert objects[0]["filepath"] == file_path
