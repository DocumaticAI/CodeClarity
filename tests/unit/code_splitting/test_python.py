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
    file_path = str(tmp_path / "file1.py")
    file_contents = """
def testfunc():
    # this has worked
    a = None


def func2(
    a,
    b,
    c
):pass


""".lstrip()

    splitter = split_code.PythonSplitter(tmp_path, parser_path)
    objects = splitter.split_file(file_path, file_contents)

    assert len(objects) == 2

    assert objects[0]["language"] == "python"
    assert objects[0]["name"] == "testfunc"
    assert objects[0]["start_line"] == 1
    assert objects[0]["end_line"] == 3
    assert objects[0]["start_char"] == 0
    assert objects[0]["end_char"] == 12
    assert objects[0]["filepath"] == file_path

    assert objects[1]["language"] == "python"
    assert objects[1]["name"] == "func2"
    assert objects[1]["start_line"] == 6
    assert objects[1]["end_line"] == 10
    assert objects[1]["start_char"] == 0
    assert objects[1]["end_char"] == 6
    assert objects[0]["filepath"] == file_path


def test_splits_single_line_function(tmp_path):
    file_path = str(tmp_path / "file1.py")
    file_contents = "def somefunc():pass"

    splitter = split_code.PythonSplitter(tmp_path, parser_path)
    objects = splitter.split_file(file_path, file_contents)

    assert len(objects) == 1

    assert objects[0]["language"] == "python"
    assert objects[0]["name"] == "somefunc"
    assert objects[0]["start_line"] == 1
    assert objects[0]["end_line"] == 1
    assert objects[0]["start_char"] == 0
    assert objects[0]["end_char"] == 19


def test_does_not_split_classes(tmp_path):
    file_path = str(tmp_path / "file1.py")
    file_contents = """
class AClass:
    def __init__(self):pass

    def do_something(a, b):
        print(a)
        print(b)
"""

    splitter = split_code.PythonSplitter(tmp_path, parser_path)
    objects = splitter.split_file(file_path, file_contents)

    assert len(objects) == 0


def test_does_not_split_lambdas(tmp_path):
    file_path = str(tmp_path / "file1.py")
    file_contents = "lambda x: x + 1"
    splitter = split_code.PythonSplitter(tmp_path, parser_path)
    objects = splitter.split_file(file_path, file_contents)

    assert len(objects) == 0


def test_does_not_split_free_code(tmp_path):
    file_path = str(tmp_path / "file1.py")
    file_contents = """
x = 1
y = 2
print(x + y)
z = x ** y
"""

    splitter = split_code.PythonSplitter(tmp_path, parser_path)
    objects = splitter.split_file(file_path, file_contents)

    assert len(objects) == 0


@pytest.mark.parametrize("file_path", ("test.py", "test.js", "some/path/to/file.py"))
def test_adds_filepath_to_object_dict(tmp_path, file_path):
    file_contents = "def somefunc():pass"

    splitter = split_code.PythonSplitter(tmp_path, parser_path)
    objects = splitter.split_file(file_path, file_contents)

    assert objects[0]["filepath"] == file_path
