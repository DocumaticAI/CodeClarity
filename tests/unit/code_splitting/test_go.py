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
func testfunc(x int, y int) int {
	return x + y
}


func func2(x int, y int) int {
	return x + y
}


""".lstrip()

    splitter = split_code.GoSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert len(objects) == 2

    assert objects[0]["language"] == "go"
    assert objects[0]["name"] == "testfunc"
    assert objects[0]["start_line"] == 1
    assert objects[0]["end_line"] == 3
    assert objects[0]["start_char"] == 0
    assert objects[0]["end_char"] == 1
    assert objects[0]["filepath"] == file_path

    assert objects[1]["language"] == "go"
    assert objects[1]["name"] == "func2"
    assert objects[1]["start_line"] == 6
    assert objects[1]["end_line"] == 8
    assert objects[1]["start_char"] == 0
    assert objects[1]["end_char"] == 1
    assert objects[1]["filepath"] == file_path


def test_splits_functions_with_shared_type(tmp_path):
    file_path = str(tmp_path / "file1.js")
    file_contents = """
func func1(x, y int) int {
	return x + y
}
"""

    splitter = split_code.GoSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert len(objects) == 1

    assert objects[0]["language"] == "go"
    assert objects[0]["name"] == "func1"
    assert objects[0]["start_line"] == 2
    assert objects[0]["end_line"] == 4
    assert objects[0]["start_char"] == 0
    assert objects[0]["end_char"] == 1
    assert objects[0]["filepath"] == file_path


def test_does_not_split_free_code(tmp_path):
    file_path = str(tmp_path / "file1.js")
    file_contents = """
package main

import "fmt"

var c, python, java bool

"""

    splitter = split_code.GoSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert len(objects) == 0


def test_does_not_split_structs(tmp_path):
    file_path = str(tmp_path / "file1.js")
    file_contents = """
type Vertex struct {
	X int
	Y int
}
"""

    splitter = split_code.GoSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert len(objects) == 0


@pytest.mark.parametrize("file_path", ("test.ts", "test.go", "some/path/to/file.go"))
def test_adds_filepath_to_object_dict(tmp_path, file_path):
    file_contents = """
func func1(x, y int) int {
	return x + y
}
"""

    splitter = split_code.GoSplitter(tmp_path, parser_path)
    parser = splitter.get_parser()
    objects = splitter.split_file(file_path, file_contents, parser)

    assert objects[0]["filepath"] == file_path
