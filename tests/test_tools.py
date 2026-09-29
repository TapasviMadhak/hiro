"""Tests for Hiro built-in tools."""
import pytest
from pathlib import Path
from hiro.tools import BuiltinToolExecutor, get_builtin_tools


def test_builtin_tool_definitions():
    tools = get_builtin_tools()
    names = {t.name for t in tools}
    assert "read_file" in names
    assert "write_file" in names
    assert "edit_file" in names
    assert "list_dir" in names
    assert "search_files" in names
    assert "bash" in names
    assert "web_fetch" in names


@pytest.mark.asyncio
async def test_file_operations(tmp_path):
    executor = BuiltinToolExecutor(cwd=str(tmp_path))

    # 1. write_file
    test_file = tmp_path / "hello.py"
    write_res = await executor.execute("write_file", {
        "path": "hello.py",
        "content": "def hello():\n    print('world')\n",
    })
    assert "Written" in write_res
    assert test_file.exists()

    # 2. read_file
    read_res = await executor.execute("read_file", {"path": "hello.py"})
    assert "def hello():" in read_res

    # 3. edit_file
    edit_res = await executor.execute("edit_file", {
        "path": "hello.py",
        "old_string": "print('world')",
        "new_string": "print('hiro')",
    })
    assert "Edited" in edit_res
    assert "print('hiro')" in test_file.read_text()

    # 4. list_dir
    list_res = await executor.execute("list_dir", {"path": "."})
    assert "hello.py" in list_res

    # 5. search_files
    search_res = await executor.execute("search_files", {"pattern": "hiro", "path": "."})
    assert "hello.py" in search_res

    # 6. move_file
    move_res = await executor.execute("move_file", {"source": "hello.py", "destination": "greet.py"})
    assert "Moved" in move_res
    assert not test_file.exists()
    assert (tmp_path / "greet.py").exists()

    # 7. delete_file
    del_res = await executor.execute("delete_file", {"path": "greet.py"})
    assert "Deleted" in del_res
    assert not (tmp_path / "greet.py").exists()


@pytest.mark.asyncio
async def test_edit_file_validation(tmp_path):
    executor = BuiltinToolExecutor(cwd=str(tmp_path))
    test_file = tmp_path / "dup.txt"
    test_file.write_text("foo\nfoo\nbar\n")

    # Missing pattern
    res_missing = await executor.execute("edit_file", {
        "path": "dup.txt",
        "old_string": "baz",
        "new_string": "qux",
    })
    assert "Error: Pattern not found" in res_missing

    # Ambiguous pattern
    res_ambiguous = await executor.execute("edit_file", {
        "path": "dup.txt",
        "old_string": "foo",
        "new_string": "qux",
    })
    assert "Error: Pattern found 2 times" in res_ambiguous


@pytest.mark.asyncio
async def test_bash_execution(tmp_path):
    executor = BuiltinToolExecutor(cwd=str(tmp_path))
    res = await executor.execute("bash", {"command": "echo test_bash_output"})
    assert "test_bash_output" in res
