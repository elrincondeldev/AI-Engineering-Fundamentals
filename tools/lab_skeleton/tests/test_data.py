from m00_skeleton.data import datasets_dir


def test_datasets_dir_is_found() -> None:
    assert (datasets_dir() / "generate.py").is_file()
