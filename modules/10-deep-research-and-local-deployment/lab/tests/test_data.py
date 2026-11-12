from m10_deep_research.data import datasets_dir


def test_datasets_dir_is_found() -> None:
    assert (datasets_dir() / "generate.py").is_file()
