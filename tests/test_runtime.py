from runtime import run_metadata


def test_run_metadata_records_dependencies_and_git_state() -> None:
    metadata = run_metadata()

    assert metadata["python_version"]
    assert metadata["dependencies"]["torch"] is not None
    assert {"commit", "branch", "is_dirty", "has_commit"}.issubset(metadata["git"])
