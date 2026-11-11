"""Tests for the dataset download script, on a toy nested archive (no network)."""

import io
import zipfile

import download_data as dl
import pytest


def _nested_archive(members: dict[str, bytes]) -> bytes:
    """Build an outer zip holding `student.zip` plus a stale copy, like the UCI archive."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr(dl.OUTER_MEMBER, inner.getvalue())
        zf.writestr(".student.zip_old", b"stale copy, must be ignored")
    return outer.getvalue()


@pytest.fixture
def toy_dataset(monkeypatch, tmp_path):
    """Point the script at a temporary output folder and toy checksums."""
    contents = {"student-mat.csv": b"a;b\n1;2\n", "student.txt": b"docs"}
    expected = {name: dl.sha256_of(data) for name, data in contents.items()}
    monkeypatch.setattr(dl, "DATASET_DIR", tmp_path / "out")
    monkeypatch.setattr(dl, "EXPECTED_SHA256", expected)
    archive = _nested_archive({**contents, "unlisted.txt": b"not extracted"})
    return archive, expected, tmp_path / "out"


def test_extracts_only_listed_members(toy_dataset):
    archive, _, out_dir = toy_dataset
    dl.extract(archive)
    assert sorted(p.name for p in out_dir.iterdir()) == ["student-mat.csv", "student.txt"]
    assert (out_dir / "student-mat.csv").read_bytes() == b"a;b\n1;2\n"


def test_checksum_mismatch_stops_the_script(monkeypatch, toy_dataset):
    archive, expected, _ = toy_dataset
    monkeypatch.setattr(dl, "EXPECTED_SHA256", {**expected, "student.txt": "0" * 64})
    with pytest.raises(SystemExit, match="Checksum mismatch for student.txt"):
        dl.extract(archive)


def test_files_present_and_valid_detects_corruption(toy_dataset):
    archive, _, out_dir = toy_dataset
    assert not dl.files_present_and_valid()
    dl.extract(archive)
    assert dl.files_present_and_valid()
    (out_dir / "student.txt").write_bytes(b"tampered")
    assert not dl.files_present_and_valid()
