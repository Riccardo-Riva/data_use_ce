"""Download the UCI Student Performance dataset into data/raw/student-performance/.

Usage (from the repository root):

    uv run python scripts/download_data.py            # download only if files are missing
    uv run python scripts/download_data.py --force    # download and extract again

Source: https://archive.ics.uci.edu/dataset/320/student+performance
Citation: Cortez, P. (2008). Student Performance. UCI Machine Learning Repository.
          https://doi.org/10.24432/C5TG7T
          Cortez, P. & Silva, A. M. G. (2008). Using data mining to predict secondary
          school student performance. Proc. 5th FUBUTEC.

The official archive is a zip that contains an inner zip (student.zip) and a stale copy
of the data (.student.zip_old). The stale copy is ignored. Only the members listed in
EXPECTED_SHA256 are extracted, and their SHA-256 hashes are checked.

Only the standard library is used, so the script runs before any dependency is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
OUTER_MEMBER = "student.zip"  # the current inner archive (not .student.zip_old)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATASET_DIR = RAW_DIR / "student-performance"
ARCHIVE_PATH = RAW_DIR / "student+performance.zip"

# Hashes of the extracted files as published in the 2023 archive.
# If UCI changes a file, the script stops and reports the mismatch: check the new
# file by hand before updating these values.
EXPECTED_SHA256 = {
    "student-mat.csv": "e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80",
    "student-por.csv": "a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f",
    "student.txt": "f8d3e734e237071312790ca667330c66d75e706317f8d5e2125c479d70e962c1",
    "student-merge.R": "a04f4ef19551c319428f642269c125faa21c521045946beead1c32e656269d31",
}


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_archive() -> bytes:
    """Return the outer archive bytes, reusing the copy in data/raw/ when present."""
    if ARCHIVE_PATH.exists():
        print(f"Using existing archive: {ARCHIVE_PATH}")
        return ARCHIVE_PATH.read_bytes()
    print(f"Downloading {ARCHIVE_URL}")
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "data-use-ce"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ARCHIVE_PATH.with_suffix(".zip.part")
    tmp.write_bytes(data)
    tmp.replace(ARCHIVE_PATH)  # atomic: a failed download never leaves a partial archive
    return data


def extract(outer_bytes: bytes) -> None:
    """Extract the files listed in EXPECTED_SHA256 from the nested archive."""
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer:
        inner_bytes = outer.read(OUTER_MEMBER)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        for name, expected in EXPECTED_SHA256.items():
            # Only names from the allowlist are written, so archive paths cannot escape
            # DATASET_DIR.
            data = inner.read(name)
            actual = sha256_of(data)
            if actual != expected:
                sys.exit(
                    f"Checksum mismatch for {name}: expected {expected}, got {actual}.\n"
                    "UCI may have changed the archive. Inspect the file before updating "
                    "EXPECTED_SHA256 in scripts/download_data.py."
                )
            (DATASET_DIR / name).write_bytes(data)
            print(f"  extracted {DATASET_DIR / name}")


def files_present_and_valid() -> bool:
    for name, expected in EXPECTED_SHA256.items():
        path = DATASET_DIR / name
        if not path.exists() or sha256_of(path.read_bytes()) != expected:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="download and extract again")
    args = parser.parse_args()

    if not args.force and files_present_and_valid():
        print(f"Dataset already present and verified: {DATASET_DIR}")
        return

    extract(fetch_archive())

    for name in ("student-mat.csv", "student-por.csv"):
        # Count rows without loading pandas: header line is not a student.
        n_rows = sum(1 for _ in open(DATASET_DIR / name, encoding="utf-8")) - 1
        print(f"{name}: {n_rows} students")
    print("Done.")


if __name__ == "__main__":
    main()
