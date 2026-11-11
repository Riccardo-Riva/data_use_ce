"""Loading of the UCI Student Performance data and the excellence target."""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "student-performance"

FINAL_GRADE = "G3"
MAX_GRADE = 20
TARGET = "Excellent"


def load_students(
    dataset: str = "mat", drop_zero_final: bool = True, data_dir: Path | None = None
) -> pd.DataFrame:
    """Reads one of the two student files.

    Args:
        dataset: "mat" for the maths course or "por" for the Portuguese course.
        drop_zero_final: Drop students whose final grade is 0. They probably did not sit
            the final exam, and all of them have zero absences, which would look like a
            real pattern in the school features.
        data_dir: Folder with the CSV files. Defaults to data/raw/student-performance
            in the repository.

    Returns:
        One row per student, with the original UCI columns.

    Raises:
        FileNotFoundError: If the CSV file is missing.
    """
    path = (data_dir or DEFAULT_DATA_DIR) / f"student-{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python scripts/download_data.py")
    df = pd.read_csv(path, sep=";")
    if drop_zero_final:
        df = df[df[FINAL_GRADE] > 0].reset_index(drop=True)
    return df


def add_excellence(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Adds the boolean column "Excellent", true when the final grade is >= threshold.

    Args:
        df: Students, with a "G3" column.
        threshold: Smallest final grade that counts as excellent, from 1 to 20.

    Returns:
        A copy of `df` with the new column; `df` is not modified.

    Raises:
        ValueError: If the threshold is outside the grade scale.
    """
    if not 1 <= threshold <= MAX_GRADE:
        raise ValueError(f"threshold must be between 1 and {MAX_GRADE}, got {threshold}")
    out = df.copy()
    out[TARGET] = out[FINAL_GRADE] >= threshold
    return out
