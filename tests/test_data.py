"""Tests for data loading and the excellence target, on a toy CSV."""

import pandas as pd
import pytest

from data_use_ce.data import TARGET, add_excellence, load_students

# Semicolon-separated, with quoted numbers in some rows like the original files.
TOY_CSV = """school;sex;G1;G2;G3
"GP";"F";"5";"6";6
"GP";"M";10;12;14
"MS";"F";15;0;0
"MS";"M";16;17;18
"""


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "student-mat.csv").write_text(TOY_CSV, encoding="utf-8")
    return tmp_path


def test_load_students_reads_semicolons_and_quoted_numbers(data_dir):
    df = load_students("mat", drop_zero_final=False, data_dir=data_dir)
    assert list(df.columns) == ["school", "sex", "G1", "G2", "G3"]
    assert len(df) == 4
    assert pd.api.types.is_integer_dtype(df["G1"])


def test_load_students_drops_zero_final_grades_by_default(data_dir):
    df = load_students("mat", data_dir=data_dir)
    assert df["G3"].tolist() == [6, 14, 18]
    assert df.index.tolist() == [0, 1, 2]


def test_load_students_missing_file_points_to_download_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_students("por", data_dir=tmp_path)


def test_add_excellence_counts_the_threshold_grade_as_excellent():
    df = pd.DataFrame({"G3": [13, 14, 15]})
    out = add_excellence(df, 14)
    assert out[TARGET].tolist() == [False, True, True]
    assert TARGET not in df.columns


@pytest.mark.parametrize("threshold", [0, 21])
def test_add_excellence_rejects_thresholds_outside_the_grade_scale(threshold):
    with pytest.raises(ValueError, match="between 1 and 20"):
        add_excellence(pd.DataFrame({"G3": [10]}), threshold)
