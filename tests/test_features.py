"""Tests for the feature-group config."""

import pytest

from data_use_ce.features import load_feature_groups

# The 33 columns of the UCI files, in file order.
STUDENT_COLUMNS = (
    "school sex age address famsize Pstatus Medu Fedu Mjob Fjob reason guardian traveltime "
    "studytime failures schoolsup famsup paid activities nursery higher internet romantic "
    "famrel freetime goout Dalc Walc health absences G1 G2 G3"
).split()


def test_shipped_groups_use_every_predictor_exactly_once():
    groups = load_feature_groups(STUDENT_COLUMNS)
    assert list(groups) == [
        "prior_grades",
        "school",
        "family_socioeconomic",
        "demographic",
        "lifestyle_health",
    ]
    listed = [column for features in groups.values() for column in features]
    assert sorted(listed) == sorted(c for c in STUDENT_COLUMNS if c != "G3")


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("a: [G1, nope]", "Unknown column 'nope' in group 'a'"),
        ("a: [G1, G2]\nb: [G2]", "Column 'G2' is in both 'a' and 'b'"),
        ("a: []", "Feature group 'a' is empty"),
        ("a: [G1, G3]", "'G3' is the outcome"),
        ("a: [Excellent]", "'Excellent' is the outcome"),
    ],
)
def test_invalid_group_files_are_rejected(tmp_path, yaml_text, message):
    path = tmp_path / "groups.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_feature_groups(STUDENT_COLUMNS + ["Excellent"], path=path)
