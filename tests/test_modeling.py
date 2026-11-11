"""Tests for preprocessing and the train/test split, on a toy table."""

import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from data_use_ce.data import TARGET
from data_use_ce.modeling import make_preprocessor, split_students

FEATURES = ["address", "job", "studytime"]


@pytest.fixture
def students():
    return pd.DataFrame(
        {
            "address": ["U", "R"] * 10,  # two values: a single 0/1 column
            "job": ["a", "b", "c", "d"] * 5,  # four values: four columns
            "studytime": [1, 2, 3, 4, 1] * 4,
            TARGET: [True] * 5 + [False] * 15,
        }
    )


def test_preprocessor_encodes_text_and_keeps_numbers(students):
    out = make_preprocessor(students, FEATURES).fit_transform(students)
    assert list(out.columns) == ["address_U", "job_a", "job_b", "job_c", "job_d", "studytime"]
    assert out["address_U"].tolist() == [1, 0] * 10
    assert out["studytime"].tolist() == students["studytime"].tolist()


def test_preprocessor_works_when_every_column_is_numeric(students):
    out = make_preprocessor(students, ["studytime"]).fit_transform(students)
    assert list(out.columns) == ["studytime"]


def test_preprocessor_learns_categories_from_training_students_only(students):
    train = students[students["job"] != "d"]
    preprocessor = make_preprocessor(students, FEATURES).fit(train)
    with pytest.warns(UserWarning, match="unknown categories"):
        out = preprocessor.transform(students)
    assert "job_d" not in out.columns
    unseen = out.loc[students["job"] == "d", ["job_a", "job_b", "job_c"]]
    assert (unseen == 0).all().all()


def test_preprocessor_and_tree_fit_together_in_a_pipeline(students):
    model = make_pipeline(
        make_preprocessor(students, FEATURES), DecisionTreeClassifier(max_depth=2, random_state=0)
    )
    model.fit(students[FEATURES], students[TARGET])
    assert model.predict_proba(students[FEATURES]).shape == (20, 2)


def test_split_keeps_the_excellent_share_and_never_overlaps(students):
    X_train, X_test, y_train, y_test = split_students(students, FEATURES, test_size=0.4, seed=1)
    assert (len(X_train), len(X_test)) == (12, 8)
    assert (int(y_train.sum()), int(y_test.sum())) == (3, 2)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert list(X_train.columns) == FEATURES


def test_split_is_reproducible_for_the_same_seed(students):
    first = split_students(students, FEATURES, seed=5)[1]
    again = split_students(students, FEATURES, seed=5)[1]
    other = split_students(students, FEATURES, seed=6)[1]
    assert first.index.tolist() == again.index.tolist()
    assert first.index.tolist() != other.index.tolist()
