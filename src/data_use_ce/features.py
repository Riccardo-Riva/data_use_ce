"""Feature groups, defined in configs/feature_groups.yaml."""

from collections.abc import Iterable
from pathlib import Path

import yaml

from data_use_ce.data import FINAL_GRADE, REPO_ROOT, TARGET

DEFAULT_GROUPS_FILE = REPO_ROOT / "configs" / "feature_groups.yaml"


def load_feature_groups(columns: Iterable[str], path: Path | None = None) -> dict[str, list[str]]:
    """Reads the feature groups and checks them against the data columns.

    Args:
        columns: Column names of the students table; every feature must be one of them.
        path: YAML file mapping each group name to a list of columns. Defaults to
            configs/feature_groups.yaml in the repository.

    Returns:
        Group name to list of column names, in file order.

    Raises:
        ValueError: If a group is empty, a column is unknown or appears in two groups,
            or a group contains the final grade or the excellence target.
    """
    known = set(columns)
    with open(path or DEFAULT_GROUPS_FILE, encoding="utf-8") as f:
        groups = yaml.safe_load(f)

    owner: dict[str, str] = {}
    for name, features in groups.items():
        if not features:
            raise ValueError(f"Feature group '{name}' is empty")
        for column in features:
            if column in (FINAL_GRADE, TARGET):
                raise ValueError(f"'{column}' is the outcome, not a feature (group '{name}')")
            if column not in known:
                raise ValueError(f"Unknown column '{column}' in group '{name}'")
            if column in owner:
                raise ValueError(f"Column '{column}' is in both '{owner[column]}' and '{name}'")
            owner[column] = name
    return groups
