"""Tests for the project milestones."""
import json

import pytest

from app.core import milestone_1


def test_milestone_1(tmp_path):
    splits = milestone_1(tmp_path / "first")
    repeated = milestone_1(tmp_path / "second")

    assert splits == repeated
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 360,
        "validation": 60,
        "test": 60,
    }

    events = {
        name: {row["event"] for row in rows}
        for name, rows in splits.items()
    }
    assert events["train"].isdisjoint(events["validation"])
    assert events["train"].isdisjoint(events["test"])
    assert events["validation"].isdisjoint(events["test"])

    for name, rows in splits.items():
        path = tmp_path / "first" / f"{name}.jsonl"
        loaded = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert loaded == rows

        for row in rows:
            score = f"{row['home_score']}-{row['away_score']}"
            assert score in row["event"]
            assert score in row["commentary"]
            assert row["home"] in row["commentary"]
            assert row["away"] in row["commentary"]

            if row["event_type"] == "goal":
                assert row["team"] in row["commentary"]
                scoring_side = (
                    "home_score" if row["team"] == row["home"]
                    else "away_score"
                )
                assert row[scoring_side] > 0


@pytest.mark.skip(reason="start milestone 2")
def test_milestone_2():
    raise NotImplementedError


@pytest.mark.skip(reason="start milestone 3")
def test_milestone_3():
    raise NotImplementedError


@pytest.mark.skip(reason="start milestone 4")
def test_milestone_4():
    raise NotImplementedError


@pytest.mark.skip(reason="start milestone 5")
def test_milestone_5():
    raise NotImplementedError