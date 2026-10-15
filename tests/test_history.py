"""Guards on the recorded training history.

The point of these is that the history file is the evidence behind the training
curve. If it ever stops being real data, these fail.
"""

import json

from src import config as config_module


def load_history():
    cfg = config_module.load()
    return json.loads(cfg.path("history").read_text(encoding="utf-8"))


def test_history_exists_and_is_ordered():
    history = load_history()
    assert history, "training history is empty"
    assert [h["epoch"] for h in history] == list(range(1, len(history) + 1))


def test_metrics_are_in_range():
    for row in load_history():
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["val_accuracy"] <= 1.0
        assert row["loss"] >= 0
        assert row["val_loss"] >= 0
        assert row["learning_rate"] > 0


def test_learning_rate_only_decreases():
    """ReduceLROnPlateau can lower the rate but nothing should raise it."""
    rates = [row["learning_rate"] for row in load_history()]
    assert all(after <= before for before, after in zip(rates, rates[1:], strict=False))
