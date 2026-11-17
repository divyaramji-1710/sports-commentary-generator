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


def test_milestone_2():
    import torch
    from torch import nn
    from transformers.pytorch_utils import Conv1D

    from app.lora import LoRALinear

    torch.manual_seed(0)

    # Test both supported weight layouts.
    for base in [nn.Linear(8, 12), Conv1D(12, 8)]:
        adapter = LoRALinear(
            base, r=2, alpha=4, dropout=0.0
        )
        x = torch.randn(3, 8)
        original_weight = base.weight.detach().clone()
        original_bias = base.bias.detach().clone()

        # Zero B means injection starts as a no-op.
        torch.testing.assert_close(adapter(x), base(x))

        assert not base.weight.requires_grad
        assert not base.bias.requires_grad
        assert adapter.A.requires_grad
        assert adapter.B.requires_grad
        assert adapter.A.numel() + adapter.B.numel() == 40

        optimizer = torch.optim.SGD(
            [adapter.A, adapter.B], lr=0.1
        )
        target = torch.randn(3, 12)
        loss = (adapter(x) - target).square().mean()
        loss.backward()

        assert base.weight.grad is None
        assert base.bias.grad is None
        assert adapter.B.grad.abs().sum() > 0

        optimizer.step()

        # Training changes the adapter, leaving base weights untouched.
        assert adapter.B.detach().abs().sum() > 0
        torch.testing.assert_close(
            base.weight, original_weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            base.bias, original_bias, rtol=0, atol=0
        )

        # Merging preserves the adapted output.
        adapter.eval()
        expected = adapter(x).detach()
        merged = adapter.merged()
        torch.testing.assert_close(
            merged(x), expected, rtol=1e-5, atol=1e-6
        )


def test_milestone_3(monkeypatch):
    from app.core import milestone_3

    calls = {}

    class FakeGenerator:
        def __init__(self, checkpoint):
            calls["checkpoint"] = checkpoint

        def generate(self, event, max_new_tokens):
            calls["event"] = event
            calls["max_new_tokens"] = max_new_tokens
            return "City score! That makes it City 2-1 Rovers."

    monkeypatch.setattr(
        "app.inference.CommentaryGenerator",
        FakeGenerator,
    )

    result = milestone_3(
        "Goal for City.",
        checkpoint="example.pt",
        max_new_tokens=32,
    )

    assert result == "City score! That makes it City 2-1 Rovers."
    assert calls == {
        "checkpoint": "example.pt",
        "event": "Goal for City.",
        "max_new_tokens": 32,
    }


def test_milestone_4():
    from app.evaluation import check_commentary

    row = {
        "event_type": "goal",
        "team": "City",
        "home": "City",
        "away": "Rovers",
        "home_score": 2,
        "away_score": 1,
    }

    correct = check_commentary(
        row, "City score! That makes it City 2-1 Rovers."
    )
    assert correct["both_correct"]
    assert correct["matches_valid_template"]

    wrong_score = check_commentary(
        row, "City score! That makes it City 1-2 Rovers."
    )
    assert not wrong_score["score_correct"]

    wrong_actor = check_commentary(
        row, "Rovers score! That makes it City 2-1 Rovers."
    )
    assert wrong_actor["score_correct"]
    assert not wrong_actor["event_and_actor_correct"]

    extra_score = check_commentary(
        row, "City score! City 2-1 Rovers. Now it is 3-1."
    )
    assert not extra_score["score_correct"]


def test_milestone_5():
    import torch
    from torch import nn

    from app.lora import LoRALinear
    from app.merging import merge_model

    torch.manual_seed(17)
    adapter = LoRALinear(
        nn.Linear(8, 8),
        r=2,
        alpha=4,
        dropout=0.05,
    )
    with torch.no_grad():
        adapter.B.normal_(mean=0, std=0.1)

    model = nn.Sequential(adapter, nn.ReLU())
    model.eval()
    inputs = torch.randn(3, 8)

    with torch.no_grad():
        expected = model(inputs).clone()

    merge_model(model)
    assert not any(
        isinstance(module, LoRALinear)
        for module in model.modules()
    )
    torch.testing.assert_close(
        model(inputs), expected, rtol=1e-5, atol=1e-6
    )

    # Repeating merge_model must not add the update again.
    merge_model(model)
    torch.testing.assert_close(
        model(inputs), expected, rtol=1e-5, atol=1e-6
    )

def test_response_only_encoding():
    from app.encoding import encode_pair, format_prompt

    class TinyTokenizer:
        eos_token_id = 256
        pad_token_id = 256

        def encode(self, text, add_special_tokens=False):
            return list(text.encode("utf-8"))

    tokenizer = TinyTokenizer()
    row = {
        "event": "Goal for City.",
        "commentary": "City score!",
    }

    encoded = encode_pair(tokenizer, row, max_length=80)
    prompt_length = len(
        tokenizer.encode(format_prompt(row["event"]))
    )
    target = tokenizer.encode(row["commentary"]) + [256]

    assert len(encoded["input_ids"]) == 80
    assert len(encoded["labels"]) == 80
    assert encoded["labels"][:prompt_length] == [-100] * prompt_length
    assert encoded["labels"][
        prompt_length:prompt_length + len(target)
    ] == target

    real_length = prompt_length + len(target)
    assert encoded["attention_mask"][:real_length] == [1] * real_length
    assert encoded["attention_mask"][real_length:] == [0] * (80 - real_length)
    assert encoded["labels"][real_length:] == [-100] * (80 - real_length)

    with pytest.raises(ValueError, match="exceeds"):
        encode_pair(tokenizer, row, max_length=10)