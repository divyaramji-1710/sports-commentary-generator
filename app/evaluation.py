"""Compare adapted and base-model outputs on held-out events."""
import json
import re
from pathlib import Path

import torch

from app.data import TEMPLATES
from app.inference import CommentaryGenerator
from app.lora import LoRALinear


def check_commentary(row, text):
    normalized = " ".join(text.split())

    # Require the correct home/away order and associated scores.
    score_pattern = (
        rf"\b{re.escape(row['home'])}\s+"
        rf"{row['home_score']}\s*-\s*{row['away_score']}"
        rf"\s+{re.escape(row['away'])}\b"
    )
    scores = re.findall(r"\b\d+\s*-\s*\d+\b", normalized)
    expected_score = f"{row['home_score']}-{row['away_score']}"

    score_correct = (
        re.search(score_pattern, normalized, re.IGNORECASE) is not None
        and bool(scores)
        and all(
            re.sub(r"\s+", "", score) == expected_score
            for score in scores
        )
    )

    team = re.escape(row["team"])
    if row["event_type"] == "goal":
        event_pattern = (
            rf"\bgoal for {team}\b|\b{team} score\b"
        )
    elif row["event_type"] == "save":
        event_pattern = (
            rf"\bsave by {team}\b|"
            rf"\bsaved by {team}\b|"
            rf"\b{team} make the save\b"
        )
    else:
        event_pattern = r"\bfull[- ]time\b|\bmatch is over\b"

    event_correct = (
        re.search(event_pattern, normalized, re.IGNORECASE)
        is not None
    )

    # A stricter, narrow check: matches any valid authored template.
    acceptable = {
        template.format(
            team=row["team"],
            home=row["home"],
            away=row["away"],
            hs=row["home_score"],
            as_=row["away_score"],
        )
        for template in TEMPLATES[row["event_type"]]
    }

    return {
        "score_correct": score_correct,
        "event_and_actor_correct": event_correct,
        "both_correct": score_correct and event_correct,
        "matches_valid_template": normalized in acceptable,
    }


def evaluate_model(
    checkpoint="artifacts/commentary/adapter.pt",
    output_path="artifacts/evaluation.json",
):
    rows = [
        json.loads(line)
        for line in Path("data/test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Test split is empty")

    generator = CommentaryGenerator(checkpoint)
    results = []

    print("Evaluating adapted model...", flush=True)
    for index, row in enumerate(rows, start=1):
        text = generator.generate(row["event"])
        results.append({
            "event": row["event"],
            "reference": row["commentary"],
            "event_type": row["event_type"],
            "adapted_output": text,
            "adapted_checks": check_commentary(row, text),
        })
        if index % 10 == 0:
            print(f"Adapted: {index}/{len(rows)}", flush=True)

    # Zeroing B disables the detours, recovering the unchanged base.
    # This affects only the temporary in-memory model, not the file.
    with torch.no_grad():
        for module in generator.model.modules():
            if isinstance(module, LoRALinear):
                module.B.zero_()

    print("Evaluating base model...", flush=True)
    for index, (row, result) in enumerate(zip(rows, results), start=1):
        text = generator.generate(row["event"])
        result["base_output"] = text
        result["base_checks"] = check_commentary(row, text)

        if index % 10 == 0:
            print(f"Base: {index}/{len(rows)}", flush=True)

    metrics = [
        "score_correct",
        "event_and_actor_correct",
        "both_correct",
        "matches_valid_template",
    ]
    summary = {
        model: {
            metric: sum(
                result[f"{model}_checks"][metric]
                for result in results
            ) / len(results)
            for metric in metrics
        }
        for model in ["base", "adapted"]
    }

    report = {
        "checkpoint": str(checkpoint),
        "test_examples": len(rows),
        "decoding": "greedy, max_new_tokens=48",
        "limitations": (
            "Rule-based checks with shared training/test templates. "
            "Not a general fluency or creativity assessment."
        ),
        "summary": summary,
        "examples": results,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {path}", flush=True)
    return report