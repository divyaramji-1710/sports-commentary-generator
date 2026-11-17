"""Merge LoRA weights and verify inference equivalence."""
import json
from pathlib import Path

import torch

from app.encoding import format_prompt
from app.inference import CommentaryGenerator
from app.lora import LoRALinear


def merge_model(model):
    """Replace LoRA wrappers with their merged base layers."""
    model.eval()

    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
            else:
                replace(child)

    replace(model)
    return model


@torch.no_grad()
def verify_merge(
    checkpoint="artifacts/commentary/adapter.pt",
    output_path="artifacts/merge-check.json",
):
    generator = CommentaryGenerator(checkpoint)
    rows = [
        json.loads(line)
        for line in Path("data/validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    examples = [
        next(row for row in rows if row["event_type"] == kind)
        for kind in ["goal", "save", "full-time"]
    ]

    before = []
    for row in examples:
        inputs = generator.tokenizer(
            format_prompt(row["event"]),
            return_tensors="pt",
        )
        logits = generator.model(**inputs).logits.detach().clone()
        text = generator.generate(row["event"])
        before.append((inputs, logits, text))

    wrappers_before = sum(
        isinstance(module, LoRALinear)
        for module in generator.model.modules()
    )

    merge_model(generator.model)

    wrappers_after = sum(
        isinstance(module, LoRALinear)
        for module in generator.model.modules()
    )

    results = []
    for row, (inputs, old_logits, old_text) in zip(examples, before):
        new_logits = generator.model(**inputs).logits
        new_text = generator.generate(row["event"])

        results.append({
            "event": row["event"],
            "before": old_text,
            "after": new_text,
            "same_text": old_text == new_text,
            "max_logit_difference": (
                old_logits - new_logits
            ).abs().max().item(),
            "logits_close": torch.allclose(
                old_logits,
                new_logits,
                rtol=1e-4,
                atol=5e-4,
            ),
        })

    report = {
        "checkpoint": str(checkpoint),
        "wrappers_before": wrappers_before,
        "wrappers_after": wrappers_after,
        "all_text_matches": all(r["same_text"] for r in results),
        "all_logits_close": all(r["logits_close"] for r in results),
        "examples": results,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return report