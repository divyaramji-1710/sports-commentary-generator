"""Train only LoRA parameters on event/commentary pairs."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.encoding import encode_pair
from app.lora import add_lora


def train_adapter(
    output_dir="artifacts",
    epochs=6,
    max_steps=None,
    batch_size=8,
    learning_rate=0.001,
    seed=17,
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")

    torch.manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def load_split(name):
        path = Path("data") / f"{name}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"{name} split is empty")

        return [
            {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in encode_pair(
                    tokenizer, row, max_length=64
                ).items()
            }
            for row in rows
        ]

    train_data = load_split("train")
    validation_data = load_split("validation")

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
    )

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.config.use_cache = False

    adapter_config = {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
    }
    add_lora(model, **adapter_config)

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    trainable = sum(parameter.numel() for parameter in parameters)
    print(f"Trainable parameters: {trainable:,}", flush=True)

    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate
    )

    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss = 0.0
        total_targets = 0

        for batch in validation_loader:
            result = model(**batch)
            # GPT-2 shifts labels internally for next-token prediction.
            count = (batch["labels"][:, 1:] != -100).sum().item()
            total_loss += result.loss.item() * count
            total_targets += count

        model.train()
        return total_loss / total_targets

    history = []
    best_loss = float("inf")
    best_step = 0

    def record(step):
        nonlocal best_loss, best_step

        validation_loss = evaluate()
        history.append({
            "step": step,
            "validation_loss": validation_loss,
        })
        print(
            f"step {step}: validation loss {validation_loss:.4f}",
            flush=True,
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_step = step

            adapter_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if name.endswith(".A") or name.endswith(".B")
            }
            torch.save(
                {
                    "base_model": "distilgpt2",
                    "adapter_config": adapter_config,
                    "adapter_state": adapter_state,
                    "step": step,
                    "validation_loss": validation_loss,
                    "max_length": 64,
                },
                output / "adapter.pt",
            )

    # At step zero, zero-initialized B makes this the base-model baseline.
    record(0)
    step = 0

    for epoch in range(epochs):
        model.train()

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch)

            if not torch.isfinite(result.loss):
                raise RuntimeError("Training loss is not finite")

            result.loss.backward()
            optimizer.step()
            step += 1

            if step % 10 == 0:
                print(
                    f"step {step}: training loss "
                    f"{result.loss.item():.4f}",
                    flush=True,
                )

            if max_steps is not None and step >= max_steps:
                break

        record(step)

        if max_steps is not None and step >= max_steps:
            break

    report = {
        "base_model": "distilgpt2",
        "seed": seed,
        "requested_epochs": epochs,
        "completed_steps": step,
        "learning_rate": learning_rate,
        "trainable_parameters": trainable,
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "history": history,
    }
    (output / "training.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"Saved best adapter and training report in {output}",
        flush=True,
    )
    return report