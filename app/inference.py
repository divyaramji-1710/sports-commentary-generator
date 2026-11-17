"""Load the saved LoRA adapter and generate commentary."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.encoding import format_prompt
from app.lora import add_lora


class CommentaryGenerator:
    def __init__(
        self,
        checkpoint="artifacts/commentary/adapter.pt",
    ):
        saved = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            saved["base_model"]
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            saved["base_model"]
        )
        add_lora(self.model, **saved["adapter_config"])

        missing, unexpected = self.model.load_state_dict(
            saved["adapter_state"],
            strict=False,
        )
        if unexpected:
            raise ValueError(f"Unexpected adapter keys: {unexpected}")
        if any(
            name.endswith(".A") or name.endswith(".B")
            for name in missing
        ):
            raise ValueError("Checkpoint is missing adapter tensors")

        self.model.eval()

    @torch.no_grad()
    def generate(self, event, max_new_tokens=48):
        if not 1 <= max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be between 1 and 128")

        inputs = self.tokenizer(
            format_prompt(event),
            return_tensors="pt",
        )
        prompt_length = inputs["input_ids"].shape[1]

        if (
            prompt_length + max_new_tokens
            > self.model.config.n_positions
        ):
            raise ValueError("Event is too long for the model")

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        return self.tokenizer.decode(
            output[0, prompt_length:],
            skip_special_tokens=True,
        ).strip()