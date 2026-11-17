"""LoRA adapters implemented directly in PyTorch."""
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.05):
        super().__init__()

        if r < 1:
            raise ValueError("r must be positive")

        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad = False

        if isinstance(base, Conv1D):
            d_in, d_out = base.weight.shape
        elif isinstance(base, nn.Linear):
            d_in, d_out = base.in_features, base.out_features
        else:
            raise TypeError("Expected nn.Linear or GPT-2 Conv1D")

        self.A = nn.Parameter(
            torch.randn(
                r,
                d_in,
                device=base.weight.device,
                dtype=base.weight.dtype,
            ) * 0.01
        )
        self.B = nn.Parameter(
            base.weight.new_zeros(d_out, r)
        )
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        detour = (self.dropout(x) @ self.A.T) @ self.B.T
        return self.base(x) + detour * self.scale

    @torch.no_grad()
    def merged(self):
        delta = (self.B @ self.A) * self.scale

        if isinstance(self.base, Conv1D):
            self.base.weight.add_(delta.T)
        else:
            self.base.weight.add_(delta)

        return self.base


def add_lora(model, r=8, alpha=16, dropout=0.05):
    if any(
        isinstance(block.attn.c_attn, LoRALinear)
        for block in model.transformer.h
    ):
        raise ValueError("Model already contains LoRA adapters")

    for parameter in model.parameters():
        parameter.requires_grad = False

    for block in model.transformer.h:
        block.attn.c_attn = LoRALinear(
            block.attn.c_attn,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

    return model