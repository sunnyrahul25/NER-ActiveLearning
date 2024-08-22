from typing import Dict

import torch.nn as nn
from termcolor import colored
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


def get_huggingface_optimizer_and_scheduler(
    config,
    model,
    num_training_steps: int,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
    warmup_step: int = 0,
):
    """Copying the optimizer code from HuggingFace."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(_nd in n for _nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(_nd in n for _nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config["learningrate"], eps=eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
    )
    return optimizer, scheduler
