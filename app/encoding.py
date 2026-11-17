"""Format event prompts and supervise commentary tokens only."""


def format_prompt(event):
    event = event.strip()
    if not event:
        raise ValueError("Event must not be empty")
    return f"Event: {event}\nCommentary:\n"


def encode_pair(tokenizer, row, max_length=128):
    prompt_ids = tokenizer.encode(
        format_prompt(row["event"]),
        add_special_tokens=False,
    )
    response_ids = tokenizer.encode(
        row["commentary"].strip(),
        add_special_tokens=False,
    )

    if not response_ids:
        raise ValueError("Commentary must not be empty")

    # Teach the model to stop after its commentary.
    response_ids = response_ids + [tokenizer.eos_token_id]

    if len(prompt_ids) + len(response_ids) > max_length:
        raise ValueError(
            "Example exceeds max_length; increase it rather "
            "than silently removing event facts or response tokens"
        )

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)

    padding = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding
    labels += [-100] * padding
    attention_mask += [0] * padding

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }