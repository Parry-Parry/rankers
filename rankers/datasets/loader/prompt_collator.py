from typing import Any

import torch


class PromptDataCollator:
    def __init__(
        self,
        tokenizer,
        prompt: Any,
        max_length=512,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length

    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for q, dx, *args in batch:
            batch_queries.extend([q] * len(dx))
            batch_docs.extend(dx)
            if len(args) == 0:
                continue
            batch_scores.extend(args[0])

        sequences = [self.prompt(query=q, doc=d) for q, d in zip(batch_queries, batch_docs)]

        tokenized_sequences = self.tokenizer(
            sequences,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {
            "sequences": dict(tokenized_sequences),
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }
