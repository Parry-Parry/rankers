from typing import Any

import torch


class ListwisePromptDataCollator:
    def __init__(
        self,
        tokenizer,
        prompt: Any,
        score_format: callable = None,
        max_length=8192,
        use_scores=False,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.score_format = score_format
        self.max_length = max_length
        self.use_scores = use_scores

    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for q, dx, *args in batch:
            batch_queries.append(q)
            batch_docs.append(dx)
            if len(args) == 0:
                continue
            batch_scores.append(args[0])
        if self.use_scores:
            assert len(batch_scores) > 0, (
                "Scores are required for ListWisePromptDataCollator with current settings"
            )
            sequences = [
                self.prompt(query=q, docs=dx, scores=score)
                for q, dx, score in zip(batch_queries, batch_docs, batch_scores)
            ]
        else:
            sequences = [
                self.prompt(query=q, docs=dx)
                for q, dx in zip(batch_queries, batch_docs)
            ]

        tokenized_sequences = self.tokenizer(
            sequences,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        if self.score_format:
            batch_scores = [self.score_format(score) for score in batch_scores]
            tokenized_scores = self.tokenizer(
                batch_scores,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=True,
            )
            return {
                "sequences": dict(tokenized_sequences),
                "labels": dict(tokenized_scores),
            }

        return {
            "sequences": dict(tokenized_sequences),
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }
