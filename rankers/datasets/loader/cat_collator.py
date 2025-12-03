import torch


class CatDataCollator:
    def __init__(
        self,
        tokenizer,
        q_max_length=30,
        d_max_length=200,
    ) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for elt in batch:
            q = elt[0]
            dx = elt[1]
            batch_queries.extend([q] * len(dx))
            batch_docs.extend(dx)
            if len(elt) < 3:
                continue
            batch_scores.extend(elt[2])
        tokenized_sequences = self.tokenizer(
            batch_queries,
            batch_docs,
            padding="longest",
            truncation="only_second",
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
        )
        return {
            "sequences": dict(tokenized_sequences),
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }
