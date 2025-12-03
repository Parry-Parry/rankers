import torch


class DotDataCollator:
    def __init__(
        self,
        tokenizer,
        special_mask=False,
        q_max_length=30,
        d_max_length=200,
    ) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length
        self.special_mask = special_mask

    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for elt in batch:
            q = elt[0]
            dx = elt[1]
            batch_queries.append(q)
            batch_docs.extend(dx)
            if len(elt) < 3:
                continue
            batch_scores.extend(elt[2])

        tokenized_queries = self.tokenizer(
            batch_queries,
            padding=True,
            truncation=True,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )
        tokenized_docs = self.tokenizer(
            batch_docs,
            padding="longest",
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )

        return {
            "queries": dict(tokenized_queries),
            "docs_batch": dict(tokenized_docs),
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }
