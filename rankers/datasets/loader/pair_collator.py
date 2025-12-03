import torch


def _make_pos_pairs(texts) -> list:
    output = []
    pos = texts[0]
    for i in range(1, len(texts)):
        output.append([pos, texts[i]])
    return output


class PairDataCollator:
    def __init__(self, tokenizer, max_length=512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch) -> dict:
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for q, dx, *args in batch:
            batch_queries.append(q)
            batch_document_pairs = _make_pos_pairs(dx)
            batch_docs.append(batch_document_pairs)
            if len(args) == 0:
                continue
            batch_score_pairs = _make_pos_pairs(args[0])
            batch_scores.extend(batch_score_pairs)

        # tokenize each pair with each query
        sequences = [
            f"[CLS] {query} [SEP] {pair[0]} [SEP] {pair[1]}"
            for query, pairs in zip(batch_queries, batch_docs)
            for pair in pairs
        ]

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
            "labels": (
                torch.tensor(batch_scores).squeeze() if len(batch_scores) > 0 else None
            ),
        }
