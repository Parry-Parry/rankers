"""Tests for ID casting in TrainingDataset with different ID types (int, float, string)."""

import json
import tempfile
from pathlib import Path

from rankers.datasets import Corpus, TrainingDataset
from tests.fixtures.data import create_synthetic_corpus


class TestTrainingDatasetIDCasting:
    """Tests for ID casting with different ID types in TrainingDataset."""

    def test_integer_ids_are_cast_to_strings(self):
        """Test that integer IDs are properly cast to strings during training."""
        # Create JSONL with integer IDs
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            jsonl_file = f.name
            # Write records with integer IDs
            f.write(
                json.dumps(
                    {
                        "query_id": 1,
                        "doc_id_a": 101,
                        "doc_id_b": 102,
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "query_id": 2,
                        "doc_id_a": 103,
                        "doc_id_b": 104,
                    }
                )
                + "\n"
            )

        try:
            corpus_dict = create_synthetic_corpus(num_docs=200, num_queries=10)
            # Create corpus with integer ID keys for docs
            corpus_dict["documents"] = {
                101: "doc 101",
                102: "doc 102",
                103: "doc 103",
                104: "doc 104",
                **{
                    i: f"doc {i}"
                    for i in range(200)
                    if i not in [101, 102, 103, 104]
                },
            }
            corpus_dict["queries"] = {
                1: "query 1",
                2: "query 2",
                **{i: f"query {i}" for i in range(10) if i not in [1, 2]},
            }

            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
            )

            # Should handle integer IDs and convert to strings
            dataset = TrainingDataset(jsonl_file, corpus=corpus, group_size=2)

            assert dataset is not None
            assert len(dataset) > 0

            # Access internal data to verify IDs are strings
            # Use _get_line_by_index and _standard_get directly to test ID casting
            line = dataset._get_line_by_index(0)
            query_id, query_text, positive_id, positive_text, negative_id, negative_text = dataset._standard_get(line)

            # IDs should be strings
            assert isinstance(query_id, str)
            assert isinstance(positive_id, str)
            if isinstance(negative_id, list):
                assert all(isinstance(nid, str) for nid in negative_id)
            else:
                assert isinstance(negative_id, str)
        finally:
            Path(jsonl_file).unlink(missing_ok=True)

    def test_float_ids_are_cast_to_strings(self):
        """Test that float IDs are properly cast to strings in training."""
        # Create JSONL with float IDs
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            jsonl_file = f.name
            f.write(
                json.dumps(
                    {
                        "query_id": 1.5,
                        "doc_id_a": 101.5,
                        "doc_id_b": 102.5,
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "query_id": 2.5,
                        "doc_id_a": 103.5,
                        "doc_id_b": 104.5,
                    }
                )
                + "\n"
            )

        try:
            corpus_dict = create_synthetic_corpus(
                num_docs=200, num_queries=10
            )
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
            )

            # Should handle float IDs and convert to strings
            dataset = TrainingDataset(
                jsonl_file, corpus=corpus, group_size=2
            )

            assert dataset is not None
            assert len(dataset) > 0

            # Access internal data to verify IDs are strings
            line = dataset._get_line_by_index(0)
            (
                query_id,
                query_text,
                positive_id,
                positive_text,
                negative_id,
                negative_text,
            ) = dataset._standard_get(line)

            # IDs should be strings
            assert isinstance(query_id, str)
            assert isinstance(positive_id, str)
            if isinstance(negative_id, list):
                assert all(isinstance(nid, str) for nid in negative_id)
            else:
                assert isinstance(negative_id, str)
        finally:
            Path(jsonl_file).unlink(missing_ok=True)

    def test_string_ids_remain_strings(self):
        """Test that string IDs are properly handled in training."""
        # Create JSONL with string IDs
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            jsonl_file = f.name
            f.write(
                json.dumps(
                    {
                        "query_id": "q1",
                        "doc_id_a": "d1",
                        "doc_id_b": "d2",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "query_id": "q2",
                        "doc_id_a": "d3",
                        "doc_id_b": "d4",
                    }
                )
                + "\n"
            )

        try:
            corpus_dict = create_synthetic_corpus(
                num_docs=30, num_queries=10
            )
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
            )

            dataset = TrainingDataset(
                jsonl_file, corpus=corpus, group_size=2
            )

            assert dataset is not None
            assert len(dataset) > 0

            # Access internal data to verify IDs are strings
            line = dataset._get_line_by_index(0)
            (
                query_id,
                query_text,
                positive_id,
                positive_text,
                negative_id,
                negative_text,
            ) = dataset._standard_get(line)

            # IDs should be strings
            assert isinstance(query_id, str)
            assert isinstance(positive_id, str)
            assert query_id in ["q1", "q2"]
            assert positive_id in ["d1", "d3"]
            if isinstance(negative_id, list):
                assert all(isinstance(nid, str) for nid in negative_id)
                assert all(nid in ["d2", "d4"] for nid in negative_id)
            else:
                assert isinstance(negative_id, str)
                assert negative_id in ["d2", "d4"]
        finally:
            Path(jsonl_file).unlink(missing_ok=True)

    def test_mixed_numeric_string_ids(self):
        """Test that mixed numeric and string IDs are all cast to strings."""
        # Create JSONL with mixed ID types
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            jsonl_file = f.name
            f.write(
                json.dumps(
                    {
                        "query_id": 1,
                        "doc_id_a": "d_101",
                        "doc_id_b": 102,
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "query_id": "q2",
                        "doc_id_a": 103,
                        "doc_id_b": "d_104",
                    }
                )
                + "\n"
            )

        try:
            corpus_dict = create_synthetic_corpus(
                num_docs=200, num_queries=10
            )
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
            )

            # Should handle mixed ID types and convert all to strings
            dataset = TrainingDataset(
                jsonl_file, corpus=corpus, group_size=2
            )

            assert dataset is not None
            assert len(dataset) > 0

            # Access internal data to verify IDs are strings
            line = dataset._get_line_by_index(0)
            (
                query_id,
                query_text,
                positive_id,
                positive_text,
                negative_id,
                negative_text,
            ) = dataset._standard_get(line)

            # All IDs should be strings
            assert isinstance(query_id, str)
            assert isinstance(positive_id, str)
            if isinstance(negative_id, list):
                assert all(isinstance(nid, str) for nid in negative_id)
            else:
                assert isinstance(negative_id, str)
        finally:
            Path(jsonl_file).unlink(missing_ok=True)
