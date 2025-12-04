"""Tests for ID casting in TrainingDataset with different ID types."""

import json
import tempfile
from pathlib import Path

from rankers.datasets import Corpus, TrainingDataset
from tests.fixtures.data import create_synthetic_corpus


class TestTrainingDatasetIDCasting:
    """Tests for ID casting with different ID types in TrainingDataset."""

    def test_integer_ids_are_cast_to_strings(self):
        """Test that integer IDs are properly cast to strings during training."""
        # Create JSONL with integer query IDs
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            jsonl_file = f.name
            f.write(
                json.dumps(
                    {
                        "query_id": 1,
                        "doc_id_a": "d1",
                        "doc_id_b": "d2",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "query_id": 2,
                        "doc_id_a": "d3",
                        "doc_id_b": "d4",
                    }
                )
                + "\n"
            )

        try:
            corpus_dict = create_synthetic_corpus(num_docs=200, num_queries=10)
            # Map integer query IDs to string keys in corpus
            corpus_dict["queries"] = {
                "1": "query 1",
                "2": "query 2",
                **{str(i): f"query {i}" for i in range(3, 10)},
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

    def test_float_ids_are_cast_to_strings(self):
        """Test that float IDs are properly cast to strings in training."""
        # Create JSONL with float query IDs
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            jsonl_file = f.name
            f.write(
                json.dumps(
                    {
                        "query_id": 1.5,
                        "doc_id_a": "d1",
                        "doc_id_b": "d2",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "query_id": 2.5,
                        "doc_id_a": "d3",
                        "doc_id_b": "d4",
                    }
                )
                + "\n"
            )

        try:
            corpus_dict = create_synthetic_corpus(num_docs=200, num_queries=10)
            # Map float query IDs to string keys in corpus
            corpus_dict["queries"] = {
                "1.5": "query 1",
                "2.5": "query 2",
                **{str(i): f"query {i}" for i in range(3, 10)},
            }

            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
            )

            # Should handle float IDs and convert to strings
            dataset = TrainingDataset(jsonl_file, corpus=corpus, group_size=2)

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
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
            corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=10)
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
            )

            dataset = TrainingDataset(jsonl_file, corpus=corpus, group_size=2)

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            jsonl_file = f.name
            f.write(
                json.dumps(
                    {
                        "query_id": 1,
                        "doc_id_a": "d1",
                        "doc_id_b": 2,
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "query_id": "q2",
                        "doc_id_a": 3,
                        "doc_id_b": "d4",
                    }
                )
                + "\n"
            )

        try:
            corpus_dict = create_synthetic_corpus(num_docs=200, num_queries=10)
            # Map integer query ID 1 to string key
            corpus_dict["queries"] = {
                "1": "query 1",
                **{str(i): f"query {i}" for i in range(2, 10)},
            }
            # Update documents to include the IDs referenced in JSONL (as strings)
            corpus_dict["documents"]["d1"] = "document d1"
            corpus_dict["documents"]["2"] = "document 2"
            corpus_dict["documents"]["3"] = "document 3"
            corpus_dict["documents"]["d4"] = "document d4"

            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
            )

            # Should handle mixed ID types and convert all to strings
            dataset = TrainingDataset(jsonl_file, corpus=corpus, group_size=2)

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
