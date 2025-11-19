"""Integration tests for EvaluationDataset class."""

from rankers.datasets import EvaluationDataset, Corpus
from tests.fixtures.data import (
    create_synthetic_jsonl,
    create_synthetic_trec,
    create_synthetic_corpus,
    create_synthetic_qrels,
    cleanup_temp_files,
)


class TestEvaluationDatasetFromJsonl:
    """Tests for EvaluationDataset.from_jsonl() method."""

    def test_from_jsonl_creates_dataset(self):
        """Test that from_jsonl creates a valid dataset."""
        jsonl_file, records = create_synthetic_jsonl(num_queries=5)

        try:
            corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
            qrels_df = create_synthetic_qrels(num_queries=5)
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
                qrels=qrels_df,
            )

            dataset = EvaluationDataset.from_jsonl(jsonl_file, corpus)

            assert dataset is not None
            assert hasattr(dataset, "data")
            assert hasattr(dataset, "qrels")
        finally:
            cleanup_temp_files([jsonl_file])

    def test_from_jsonl_builds_qrels(self):
        """Test that from_jsonl properly builds qrels from positives."""
        jsonl_file, records = create_synthetic_jsonl(num_queries=5)

        try:
            corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
            qrels_df = create_synthetic_qrels(num_queries=5)
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
                qrels=qrels_df,
            )

            dataset = EvaluationDataset.from_jsonl(jsonl_file, corpus)

            # Check qrels has query_id, doc_id, relevance columns
            assert "query_id" in dataset.qrels.columns
            assert "doc_id" in dataset.qrels.columns
            assert "relevance" in dataset.qrels.columns

            # Should have positive documents as qrels
            assert len(dataset.qrels) > 0
        finally:
            cleanup_temp_files([jsonl_file])

    def test_from_jsonl_with_custom_relevance_label(self):
        """Test from_jsonl with custom relevance label."""
        jsonl_file, records = create_synthetic_jsonl(num_queries=5)

        try:
            corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
            qrels_df = create_synthetic_qrels(num_queries=5)
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
                qrels=qrels_df,
            )

            dataset = EvaluationDataset.from_jsonl(
                jsonl_file, corpus, relevance_label=3
            )

            # All positive docs should have relevance_label=3
            assert (dataset.qrels["relevance"] == 3).all()
        finally:
            cleanup_temp_files([jsonl_file])

    def test_from_jsonl_include_negatives_true(self):
        """Test from_jsonl with include_negatives=True."""
        jsonl_file, records = create_synthetic_jsonl(num_queries=5)

        try:
            corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
            qrels_df = create_synthetic_qrels(num_queries=5)
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
                qrels=qrels_df,
            )

            dataset = EvaluationDataset.from_jsonl(
                jsonl_file, corpus, include_negatives=True
            )

            # Data should include both positives and negatives
            assert "score" in dataset.data.columns
            # Should have more rows than queries (pos + negatives)
            assert len(dataset.data) > 5
        finally:
            cleanup_temp_files([jsonl_file])

    def test_from_jsonl_include_negatives_false(self):
        """Test from_jsonl with include_negatives=False."""
        jsonl_file, records = create_synthetic_jsonl(num_queries=5)

        try:
            corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
            qrels_df = create_synthetic_qrels(num_queries=5)
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
                qrels=qrels_df,
            )

            dataset = EvaluationDataset.from_jsonl(
                jsonl_file, corpus, include_negatives=False
            )

            # Data should include only positives
            # Should have only 5 rows (one per query)
            assert len(dataset.data) == 5
        finally:
            cleanup_temp_files([jsonl_file])


class TestEvaluationDatasetFromTrec:
    """Tests for EvaluationDataset.from_trec() method."""

    def test_from_trec_creates_dataset(self):
        """Test that from_trec creates a valid dataset."""
        trec_file, trec_df = create_synthetic_trec(num_queries=5)

        try:
            corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
            qrels_df = create_synthetic_qrels(num_queries=5)
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
                qrels=qrels_df,
            )

            dataset = EvaluationDataset.from_trec(trec_file, corpus)

            assert dataset is not None
            assert hasattr(dataset, "data")
            assert len(dataset.data) > 0
        finally:
            cleanup_temp_files([trec_file])

    def test_from_trec_preserves_rankings(self):
        """Test that from_trec preserves ranking order."""
        trec_file, trec_df = create_synthetic_trec(num_queries=5)

        try:
            corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
            qrels_df = create_synthetic_qrels(num_queries=5)
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
                qrels=qrels_df,
            )

            dataset = EvaluationDataset.from_trec(trec_file, corpus)

            # Data should have qid, docno, score columns
            assert "qid" in dataset.data.columns
            assert "docno" in dataset.data.columns
            assert "score" in dataset.data.columns
        finally:
            cleanup_temp_files([trec_file])


class TestEvaluationDatasetFromQrels:
    """Tests for EvaluationDataset.from_qrels() method."""

    def test_from_qrels_creates_dataset(self):
        """Test that from_qrels creates a valid dataset."""
        qrels_df = create_synthetic_qrels(num_queries=5)

        corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
        corpus = Corpus(
            documents=corpus_dict["documents"],
            queries=corpus_dict["queries"],
            qrels=qrels_df,
        )

        dataset = EvaluationDataset.from_qrels(qrels_df, corpus)

        assert dataset is not None
        assert hasattr(dataset, "data")
        assert hasattr(dataset, "qrels")

    def test_from_qrels_converts_column_names(self):
        """Test that from_qrels properly converts column names."""
        qrels_df = create_synthetic_qrels(num_queries=5)

        corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
        corpus = Corpus(
            documents=corpus_dict["documents"],
            queries=corpus_dict["queries"],
            qrels=qrels_df,
        )

        dataset = EvaluationDataset.from_qrels(qrels_df, corpus)

        # Data should have TREC format columns
        assert "qid" in dataset.data.columns
        assert "docno" in dataset.data.columns
        assert "score" in dataset.data.columns

    def test_from_qrels_preserves_relevance(self):
        """Test that from_qrels preserves relevance values."""
        qrels_df = create_synthetic_qrels(num_queries=5)

        corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
        corpus = Corpus(
            documents=corpus_dict["documents"],
            queries=corpus_dict["queries"],
            qrels=qrels_df,
        )

        dataset = EvaluationDataset.from_qrels(qrels_df, corpus)

        # Relevance values should be preserved as scores
        original_relevances = set(qrels_df["relevance"].values)
        data_scores = set(dataset.data["score"].values)

        # Should have overlapping relevance values
        assert len(original_relevances & data_scores) > 0


class TestEvaluationDatasetCommon:
    """Tests for common EvaluationDataset functionality."""

    def test_dataset_has_length(self):
        """Test that EvaluationDataset has proper __len__ implementation."""
        jsonl_file, records = create_synthetic_jsonl(num_queries=5)

        try:
            corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
            qrels_df = create_synthetic_qrels(num_queries=5)
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
                qrels=qrels_df,
            )

            dataset = EvaluationDataset.from_jsonl(jsonl_file, corpus)

            # Length should be number of unique queries
            assert len(dataset) == 5
        finally:
            cleanup_temp_files([jsonl_file])

    def test_lazy_load_text_parameter(self):
        """Test lazy_load_text parameter."""
        jsonl_file, records = create_synthetic_jsonl(num_queries=5)

        try:
            corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
            qrels_df = create_synthetic_qrels(num_queries=5)
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
                qrels=qrels_df,
            )

            # With lazy loading
            dataset_lazy = EvaluationDataset.from_jsonl(
                jsonl_file, corpus, lazy_load_text=True
            )
            assert dataset_lazy.lazy_load_text is True

            # Without lazy loading
            dataset_eager = EvaluationDataset.from_jsonl(
                jsonl_file, corpus, lazy_load_text=False
            )
            assert dataset_eager.lazy_load_text is False
        finally:
            cleanup_temp_files([jsonl_file])

    def test_custom_key_mapping(self):
        """Test custom JSON key mapping for from_jsonl."""
        jsonl_file, records = create_synthetic_jsonl(num_queries=5)

        try:
            corpus_dict = create_synthetic_corpus(num_docs=30, num_queries=5)
            qrels_df = create_synthetic_qrels(num_queries=5)
            corpus = Corpus(
                documents=corpus_dict["documents"],
                queries=corpus_dict["queries"],
                qrels=qrels_df,
            )

            # Default key mapping should work
            dataset = EvaluationDataset.from_jsonl(
                jsonl_file,
                corpus,
                query_id_key="query_id",
                positive_id_key="doc_id_a",
                negative_id_key="doc_id_b",
            )

            assert dataset is not None
        finally:
            cleanup_temp_files([jsonl_file])
