"""Integration tests for PyTerrier transformer transform() methods."""

import pytest
import pandas as pd

from tests.fixtures.models import (
    create_tiny_cat_ranker,
    create_tiny_dot_ranker,
)


# Skip all tests if pyterrier is not available
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("pyterrier", reason="PyTerrier not installed"),
    reason="PyTerrier not installed",
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing transforms."""
    return pd.DataFrame(
        {
            "qid": ["q1", "q1", "q2", "q2"],
            "query": ["test query", "test query", "another query", "another query"],
            "docno": ["d1", "d2", "d3", "d4"],
            "text": ["document one", "document two", "document three", "document four"],
        }
    )


class TestCatTransformerTransform:
    """Tests for CatTransformer.transform()."""

    def test_cat_transformer_transform_returns_dataframe(self, sample_df):
        """Test transform returns a DataFrame."""
        from rankers.pyterrier.cat import CatTransformer

        cat = create_tiny_cat_ranker()
        cat = cat.to("cpu")

        transformer = CatTransformer.from_model(
            model=cat,
            tokenizer=cat.tokenizer,
            batch_size=2,
            device="cpu",
        )

        result = transformer.transform(sample_df)

        assert isinstance(result, pd.DataFrame)

    def test_cat_transformer_transform_has_score_column(self, sample_df):
        """Test transform adds score column."""
        from rankers.pyterrier.cat import CatTransformer

        cat = create_tiny_cat_ranker()
        cat = cat.to("cpu")

        transformer = CatTransformer.from_model(
            model=cat,
            tokenizer=cat.tokenizer,
            batch_size=2,
            device="cpu",
        )

        result = transformer.transform(sample_df)

        assert "score" in result.columns

    def test_cat_transformer_transform_has_rank_column(self, sample_df):
        """Test transform adds rank column."""
        from rankers.pyterrier.cat import CatTransformer

        cat = create_tiny_cat_ranker()
        cat = cat.to("cpu")

        transformer = CatTransformer.from_model(
            model=cat,
            tokenizer=cat.tokenizer,
            batch_size=2,
            device="cpu",
        )

        result = transformer.transform(sample_df)

        assert "rank" in result.columns

    def test_cat_transformer_transform_preserves_qid(self, sample_df):
        """Test transform preserves qid column."""
        from rankers.pyterrier.cat import CatTransformer

        cat = create_tiny_cat_ranker()
        cat = cat.to("cpu")

        transformer = CatTransformer.from_model(
            model=cat,
            tokenizer=cat.tokenizer,
            batch_size=2,
            device="cpu",
        )

        result = transformer.transform(sample_df)

        assert "qid" in result.columns
        assert set(result["qid"]) == set(sample_df["qid"])

    def test_cat_transformer_transform_scores_are_numeric(self, sample_df):
        """Test transform produces numeric scores."""
        from rankers.pyterrier.cat import CatTransformer

        cat = create_tiny_cat_ranker()
        cat = cat.to("cpu")

        transformer = CatTransformer.from_model(
            model=cat,
            tokenizer=cat.tokenizer,
            batch_size=2,
            device="cpu",
        )

        result = transformer.transform(sample_df)

        assert pd.api.types.is_numeric_dtype(result["score"])


class TestDotTransformerTransform:
    """Tests for DotTransformer.transform() via scorer mode."""

    def test_dot_transformer_query_model(self, sample_df):
        """Test DotTransformer query encoding."""
        from rankers.pyterrier.dot import DotTransformer

        dot = create_tiny_dot_ranker()
        dot = dot.to("cpu")

        transformer = DotTransformer.from_model(
            model=dot,
            tokenizer=dot.tokenizer,
            batch_size=2,
            device="cpu",
        )

        # Use query_model mode
        query_df = sample_df[["query"]].drop_duplicates()
        result = transformer.query_model()(query_df)

        assert "query_vec" in result.columns

    def test_dot_transformer_doc_model(self, sample_df):
        """Test DotTransformer document encoding."""
        from rankers.pyterrier.dot import DotTransformer

        dot = create_tiny_dot_ranker()
        dot = dot.to("cpu")

        transformer = DotTransformer.from_model(
            model=dot,
            tokenizer=dot.tokenizer,
            batch_size=2,
            device="cpu",
        )

        # Use doc_model mode
        result = transformer.doc_model()(sample_df)

        assert "doc_vec" in result.columns

    def test_dot_transformer_encode_queries(self):
        """Test DotTransformer.encode_queries()."""
        from rankers.pyterrier.dot import DotTransformer

        dot = create_tiny_dot_ranker()
        dot = dot.to("cpu")

        transformer = DotTransformer.from_model(
            model=dot,
            tokenizer=dot.tokenizer,
            batch_size=2,
            device="cpu",
        )

        queries = ["test query one", "test query two"]
        embeddings = transformer.encode_queries(queries)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Has embedding dimension

    def test_dot_transformer_encode_docs(self):
        """Test DotTransformer.encode_docs()."""
        from rankers.pyterrier.dot import DotTransformer

        dot = create_tiny_dot_ranker()
        dot = dot.to("cpu")

        transformer = DotTransformer.from_model(
            model=dot,
            tokenizer=dot.tokenizer,
            batch_size=2,
            device="cpu",
        )

        docs = ["document one", "document two", "document three"]
        embeddings = transformer.encode_docs(docs)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0  # Has embedding dimension


class TestTransformerFromToPyterrier:
    """Tests for end-to-end to_pyterrier flow."""

    def test_cat_to_pyterrier_transform(self, sample_df):
        """Test Cat.to_pyterrier() produces working transformer."""
        cat = create_tiny_cat_ranker()
        cat = cat.to("cpu")

        transformer = cat.to_pyterrier(batch_size=2, device="cpu")
        result = transformer.transform(sample_df)

        assert isinstance(result, pd.DataFrame)
        assert "score" in result.columns
        assert "rank" in result.columns

    def test_dot_to_pyterrier_query_encoding(self):
        """Test Dot.to_pyterrier() produces working transformer."""
        dot = create_tiny_dot_ranker()
        dot = dot.to("cpu")

        transformer = dot.to_pyterrier(batch_size=2, device="cpu")
        queries = ["test query"]
        embeddings = transformer.encode_queries(queries)

        assert embeddings.shape[0] == 1
