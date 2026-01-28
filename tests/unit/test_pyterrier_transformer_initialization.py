"""Tests for PyTerrier transformer initialization."""

import pytest

from tests.fixtures.models import (
    create_tiny_cat_ranker,
    create_tiny_dot_ranker,
)


# Skip all tests if pyterrier is not available
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("pyterrier", reason="PyTerrier not installed"),
    reason="PyTerrier not installed",
)


class TestCatTransformerInitialization:
    """Tests for CatTransformer initialization paths."""

    def test_cat_transformer_from_model(self):
        """Test CatTransformer.from_model with Cat ranker."""
        from rankers.pyterrier.cat import CatTransformer

        cat = create_tiny_cat_ranker()
        tokenizer = cat.tokenizer

        transformer = CatTransformer.from_model(
            model=cat,
            tokenizer=tokenizer,
            batch_size=32,
            device="cpu",
        )

        assert transformer is not None
        assert transformer.model is not None
        assert transformer.tokenizer is tokenizer
        assert transformer.batch_size == 32
        assert transformer.device == "cpu"

    def test_cat_transformer_from_model_creates_copy(self):
        """Test from_model creates a deepcopy of the model."""
        from rankers.pyterrier.cat import CatTransformer

        cat = create_tiny_cat_ranker()
        tokenizer = cat.tokenizer

        transformer = CatTransformer.from_model(
            model=cat,
            tokenizer=tokenizer,
            device="cpu",
        )

        # The transformer's model should be a different object
        assert transformer.model is not cat

    def test_cat_transformer_init_directly(self):
        """Test CatTransformer.__init__ with Cat ranker."""
        from rankers.pyterrier.cat import CatTransformer

        cat = create_tiny_cat_ranker()
        tokenizer = cat.tokenizer

        transformer = CatTransformer(
            model=cat,
            tokenizer=tokenizer,
            config=cat.config,
            batch_size=64,
            text_field="text",
            device="cpu",
            verbose=False,
        )

        assert transformer is not None
        assert transformer.batch_size == 64
        assert transformer.text_field == "text"

    def test_cat_to_pyterrier(self):
        """Test Cat.to_pyterrier() end-to-end."""
        cat = create_tiny_cat_ranker()
        cat = cat.to("cpu")

        transformer = cat.to_pyterrier(batch_size=16, device="cpu")

        assert transformer is not None
        assert transformer.batch_size == 16

    def test_cat_transformer_model_in_eval_mode(self):
        """Test transformer sets model to eval mode."""
        from rankers.pyterrier.cat import CatTransformer

        cat = create_tiny_cat_ranker()
        cat.train()  # Set to train mode first

        transformer = CatTransformer.from_model(
            model=cat,
            tokenizer=cat.tokenizer,
            device="cpu",
        )

        # After creating transformer, model should be in eval mode
        assert not transformer.model.training


class TestDotTransformerInitialization:
    """Tests for DotTransformer initialization paths."""

    def test_dot_transformer_from_model(self):
        """Test DotTransformer.from_model with Dot ranker."""
        from rankers.pyterrier.dot import DotTransformer

        dot = create_tiny_dot_ranker()
        tokenizer = dot.tokenizer

        transformer = DotTransformer.from_model(
            model=dot,
            tokenizer=tokenizer,
            batch_size=32,
            device="cpu",
        )

        assert transformer is not None
        assert transformer.model is not None
        assert transformer.tokenizer is tokenizer
        assert transformer.batch_size == 32
        assert transformer.device == "cpu"

    def test_dot_transformer_from_model_creates_copy(self):
        """Test from_model creates a deepcopy of the model."""
        from rankers.pyterrier.dot import DotTransformer

        dot = create_tiny_dot_ranker()
        tokenizer = dot.tokenizer

        transformer = DotTransformer.from_model(
            model=dot,
            tokenizer=tokenizer,
            device="cpu",
        )

        # The transformer's model should be a different object
        assert transformer.model is not dot

    def test_dot_transformer_init_directly(self):
        """Test DotTransformer.__init__ with Dot ranker."""
        from rankers.pyterrier.dot import DotTransformer

        dot = create_tiny_dot_ranker()
        tokenizer = dot.tokenizer

        transformer = DotTransformer(
            model=dot,
            tokenizer=tokenizer,
            config=dot.config,
            batch_size=64,
            text_field="text",
            device="cpu",
            verbose=False,
        )

        assert transformer is not None
        assert transformer.batch_size == 64
        assert transformer.text_field == "text"

    def test_dot_to_pyterrier(self):
        """Test Dot.to_pyterrier() end-to-end."""
        dot = create_tiny_dot_ranker()
        dot = dot.to("cpu")

        transformer = dot.to_pyterrier(batch_size=16, device="cpu")

        assert transformer is not None
        assert transformer.batch_size == 16

    def test_dot_transformer_model_in_eval_mode(self):
        """Test transformer sets model to eval mode."""
        from rankers.pyterrier.dot import DotTransformer

        dot = create_tiny_dot_ranker()
        dot.train()  # Set to train mode first

        transformer = DotTransformer.from_model(
            model=dot,
            tokenizer=dot.tokenizer,
            device="cpu",
        )

        # After creating transformer, model should be in eval mode
        assert not transformer.model.training


class TestPairTransformerInitialization:
    """Tests for PairTransformer initialization paths."""

    def test_pair_transformer_from_model(self):
        """Test PairTransformer.from_model with Cat ranker."""
        from rankers.pyterrier.cat.cat import PairTransformer

        cat = create_tiny_cat_ranker()
        tokenizer = cat.tokenizer

        transformer = PairTransformer.from_model(
            model=cat,
            tokenizer=tokenizer,
            batch_size=32,
            device="cpu",
        )

        assert transformer is not None
        assert transformer.model is not None
        assert transformer.tokenizer is tokenizer
        assert transformer.batch_size == 32

    def test_pair_transformer_from_model_creates_copy(self):
        """Test from_model creates a deepcopy of the model."""
        from rankers.pyterrier.cat.cat import PairTransformer

        cat = create_tiny_cat_ranker()
        tokenizer = cat.tokenizer

        transformer = PairTransformer.from_model(
            model=cat,
            tokenizer=tokenizer,
            device="cpu",
        )

        # The transformer's model should be a different object
        assert transformer.model is not cat


class TestTransformerDeviceInference:
    """Tests for device inference in transformers."""

    def test_cat_transformer_infers_device_from_model(self):
        """Test CatTransformer infers device from model when not specified."""
        from rankers.pyterrier.cat import CatTransformer

        cat = create_tiny_cat_ranker()
        cat = cat.to("cpu")

        transformer = CatTransformer.from_model(
            model=cat,
            tokenizer=cat.tokenizer,
            device=None,  # Should infer from model
        )

        assert transformer.device == "cpu" or str(transformer.device) == "cpu"

    def test_dot_transformer_infers_device_from_model(self):
        """Test DotTransformer infers device from model when not specified."""
        from rankers.pyterrier.dot import DotTransformer

        dot = create_tiny_dot_ranker()
        dot = dot.to("cpu")

        transformer = DotTransformer.from_model(
            model=dot,
            tokenizer=dot.tokenizer,
            device=None,  # Should infer from model
        )

        assert transformer.device == "cpu" or str(transformer.device) == "cpu"
