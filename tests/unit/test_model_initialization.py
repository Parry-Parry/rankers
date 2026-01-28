"""Tests for model class initialization."""

import pytest
import torch

from tests.fixtures.models import (
    create_tiny_bert_config,
    create_tiny_bert_tokenizer,
    create_tiny_cat_ranker,
    create_tiny_dot_ranker,
)


class TestCatInitialization:
    """Tests for Cat model initialization."""

    def test_cat_init_with_tiny_model(self):
        """Test Cat can be initialized with a tiny model."""
        cat = create_tiny_cat_ranker()

        assert cat is not None
        assert cat.model is not None
        assert cat.tokenizer is not None
        assert cat.config is not None
        assert cat.model_type == "Cat"

    def test_cat_has_required_attributes(self):
        """Test Cat has all required attributes after init."""
        cat = create_tiny_cat_ranker()

        assert hasattr(cat, "model")
        assert hasattr(cat, "tokenizer")
        assert hasattr(cat, "config")
        assert hasattr(cat, "prepare_outputs")

    def test_cat_model_is_sequence_classifier(self):
        """Test Cat wraps a sequence classification model."""
        cat = create_tiny_cat_ranker()

        # The underlying model should have a classifier head
        assert hasattr(cat.model, "classifier") or hasattr(cat.model, "cls")

    def test_cat_config_has_group_size(self):
        """Test Cat config has group_size attribute."""
        cat = create_tiny_cat_ranker()

        assert hasattr(cat.config, "group_size")
        assert cat.config.group_size == 2


class TestDotInitialization:
    """Tests for Dot model initialization."""

    def test_dot_init_with_tiny_model(self):
        """Test Dot can be initialized with a tiny model."""
        dot = create_tiny_dot_ranker()

        assert dot is not None
        assert dot.model is not None
        assert dot.tokenizer is not None
        assert dot.config is not None
        assert dot.model_type == "Dot"

    def test_dot_has_required_attributes(self):
        """Test Dot has all required attributes after init."""
        dot = create_tiny_dot_ranker()

        assert hasattr(dot, "model")
        assert hasattr(dot, "model_d")
        assert hasattr(dot, "tokenizer")
        assert hasattr(dot, "config")
        assert hasattr(dot, "pooling")
        assert hasattr(dot, "prepare_outputs")

    def test_dot_tied_model_shares_encoders(self):
        """Test Dot with tied=True shares query and doc encoders."""
        dot = create_tiny_dot_ranker()

        # When model_tied=True, model and model_d should be the same object
        assert dot.config.model_tied is True
        assert dot.model is dot.model_d

    def test_dot_pooling_function_set(self):
        """Test Dot pooling function is properly set from config."""
        dot = create_tiny_dot_ranker()

        assert dot.config.pooling_type == "cls"
        assert callable(dot.pooling)

    def test_dot_config_has_group_size(self):
        """Test Dot config has group_size attribute."""
        dot = create_tiny_dot_ranker()

        assert hasattr(dot.config, "group_size")
        assert dot.config.group_size == 2


class TestModelDeviceHandling:
    """Tests for model device handling."""

    def test_cat_can_move_to_cpu(self):
        """Test Cat model can be moved to CPU."""
        cat = create_tiny_cat_ranker()
        cat = cat.to("cpu")

        device = next(cat.parameters()).device
        assert device.type == "cpu"

    def test_dot_can_move_to_cpu(self):
        """Test Dot model can be moved to CPU."""
        dot = create_tiny_dot_ranker()
        dot = dot.to("cpu")

        device = next(dot.parameters()).device
        assert device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cat_can_move_to_cuda(self):
        """Test Cat model can be moved to CUDA."""
        cat = create_tiny_cat_ranker()
        cat = cat.to("cuda")

        device = next(cat.parameters()).device
        assert device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_dot_can_move_to_cuda(self):
        """Test Dot model can be moved to CUDA."""
        dot = create_tiny_dot_ranker()
        dot = dot.to("cuda")

        device = next(dot.parameters()).device
        assert device.type == "cuda"
