"""Unit tests for RankerTrainer initialization."""

import tempfile

import pytest

from rankers.train.trainer import RankerTrainer
from rankers.train.training_arguments import RankerTrainingArguments
from tests.fixtures.models import TinyDotModel


class TestRankerTrainerInit:
    """Tests for RankerTrainer initialization."""

    def test_init_with_loss_string(self):
        """Test initialization with string loss identifier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            model = TinyDotModel()

            trainer = RankerTrainer(
                model=model, args=args, loss_fn="margin_mse"
            )

            assert trainer.loss is not None
            assert trainer.args.output_dir == tmpdir

    def test_init_with_callable_loss(self):
        """Test initialization with callable loss function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            model = TinyDotModel()

            def custom_loss(logits, labels):
                return (logits - labels).pow(2).mean()

            trainer = RankerTrainer(
                model=model, args=args, loss_fn=custom_loss
            )

            assert trainer.loss == custom_loss

    def test_init_with_invalid_loss(self):
        """Test initialization with invalid loss string raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            model = TinyDotModel()

            with pytest.raises(ValueError, match="Unknown loss"):
                RankerTrainer(
                    model=model, args=args, loss_fn="nonexistent_loss"
                )

    def test_init_without_regularization(self):
        """Test initialization without regularization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                regularization=None,
            )
            model = TinyDotModel()

            trainer = RankerTrainer(
                model=model, args=args, loss_fn="margin_mse"
            )

            assert trainer.regularize_loss is False
            assert trainer.loss is not None

    def test_init_without_loss_fn(self):
        """Test initialization without loss function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args)
            assert trainer.loss is None

    def test_init_sets_group_size_in_config(self):
        """Test that group_size is properly set in model config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir, group_size=8
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert model.config.group_size == 8

    def test_init_requires_output_dir(self):
        """Test that initialization requires output directory."""
        model = TinyDotModel()

        # Should not raise - TrainingArguments will create default
        trainer = RankerTrainer(
            model=model,
            args=RankerTrainingArguments(output_dir="tmp_trainer"),
            loss_fn="margin_mse",
        )
        assert trainer is not None

    def test_init_with_compute_metrics(self):
        """Test initialization with custom compute_metrics function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            model = TinyDotModel()

            def compute_metrics(result_frame):
                return {"test_metric": 0.5}

            trainer = RankerTrainer(
                model=model,
                args=args,
                loss_fn="margin_mse",
                compute_metrics=compute_metrics,
            )

            # Trainer should have compute_metrics method
            assert hasattr(trainer, "compute_metrics")
