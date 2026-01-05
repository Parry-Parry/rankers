"""Unit tests for RankerTrainer metric computation."""

import tempfile

from rankers.train.trainer import RankerTrainer
from rankers.train.training_arguments import RankerTrainingArguments
from tests.fixtures.models import TinyDotModel


class TestRankerTrainerMetrics:
    """Tests for RankerTrainer metric computation."""

    def test_compute_metrics_exists(self):
        """Test that compute_metrics method exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            # Test that compute_metrics method exists and is callable
            assert hasattr(trainer, "compute_metrics")
            assert callable(trainer.compute_metrics)

    def test_trainer_has_loss_function(self):
        """Test that trainer has loss function set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert trainer.loss is not None

    def test_trainer_with_eval_metrics(self):
        """Test trainer initialization with evaluation metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                eval_metrics=["nDCG@10", "MRR"],
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            # Verify eval metrics are set
            assert args.eval_ir_metrics is not None
            assert len(args.eval_ir_metrics) == 2

    def test_trainer_default_metrics_list(self):
        """Test that default eval metrics list is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            # Default metrics should be None or empty
            assert args.eval_ir_metrics is None
