"""Integration tests for RankerTrainer evaluation functionality."""

import tempfile

from rankers.train.trainer import RankerTrainer
from rankers.train.training_arguments import RankerTrainingArguments
from tests.fixtures.models import TinyDotModel


class TestRankerTrainerEvaluation:
    """Integration tests for evaluation functionality."""

    def test_trainer_initialization_with_eval_strategy(self):
        """Test that trainer initializes with evaluation strategy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                eval_strategy="no",  # no evaluation strategy
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert trainer is not None
            assert trainer.args.eval_strategy is not None

    def test_trainer_initialization_with_metrics(self):
        """Test that trainer initializes with metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                eval_strategy="no",
                eval_metrics=["nDCG@10", "MRR"],
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert trainer is not None
            assert args.eval_ir_metrics is not None

    def test_trainer_has_compute_metrics_method(self):
        """Test that trainer has compute_metrics method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                eval_strategy="no",
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert hasattr(trainer, 'compute_metrics')
            assert callable(trainer.compute_metrics)

    def test_trainer_model_has_to_pyterrier(self):
        """Test that model has to_pyterrier method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert hasattr(model, 'to_pyterrier')
            assert callable(model.to_pyterrier)

    def test_trainer_eval_dataset_handling(self):
        """Test that trainer can handle eval_dataset attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            # Trainer should be able to have eval_dataset set
            assert hasattr(trainer, 'eval_dataset')
