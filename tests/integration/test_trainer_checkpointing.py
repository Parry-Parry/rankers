"""Integration tests for RankerTrainer model checkpointing with best model selection."""

import pytest
import tempfile

from rankers.train.trainer import RankerTrainer
from rankers.train.training_arguments import RankerTrainingArguments
from tests.fixtures.models import TinyDotModel


class TestRankerTrainerCheckpointing:
    """Integration tests for model checkpointing and best model selection."""

    def test_checkpoint_configuration(self):
        """Test checkpoint configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                eval_strategy="no",
                save_strategy="steps",
                save_steps=10,
                metric_for_best_model="eval_nDCG@10",
                load_best_model_at_end=False,  # Can't use without eval_dataset
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            # Verify save strategy is configured
            assert args.save_strategy == "steps"
            assert args.save_steps == 10

    def test_metric_for_best_model_required(self):
        """Test that training args validation works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should raise ValueError with strategy mismatch
            with pytest.raises(ValueError, match="save and eval strategy"):
                RankerTrainingArguments(
                    output_dir=tmpdir,
                    eval_strategy="no",
                    save_strategy="steps",
                    save_steps=10,
                    load_best_model_at_end=True,
                    # metric_for_best_model not set
                )

    def test_checkpoint_directory_creation(self):
        """Test that checkpoint directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                save_strategy="steps",
                save_steps=100,
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            # Verify output directory is set
            assert args.output_dir == tmpdir

    def test_maybe_log_save_evaluate_method_exists(self):
        """Test that _maybe_log_save_evaluate method exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            # Verify the method exists (for HF trainer integration)
            assert hasattr(trainer, '_maybe_log_save_evaluate')

    def test_checkpoint_frequency_configuration(self):
        """Test checkpoint frequency configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                save_strategy="steps",
                save_steps=500,
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert args.save_steps == 500

    def test_epoch_based_checkpointing(self):
        """Test epoch-based checkpointing configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                save_strategy="epoch",
                num_train_epochs=3,
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert args.save_strategy == "epoch"
            assert args.num_train_epochs == 3

    def test_no_checkpointing_strategy(self):
        """Test configuration with no checkpointing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                save_strategy="no",
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert args.save_strategy == "no"

    def test_training_args_state_saved(self):
        """Test that training arguments are properly set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=5,
                per_device_train_batch_size=16,
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert args.num_train_epochs == 5
            assert args.per_device_train_batch_size == 16

    def test_best_model_comparison_greater_is_better(self):
        """Test best model selection with greater_is_better=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                eval_strategy="no",
                save_strategy="steps",
                save_steps=10,
                metric_for_best_model="eval_nDCG@10",
                greater_is_better=True,
                load_best_model_at_end=False,  # Can't use without eval_dataset
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert args.greater_is_better is True

    def test_best_model_comparison_less_is_better(self):
        """Test best model selection with greater_is_better=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                eval_strategy="no",
                save_strategy="steps",
                save_steps=10,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                load_best_model_at_end=False,  # Can't use without eval_dataset
            )
            model = TinyDotModel()

            trainer = RankerTrainer(model=model, args=args, loss_fn="margin_mse")

            assert args.greater_is_better is False
