"""Unit tests for RankerTrainer configuration."""

import tempfile

import pytest
import torch

from rankers.train.training_arguments import RankerTrainingArguments


class TestRankerTrainingArguments:
    """Tests for RankerTrainingArguments configuration."""

    def test_default_group_size(self):
        """Test default group size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir)
            assert args.group_size == 2

    def test_custom_group_size(self):
        """Test custom group size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(output_dir=tmpdir, group_size=8)
            assert args.group_size == 8

    def test_eval_strategy_steps(self):
        """Test evaluation strategy with steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                eval_strategy="steps",
                eval_steps=500,
            )
            assert args.eval_strategy == "steps"
            assert args.eval_steps == 500

    def test_eval_strategy_epoch(self):
        """Test evaluation strategy with epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                eval_strategy="epoch",
            )
            assert args.eval_strategy == "epoch"

    def test_save_strategy_best(self):
        """Test save strategy set to best."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                save_strategy="steps",
                save_steps=500,
                metric_for_best_model="eval_nDCG@10",
            )
            assert args.save_strategy == "steps"
            assert args.metric_for_best_model == "eval_nDCG@10"

    def test_load_best_model_at_end(self):
        """Test load_best_model_at_end flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                load_best_model_at_end=True,
                metric_for_best_model="eval_nDCG@10",
                eval_strategy="steps",
                eval_steps=500,
                save_strategy="steps",
                save_steps=500,
            )
            assert args.load_best_model_at_end is True
            assert args.metric_for_best_model == "eval_nDCG@10"

    def test_regularization_config(self):
        """Test regularization configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                regularization="flops_reg",
                q_regularization_weight=0.001,
                d_regularization_weight=0.001,
                regularization_warmup_steps=1000,
            )
            assert args.regularization == "flops_reg"
            assert args.q_regularization_weight == 0.001
            assert args.d_regularization_weight == 0.001
            assert args.regularization_warmup_steps == 1000

    def test_ir_metrics_config(self):
        """Test IR metrics configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = ["nDCG@10", "MRR"]
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                eval_metrics=metrics,
            )
            # eval_ir_metrics is computed in __post_init__
            assert args.eval_ir_metrics is not None
            assert len(args.eval_ir_metrics) == 2

    def test_training_args_validation(self):
        """Test that invalid configurations are caught."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid configuration should not raise
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=3,
                per_device_train_batch_size=32,
            )
            assert args.num_train_epochs == 3
            assert args.per_device_train_batch_size == 32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fp16")
    def test_fp16_config(self):
        """Test fp16 precision configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                fp16=True,
            )
            assert args.fp16 is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16")
    def test_bf16_config(self):
        """Test bf16 precision configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                bf16=True,
            )
            assert args.bf16 is True

    def test_learning_rate_config(self):
        """Test learning rate configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                learning_rate=1e-5,
            )
            assert args.learning_rate == 1e-5

    def test_warmup_steps_config(self):
        """Test warmup steps configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = RankerTrainingArguments(
                output_dir=tmpdir,
                warmup_steps=1000,
            )
            assert args.warmup_steps == 1000
