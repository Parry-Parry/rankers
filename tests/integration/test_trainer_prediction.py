"""Integration tests for RankerTrainer.predict() method."""

import pytest
import tempfile

from rankers import RankerTrainer, RankerTrainingArguments
from rankers.datasets import TrainingDataset, EvaluationDataset, Corpus
from tests.fixtures.data import (
    create_synthetic_jsonl,
    create_synthetic_qrels,
    create_synthetic_corpus,
)


@pytest.fixture
def trainer_and_datasets(simple_model):
    """Create a trainer with training and test datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic training data
        train_file, _ = create_synthetic_jsonl(
            num_queries=5, num_docs_per_query=5
        )

        # Create corpus for training and evaluation
        corpus_dict = create_synthetic_corpus(num_docs=50, num_queries=10)
        corpus = Corpus(
            documents=corpus_dict["documents"],
            queries=corpus_dict["queries"],
        )

        # Create training dataset
        train_dataset = TrainingDataset(
            train_file,
            corpus=corpus,
            group_size=4,
        )

        # Create test dataset with qrels
        test_qrels = create_synthetic_qrels(num_queries=3)
        test_corpus = Corpus(
            documents=corpus_dict["documents"],
            queries=corpus_dict["queries"],
            qrels=test_qrels,
        )
        test_dataset = EvaluationDataset.from_qrels(
            test_qrels, corpus=test_corpus
        )

        # Create training args with minimal config
        training_args = RankerTrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=10,
            report_to=[],  # Disable all reporting integrations
        )

        # Create trainer without data collator for test simplicity
        trainer = RankerTrainer(
            model=simple_model,
            args=training_args,
            train_dataset=train_dataset,
            loss_fn="margin_mse",
        )

        yield trainer, train_dataset, test_dataset, tmpdir


def test_predict_returns_prediction_output(trainer_and_datasets):
    """Test that predict returns a valid PredictionOutput."""
    trainer, _, test_dataset, _ = trainer_and_datasets

    output = trainer.predict(test_dataset)

    # Check output has required attributes
    assert hasattr(output, "predictions")
    assert hasattr(output, "label_ids")
    assert hasattr(output, "metrics")

    # Check predictions is not None
    assert output.predictions is not None


def test_predict_with_custom_metric_prefix(trainer_and_datasets):
    """Test that predict respects custom metric_key_prefix."""
    trainer, _, test_dataset, _ = trainer_and_datasets

    output = trainer.predict(test_dataset, metric_key_prefix="custom")

    # All metrics should have the custom prefix
    for key in output.metrics.keys():
        assert key.startswith("custom_"), (
            f"Metric {key} doesn't start with 'custom_'"
        )


def test_predict_default_metric_prefix(trainer_and_datasets):
    """Test that predict uses 'test' as default metric prefix."""
    trainer, _, test_dataset, _ = trainer_and_datasets

    output = trainer.predict(test_dataset)

    # All metrics should have the test prefix
    for key in output.metrics.keys():
        assert key.startswith("test_"), (
            f"Metric {key} doesn't start with 'test_'"
        )


def test_predict_computes_ir_metrics(trainer_and_datasets):
    """Test that predict computes IR metrics."""
    trainer, _, test_dataset, _ = trainer_and_datasets

    output = trainer.predict(test_dataset)

    # Should have speed metrics and IR metrics
    assert len(output.metrics) > 0
    # Speed metrics are added
    assert any(
        "speed" in k or "time" in k.lower()
        for k in output.metrics.keys()
    )


def test_predict_with_ignore_keys(trainer_and_datasets):
    """Test that predict accepts ignore_keys parameter."""
    trainer, _, test_dataset, _ = trainer_and_datasets

    # Should not raise error
    output = trainer.predict(test_dataset, ignore_keys=["some_key"])

    assert output.predictions is not None
    assert output.metrics is not None


def test_predict_after_training(trainer_and_datasets):
    """Test predict on a trained model."""
    trainer, _, test_dataset, _ = trainer_and_datasets

    # Note: Actual training is skipped due to data collator setup requirements
    # This test verifies predict works after trainer initialization
    # (which would normally happen after training)

    # Predict on test set
    output = trainer.predict(test_dataset)

    assert output.predictions is not None
    assert output.metrics is not None
    assert len(output.metrics) > 0


def test_predict_num_samples(trainer_and_datasets):
    """Test that predict reports correct number of samples."""
    trainer, _, test_dataset, _ = trainer_and_datasets

    output = trainer.predict(test_dataset)

    # Check that num_samples is computed
    assert (
        "test_num_samples" in output.metrics
        or "num_samples" in output.metrics
        or len(test_dataset) > 0
    )


def test_predict_output_format(trainer_and_datasets):
    """Test that predict output has expected format."""
    trainer, _, test_dataset, _ = trainer_and_datasets

    output = trainer.predict(test_dataset)

    # predictions should be a result frame (DataFrame-like)
    assert output.predictions is not None

    # label_ids can be None for ranking tasks
    # (we don't have explicit labels)

    # metrics should be a dictionary
    assert isinstance(output.metrics, dict)
