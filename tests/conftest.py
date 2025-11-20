"""Pytest configuration and shared fixtures for rankers tests."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.fixtures.data import (
    create_synthetic_corpus,
    create_synthetic_jsonl,
    create_synthetic_qrels,
    create_synthetic_trec,
)
from tests.fixtures.models import TinyDotModel


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def synthetic_jsonl():
    """Provide a synthetic JSONL training file."""
    jsonl_file, records = create_synthetic_jsonl(num_queries=10)
    yield jsonl_file, records
    # Cleanup
    Path(jsonl_file).unlink(missing_ok=True)


@pytest.fixture
def synthetic_trec():
    """Provide a synthetic TREC format file."""
    trec_file, df = create_synthetic_trec(num_queries=10)
    yield trec_file, df
    # Cleanup
    Path(trec_file).unlink(missing_ok=True)


@pytest.fixture
def synthetic_corpus():
    """Provide a synthetic corpus dictionary."""
    return create_synthetic_corpus(num_docs=50, num_queries=10)


@pytest.fixture
def synthetic_qrels():
    """Provide a synthetic qrels DataFrame."""
    return create_synthetic_qrels(num_queries=10)


@pytest.fixture
def simple_model():
    """Provide a tiny BERT ranking model for testing."""
    return TinyDotModel()


@pytest.fixture
def mock_eval_dataset():
    """Provide a mock evaluation dataset."""
    import pandas as pd

    dataset = Mock()
    dataset.data = pd.DataFrame({
        "qid": ["q0", "q0", "q1", "q1"],
        "docno": ["d0", "d1", "d2", "d3"],
        "score": [10.0, 5.0, 8.0, 3.0],
    })
    dataset.qrels = pd.DataFrame({
        "query_id": ["q0", "q1"],
        "doc_id": ["d0", "d2"],
        "relevance": [2, 2],
    })
    return dataset


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line(
        "markers", "requires_ir_datasets: Requires ir_datasets package"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their location."""
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
