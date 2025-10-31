.. rankers documentation master file

Welcome to rankers documentation!
==================================

**rankers** is a neural information retrieval framework providing modular implementations of
various ranking models including dot-product (bi-encoder), concatenation (cross-encoder),
sparse, and sequence-to-sequence models.

Features
--------

- Multiple ranking model architectures (Dot, Cat, Sparse, Seq2Seq)
- Integration with HuggingFace Transformers
- PyTerrier integration for IR pipelines
- Flexible training framework with various loss functions
- Support for different embedding pooling strategies

Quick Start
-----------

Installation::

    pip install rankers

Basic Usage::

    from rankers.modelling import Dot

    # Load a pre-trained model
    model = Dot.from_pretrained("model-name")

    # Encode queries and documents
    queries = ["What is information retrieval?"]
    docs = ["Information retrieval is the process of finding relevant documents."]

    scores = model.score(queries, docs)

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
