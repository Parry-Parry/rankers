Quick Start
===========

This guide will help you get started with rankers.

Basic Usage
-----------

Dot Model (Bi-Encoder)
^^^^^^^^^^^^^^^^^^^^^^

The Dot model uses separate encoders for queries and documents::

    from rankers.modelling import Dot

    # Load a pre-trained model
    model = Dot.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Encode queries and documents
    queries = ["What is information retrieval?"]
    docs = ["Information retrieval is the process of finding relevant documents."]

    # Score query-document pairs
    scores = model.score(queries, docs)
    print(scores)

Cat Model (Cross-Encoder)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Cat model concatenates queries and documents::

    from rankers.modelling import Cat

    model = Cat.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = model.score(queries, docs)

Training a Model
----------------

Basic Training Example::

    from rankers.modelling import Dot
    from rankers.datasets import TrainingDataset
    from rankers.train import RankerTrainer, RankerTrainingArguments

    # Load dataset
    train_dataset = TrainingDataset.from_json("train.json")

    # Initialize model
    model = Dot.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Set training arguments
    args = RankerTrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
    )

    # Create trainer
    trainer = RankerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    # Train
    trainer.train()

PyTerrier Integration
---------------------

Use rankers with PyTerrier pipelines::

    from rankers.pyterrier import DotTransformer
    import pyterrier as pt

    pt.init()

    # Create a ranking pipeline
    ranker = DotTransformer("sentence-transformers/all-MiniLM-L6-v2")
    pipeline = pt.BatchRetrieve(index) >> ranker

    # Search
    results = pipeline.search("information retrieval")
