# Training Neural Retrievers

Welcome to the `rankers` repository! This package provides tools and examples for training neural retrievers using various ranking models. Our goal is to help you develop state-of-the-art information retrieval systems with ease.

## Features

- **Flexible Model Training**: Use the `RankerTrainer` class to train different ranking models like BERT.
- **Data Handling**: Supports various data formats and efficient data loading with the `TrainingDataset` and `CatDataCollator`.
- **Transformers Integration**: Built on top of the Hugging Face `transformers` library for easy model loading and training.
- **Optimization and Scheduling**: Utilize advanced optimizers like `AdamW` and learning rate schedulers.
- **Experiment Tracking**: Integrated with `wandb` for tracking experiments and hyperparameters.

## Install 
```
pip install rankers
```
### Latest (Unstable)
```
pip install -U git+https://github.com/Parry-Parry/rankers.git
```

## Quick Start

To get started, check out the examples in the `examples` directory. Here's a brief overview of what you can find:

- **`train.bert.cat.py`**: An example script for training a BERT-based ranking model. It includes setting up data arguments, model arguments, and training arguments, and demonstrates how to train and save the model.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Parry-Parry/rankers.git
cd rankers
pip install -r requirements.txt
```

## Usage

Models can be instantiated like `transformers` models, for example, a bi-encoder (dot)

```
model = Dot.from_pretrained("bert-base-uncased")
```

Models are converted into rankers in PyTerrier:


```
ranker = model.to_pyterrier()
```

Run the example training script:

```bash
python examples/train.bert.cat.py --model_name_or_path bert-base-uncased --training_data path/to/data --output_dir path/to/save/model
```

## Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Happy training!

The core of Rankers is the RankerTrainer class, which inherits functionality from the Transformers trainer.

