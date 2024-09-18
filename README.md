# Rankers
A small extension to transformers adding common IR loss components, easy wrappers and callbacks!

## Install 
```
pip install rankers
```
### Latest (Unstable)
```
pip install -U git+https://github.com/Parry-Parry/rankers.git
```

## Getting Started

Models can be instantiated like huggingface models, for example a bi-encoder (dot)

```
model = Dot.from_pretrained("bert-base-uncased")
```

Models are converted into rankers in PyTerrier:
```
ranker = model.to_pyterrier()
```

The core of Rankers is the RankerTrainer class, which inherits functionality from the Transformers trainer.

