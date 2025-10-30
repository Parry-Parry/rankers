Installation
============

Basic Installation
------------------

Install rankers using pip::

    pip install rankers

Development Installation
------------------------

For development, clone the repository and install in editable mode::

    git clone https://github.com/yourusername/rankers.git
    cd rankers
    pip install -e .

Optional Dependencies
---------------------

rankers has several optional dependencies for different features:

PyTerrier Integration::

    pip install rankers[pyterrier]

All optional dependencies::

    pip install rankers[all]

Requirements
------------

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.0+
