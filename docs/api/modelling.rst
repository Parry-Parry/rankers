Modelling
=========

Base Classes
------------

.. autoclass:: rankers.modelling.base.Ranker
   :members:
   :show-inheritance:

Dot (Bi-Encoder)
----------------

.. autoclass:: rankers.modelling.dot.dot.DotConfig
   :members:

.. autoclass:: rankers.modelling.dot.dot.Pooler
   :members:

.. autoclass:: rankers.modelling.dot.dot.Dot
   :members:
   :show-inheritance:

Cat (Cross-Encoder)
-------------------

.. autoclass:: rankers.modelling.cat.cat.CatConfig
   :members:

.. autoclass:: rankers.modelling.cat.cat.Cat
   :members:
   :show-inheritance:

Sparse Models
-------------

.. autoclass:: rankers.modelling.sparse.sparse.SparseConfig
   :members:

.. autofunction:: rankers.modelling.sparse.sparse.splade_max

.. autoclass:: rankers.modelling.sparse.sparse.Sparse
   :members:
   :show-inheritance:

Sequence-to-Sequence
--------------------

.. automodule:: rankers.modelling.seq2seq.seq2seq
   :members:
   :show-inheritance:

BGE Models
----------

.. automodule:: rankers.modelling.bge.bge
   :members:
   :show-inheritance:
