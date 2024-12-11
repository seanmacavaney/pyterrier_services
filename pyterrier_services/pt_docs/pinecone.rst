Pinecone
==================================================

`Pinecone <https://docs.pinecone.io/models/overview>`__ provides a Hosted Inference API to various embedding
and reranking models. ``pyterrier-services`` provides access to these APIs through
:class:`~pyterrier_services.PineconeApi`.

.. Note::

	To use this API, you will need to have the pinecone package installed (``pip install pinecone``)
	and have a `Pinecone API Key <https://docs.pinecone.io/guides/get-started/quickstart>`__. You can
	provide your API key through the environment variable ``PINECONE_API_KEY`` (preferred), or pass it
	to the constructor of :class:`~pyterrier_services.PineconeApi`.

Examples
--------------------------------

Learned Sparse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :caption: Indexing and retrieval with a Pinecone learned sparse model using :class:`pyterrier_pisa.PisaIndex`

   # Setup
   >>> from pyterrier_services import PineconeApi
   >>> from pyterrier_pisa import PisaIndex
   >>> pinecone = PineconeApi()
   >>> model = pinecone.sparse_model()
   >>> index = PisaIndex('my_index.pisa', stemmer='none')
   
   # Indexing
   >>> pipeline = model >> index
   >>> pipeline.index([
   ...   {'docno': 'doc1', 'text': 'PyTerrier: Declarative Experimentation in Python from BM25 to Dense Retrieval'},
   ...   {'docno': 'doc2', 'text': 'QPPTK@TIREx: Simplified Query Performance Prediction for Ad-Hoc Retrieval Experiments'},
   ... ])
   
   # Retrieval
   >>> pipeline = model >> index.quantized()
   >>> pipeline.search('pyterrier')
     qid      query          query_toks docno    score  rank
   0   1  Retrieval  {'retrieval': 1.0}  doc2  30900.0     0
   1   1  Retrieval  {'retrieval': 1.0}  doc1  29400.0     1

Dense
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :caption: Indexing and retrieval with a Pinecone dense model using :class:`pyterrier_dr.FlexIndex`

   # Setup
   >>> from pyterrier_services import PineconeApi
   >>> from pyterrier_dr import FlexIndex
   >>> pinecone = PineconeApi()
   >>> model = pinecone.dense_model()
   >>> index = FlexIndex('my_index.flex')
   
   # Indexing
   >>> pipeline = model >> index
   >>> pipeline.index([
   ...   {'docno': 'doc1', 'text': 'PyTerrier: Declarative Experimentation in Python from BM25 to Dense Retrieval'},
   ...   {'docno': 'doc2', 'text': 'QPPTK@TIREx: Simplified Query Performance Prediction for Ad-Hoc Retrieval Experiments'},
   ... ])
   
   # Retrieval
   >>> pipeline = model >> index.retriever()
   >>> pipeline.search('pyterrier')
     qid      query                                          query_vec docno  docid     score  rank
   0   1  pyterrier  [0.00923919677734375, -0.0171356201171875, -0....  doc1      0  0.814679     0
   1   1  pyterrier  [0.00923919677734375, -0.0171356201171875, -0....  doc2      1  0.722664     1

Re-Ranking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :caption: Re-Ranking results with Pinecone

   >>> import pandas as pd
   >>> from pyterrier_services import PineconeApi
   >>> pinecone = PineconeApi()
   >>> model = pinecone.reranker()
   >>> model(pd.DataFrame([
   ...   {'qid': '1', 'query': 'retrieval', 'docno': 'doc1', 'text': 'PyTerrier: Declarative Experimentation in Python from BM25 to Dense Retrieval'},
   ...    {'qid': '1', 'query': 'retrieval', 'docno': 'doc2', 'text': 'QPPTK@TIREx: Simplified Query Performance Prediction for Ad-Hoc Retrieval Experiments'},
   ]))
     qid      query docno                                               text     score  rank
   0   1  retrieval  doc2  QPPTK@TIREx: Simplified Query Performance Pred...  0.004811     0
   1   1  retrieval  doc1  PyTerrier: Declarative Experimentation in Pyth...  0.001598     1



API Documentation
--------------------------------

.. autoclass:: pyterrier_services.PineconeApi
   :members:

.. autoclass:: pyterrier_services.PineconeSparseModel
   :members:

.. autoclass:: pyterrier_services.PineconeDenseModel
   :members:

.. autoclass:: pyterrier_services.PineconeReranker
   :members:
