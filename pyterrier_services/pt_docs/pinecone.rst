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

API Documentation
--------------------------------

.. autoclass:: pyterrier_services.PineconeApi
   :members:

.. autoclass:: pyterrier_services.PineconeSparseModel
   :members:

.. autoclass:: pyterrier_services.PineconeReranker
   :members:
