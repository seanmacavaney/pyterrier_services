Pinecone
==================================================

`Pinecone <https://www.pinecone.io/>`__ provides various proprietary embedding models for search.

``pyterrier-services`` provides access to the Pinecone API through
:class:`~pyterrier_services.PineconeApi`.

.. Note::

	To use this API, you will need to have the pinecone package installed (``pip install pinecone``)
	and have a `Pinecone API Key <https://docs.pinecone.io/guides/get-started/quickstart>`__. You can
	provide your API key through the environment variable ``PINECONE_API_KEY`` (preferred), or pass it
	to the constructor of :class:`~pyterrier_services.PineconeApi`.

.. autoclass:: pyterrier_services.PineconeApi
   :members:

.. autoclass:: pyterrier_services.PineconeSparseModel
   :members:
