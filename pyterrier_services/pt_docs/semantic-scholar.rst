Semantic Scholar
==================================================

`Semantic Scholar <https://www.semanticscholar.org/>`__ is a search engine over academic
papers provided by the `Allen Institute for AI <http://allenai.org/>`__.

``pyterrier-services`` provides access to the Semantic Scholar search API through
:class:`~pyterrier_services.SemanticScholarRetriever`.

Example:

.. code-block:: python
	:caption: Retrieve from the Semantic Scholar API

	>>> from pyterrier_services import SemanticScholar
	>>> s2 = SemanticScholar()
	>>> retr = s2.retriever(num_results=5)
	>>> retr.search('pyterrier')
	# qid      query                                     docno  score  rank                                              title                                           abstract
	#   1  pyterrier  7fa92ed08eee68a945884b8744e7db9887aed9d3      0     0  PyTerrier: Declarative Experimentation in Pyth...  PyTerrier is a Python-based retrieval framewor...
	#   1  pyterrier  a6b1126e058262c57d36012d0fdedc2417ad04e1     -1     1  Declarative Experimentation in Information Ret...  The advent of deep machine learning platforms ...
	#   1  pyterrier  833b453c621099bccca028752aaa74262123706a     -2     2  PyTerrier-based Research Data Recommendations ...  Research data is of high importance in scienti...
	#   1  pyterrier  73feb5cfe491342d52d47e8817d113c072067306     -3     3      The Information Retrieval Experiment Platform  We integrate irdatasets, ir_measures, and PyTe...
	#   1  pyterrier  90b8a1adae2761e48c87fdeb68a595dc11161970     -4     4  QPPTK@TIREx: Simplified Query Performance Pred...  We describe our software submission to the ECI...


.. autoclass:: pyterrier_services.SemanticScholar
   :members:

.. autoclass:: pyterrier_services.SemanticScholarRetriever
   :members:
