# pyterrier_services

PyTerrier components for online retrieval services.

## SemanticScholar

[Semantic Scholar](https://www.semanticscholar.org/me/research) is a scientific literature
search engine provided by the [Allen Institute for AI](https://allenai.org/).

`SemanticScholar()` provides access to the [search API](https://www.semanticscholar.org/product/api).

Example:

```python
>>> import pyterrier as pt ; pt.init()
>>> from pyterrier_services import SemanticScholar
>>> service = SemanticScholar()
>>> retriever = service.retriever()
>>> retriever.search('PyTerrier')
#                                     docno                                              title                                           abstract  rank  score qid      query
#  7fa92ed08eee68a945884b8744e7db9887aed9d3  PyTerrier: Declarative Experimentation in Pyth...  PyTerrier is a Python-based retrieval framewor...     0      0   1  PyTerrier
#  a6b1126e058262c57d36012d0fdedc2417ad04e1  Declarative Experimentation in Information Ret...  The advent of deep machine learning platforms ...     1     -1   1  PyTerrier
#  73d1f9eb421b5cca4c32f6a380d880839c356dd4  PyTerrier-based Research Data Recommendations ...  Research data is of high importance in scienti...     2     -2   1  PyTerrier
#  12c9b48d013255248378f23b7078e1788b5b1ef6  Axiomatic Retrieval Experimentation with ir_ax...  Axiomatic approaches to information retrieval ...     3     -3   1  PyTerrier
#  b7da554d9f1f51e13a852ab0270dcd0d824c52e8                        A Python Interface to PISA!  PISA (Performant Indexes and Search for Academ...     4     -4   1  PyTerrier
#  6659b3daabfb7e8e6dd8c4f47e2a774816888a9d  Retrieving Comparative Arguments using Ensembl...  In this paper, we present a submission to the ...     5     -5   1  PyTerrier
#  e57c05d3eb9c2d32332dc539d32e78f2b1fb05a6  University of Glasgow Terrier Team and UFMG at...  For TREC 2020, we explore different re-ranking...     6     -6   1  PyTerrier
#  81ec8a40deb82470438d978b013a0f6094ec8843  IR From Bag-of-words to BERT and Beyond throug...  The task of adhoc search is undergoing a renai...     7     -7   1  PyTerrier
```
