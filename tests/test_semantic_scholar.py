import unittest
import pandas as pd
from pyterrier_services import SemanticScholarApi

class TestSemanticScholar(unittest.TestCase):
    def test_retriever(self):
        s2 = SemanticScholarApi()
        retr = s2.retriever(num_results=10)
        res = retr.search('PyTerrier')
