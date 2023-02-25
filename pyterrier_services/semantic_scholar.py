from functools import partial
import pandas as pd
import requests
import pyterrier as pt
from . import http_error_retry, paginated_search, multi_query

class SemanticScholar:
    API_BASE_URL = 'https://api.semanticscholar.org/graph/v1'
    def __init__(self, verbose=True):
        self.verbose = verbose

    def retriever(self, num_results=100, fields=['title', 'abstract'], verbose=None):
        return SemanticScholar.Retriever(self, num_results=num_results, fields=fields, verbose=verbose)

    class Retriever(pt.Transformer):
        def __init__(self, service, num_results=100, fields=['title', 'abstract'], verbose=None):
            self.service = service
            self.num_results = num_results
            self.fields = fields
            self.verbose = verbose if verbose is not None else service.verbose 

        def transform(self, inp):
            return multi_query(
                paginated_search(
                    http_error_retry(
                        partial(self.service.search, fields=self.fields)
                    ),
                    num_results=self.num_results,
                ),
                verbose=self.verbose,
                verbose_desc='SemanticScholar.retriever',
            )(inp)

    def search(self, query, offset=0, limit=100, fields=['title', 'abstract'], return_next=False, return_total=False):
        params = {
            'query': query,
            'offset': offset,
            'fields': ','.join(fields),
            'limit': max(min(limit, 100), 1),
        }
        http_res = requests.get(SemanticScholar.API_BASE_URL + '/paper/search', params=params)
        http_res.raise_for_status()
        http_res = http_res.json()

        if len(http_res['data']) == 0:
            result_df = pd.DataFrame(columns=['docno', *[str(f) for f in fields], 'rank', 'score'])
        else:
            result_df = pd.DataFrame(http_res['data'])
            result_df.rename(columns={'paperId': 'docno'}, inplace=True)
            result_df['rank'] = range(http_res['offset'], http_res['offset']+len(result_df))
            result_df['score'] = -result_df['rank']

        res = [result_df]
        if return_next:
            res.append(http_res.get('next'))
        if return_total:
            res.append(http_res['total'])
        if len(res) == 1:
            return res[0]
        return tuple(res)
