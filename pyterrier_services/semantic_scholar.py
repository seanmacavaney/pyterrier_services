from typing import List, Optional, Union, Tuple
from functools import partial
import pandas as pd
import requests
import pyterrier as pt
from . import http_error_retry, paginated_search, multi_query

class SemanticScholarApi:
    """Represents a reference to the Semantic Scholar search API."""
    API_BASE_URL = 'https://api.semanticscholar.org/graph/v1'

    def retriever(self,
        *,
        num_results: int = 100,
        fields: List[str] = ['title', 'abstract'],
        verbose: bool = True
    ) -> pt.Transformer:
        """Returns a :class:`~pyterrier.Transformer` that retrieves articles from Semantic Scholar.

        Args:
            num_results: The number of results to retrieve. Defaults to 100.
            fields: The fields to include in the retrieved results. Defaults to ['title', 'abstract'].
            verbose: Whether to log the progress. Defaults to True.
        """
        return SemanticScholarRetriever(api=self, num_results=num_results, fields=fields, verbose=verbose)

    def search(self,
        query: str,
        *,
        offset: int = 0,
        limit: int = 100,
        fields: List[str] = ['title', 'abstract'],
        return_next: bool = False,
        return_total: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int], Tuple[pd.DataFrame, int, int]]:
        """Searches for papers on Semantic Scholar with the provided query.

        Args:
            query: The search query.
            offset: The offset of the first result to retrieve. Defaults to 0.
            limit: The maximum number of results to retrieve. Defaults to 100.
            fields: The fields to include in the retrieved results. Defaults to ['title', 'abstract'].
            return_next: Whether to return the next query URL. Defaults to False.
            return_total: Whether to return the total number of results. Defaults to False.
        """
        params = {
            'query': query,
            'offset': offset,
            'fields': ','.join(fields),
            'limit': max(min(limit, 100), 1),
        }
        http_res = requests.get(SemanticScholarApi.API_BASE_URL + '/paper/search', params=params)
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


class SemanticScholarRetriever(pt.Transformer):
    """A :class:`~pyterrier.Transformer` retriever that queries the Semantic Scholar search API."""
    def __init__(self,
        *,
        api: Optional[SemanticScholarApi] = None,
        num_results: int = 100,
        fields: List[str] = ['title', 'abstract'],
        verbose: bool = True
    ):
        """
        Args:
            api: The Semantic Scholar api service. Defaults to a new instance of :class:`~pyterrier_services.SemanticScholarApi`.
            num_results: The number of results to retrieve per query. Defaults to 100.
            fields: The fields to include in the retrieved results. Defaults to ['title', 'abstract'].
            verbose: Whether to log the progress. Defaults to True.
        """
        self.api = api or SemanticScholarApi()
        self.num_results = num_results
        self.fields = fields
        self.verbose = verbose

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        return multi_query(
            paginated_search(
                http_error_retry(
                    partial(self.api.search, fields=self.fields)
                ),
                num_results=self.num_results,
            ),
            verbose=self.verbose,
            verbose_desc='SemanticScholarRetriever',
        )(inp)

    def fuse_rank_cutoff(self, k: int) -> Optional['SemanticScholarRetriever']:
        if k < self.num_results:
            return SemanticScholarRetriever(api=self.api, num_results=k, fields=self.fields, verbose=self.verbose)
