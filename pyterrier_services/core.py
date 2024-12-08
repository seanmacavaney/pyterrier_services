import requests
import pyterrier as pt
import pandas as pd


def http_error_retry(fn, retries=3):
    def wrapped(*args, **kwargs):
        ex = None
        for i in range(retries):
            try:
                return fn(*args, **kwargs)
            except requests.exceptions.HTTPError as e:
                ex = e
        if ex is not None:
            raise ex
    return wrapped


def paginated_search(fn, num_results):
    def wrapped(query):
        pages = []
        count = 0
        offset = 0
        while count < num_results and offset is not None:
            page, offset = fn(query, offset=offset, limit=num_results-count, return_next=True)
            pages.append(page)
            count += len(page)
        return pd.concat(pages, ignore_index=True)
    return wrapped


def multi_query(fn, verbose=True, verbose_desc='retrieving'):
    def wrapped(inp):
        it = inp.itertuples(index=False)
        if verbose:
            it = pt.tqdm(it, desc=verbose_desc, unit='q', total=len(inp))
        res = []
        for query in it:
            query_res = fn(query.query)
            query_res = query_res.assign(**{k: v for k, v in query._asdict().items() if k not in query_res.columns})
            res.append(query_res)

        df = pd.concat(res, ignore_index=True)

        desired_order = ["qid", "query", "docno", "score", "rank"]

        # Add any remaining columns not in the desired order
        remaining_columns = [col for col in df.columns if col not in desired_order]
        new_order = desired_order + remaining_columns

        # Reorder the dataframe
        return df[new_order]
    return wrapped
