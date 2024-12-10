from typing import Optional, Literal
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

class PineconeApi:
    """Represents a reference to the Pinecone API.

    This class wraps :class:`pinecone.Pinecone`.
    """

    def __init__(self,
        api_key: Optional[str] = None
    ):
        """
        Args:
            api_key (str, optional): The Pinecone API key. Defaults to the value from ``PINECONE_API_KEY``.
        """
        from pinecone import Pinecone
        self.api_key = api_key
        self.pc = Pinecone(api_key=self.api_key)
        self.embed = self.pc.inference.embed

    def sparse_model(self,
        model_name: str = 'pinecone-sparse-english-v0',
    ) -> 'PineconeSparseModel':
        """Creates a PineconeSparseModel instance."""
        return PineconeSparseModel(model_name, api=self)


class PineconeSparseModel(pt.Transformer):
    """A PyTerrier transformer that provies access to a Pinecone sparse model."""
    def __init__(self,
        model_name: str = 'pinecone-sparse-english-v0',
        *,
        api: Optional[PineconeApi] = None,
    ):
        """
        Args:
            model_name (str): The name of the model.
            api (PineconeApi, optional): The Pinecone API object. Defaults to a new instance.
        """
        self.model_name = model_name
        self.api = api or PineconeApi()

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Encodes either queries or documents using this model (based on input columns)"""
        with pta.validate.any(inp) as v:
            v.query_frame(extra_columns=['query'], mode='query_encoder')
            v.document_frame(extra_columns=['text'], mode='doc_encoder')
        if v.mode == 'query_encoder':
            return self.query_encoder()(inp)
        if v.mode == 'doc_encoder':
            return self.doc_encoder()(inp)

    def query_encoder(self) -> 'PineconeSparseEncoder':
        """Creates a transformer that encodes queries using this model."""
        return PineconeSparseEncoder(self, input_type='query')

    def doc_encoder(self) -> 'PineconeSparseEncoder':
        """Creates a transformer that encodes documents using this model."""
        return PineconeSparseEncoder(self, input_type='passage')

    def __repr__(self):
        return f"PineconeSparseModel({self.model_name!r})"


class PineconeSparseEncoder(pt.Transformer):
    def __init__(self, sparse_model: PineconeSparseModel, *, input_type: Literal['passage', 'query']):
        self.sparse_model = sparse_model
        self.input_type = input_type

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        if self.input_type == 'passage':
            pta.validate.document_frame(inp, extra_columns=['text'])
            text = inp['text'].tolist()
            toks_field = 'toks'
        elif self.input_type == 'query':
            pta.validate.query_frame(inp, extra_columns=['query'])
            text = inp['query'].tolist()
            toks_field = 'query_toks'

        embeddings = self.sparse_model.api.embed(
            model=self.sparse_model.model_name,
            inputs=text,
            parameters={"input_type": self.input_type, "return_tokens": True}
        )
        assert embeddings.vector_type == 'sparse'
        toks = [dict(zip(v.sparse_tokens, v.sparse_values)) for v in embeddings.data]
        return inp.assign(**{toks_field: toks})

    def __repr__(self):
        return f"PineconeSparseEncoder({self.sparse_model!r}, input_type={self.input_type!r})"
