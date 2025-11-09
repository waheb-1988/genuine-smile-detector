"""Indexer: Read from BigQuery, embed text with Vertex AI, upsert to Vector Search."""
from typing import Iterable, List, Dict, Optional
import pandas as pd
from google.cloud import aiplatform, bigquery
from google.cloud.aiplatform import MatchingEngineIndexEndpoint
import vertexai
from vertexai.language_models import TextEmbeddingModel

class BigQueryVectorIndexer:
    def __init__(
        self,
        project_id: str,
        location: str,
        index_endpoint_name: str,
        deployed_index_id: str,
        embed_model_id: str = "gemini-embedding-001",
        id_col: Optional[str] = None,             # allow None: we'll hash the text if missing
        text_cols: Iterable[str] = ("comments",),
        batch_size: int = 128,
    ):
        self.project_id = project_id
        self.location = location
        self.id_col = id_col
        self.text_cols = list(text_cols)
        self.batch_size = batch_size

        aiplatform.init(project=project_id, location=location)
        vertexai.init(project=project_id, location=location)
        self._embed_model = TextEmbeddingModel.from_pretrained(embed_model_id)
        self._endpoint = MatchingEngineIndexEndpoint(index_endpoint_name)
        self._deployed_index_id = deployed_index_id

    def _concat_text(self, row: pd.Series) -> str:
        return " \n".join([str(row[c]) for c in self.text_cols if pd.notna(row[c])])

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        embs = self._embed_model.get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")
        return [e.values for e in embs]

    def _datapoints(self, ids: List[str], vectors: List[List[float]]):
        dps = []
        for i, vec in enumerate(vectors):
            dp = aiplatform.matching_engine.matching_engine_index_datapoint.MatchingEngineIndexDatapoint(
                datapoint_id=str(ids[i]), feature_vector=vec
            )
            dps.append(dp)
        return dps

    def index_with_select(self, select_sql: str) -> int:
        """Run any SELECT that returns at least the text columns; may also return an id col."""
        bq = bigquery.Client(project=self.project_id)
        df = bq.query(select_sql).result().to_dataframe(create_bqstorage_client=True)

        # Build IDs
        if not self.id_col or self.id_col not in df.columns:
            import hashlib
            ids = []
            for _, r in df.iterrows():
                s = self._concat_text(r)
                ids.append(hashlib.sha256(s.encode("utf-8")).hexdigest())
        else:
            ids = df[self.id_col].astype(str).tolist()

        texts = [self._concat_text(r) for _, r in df.iterrows()]

        # Batch upserts
        for start in range(0, len(texts), self.batch_size):
            tchunk = texts[start:start+self.batch_size]
            ichunk = ids[start:start+self.batch_size]
            vecs = self._embed_texts(tchunk)
            dps = self._datapoints(ichunk, vecs)
            self._endpoint.upsert_datapoints(
                deployed_index_id=self._deployed_index_id, datapoints=dps
            )
        return len(texts)
