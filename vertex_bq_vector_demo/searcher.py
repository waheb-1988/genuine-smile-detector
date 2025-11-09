"""Similarity search client for a deployed Vector Search index."""
from typing import List, Dict
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndexEndpoint
import vertexai
from vertexai.language_models import TextEmbeddingModel

class VectorSimilaritySearcher:
    def __init__(
        self,
        project_id: str,
        location: str,
        index_endpoint_name: str,
        deployed_index_id: str,
        embed_model_id: str = "gemini-embedding-001",
        k: int = 5,
    ):
        aiplatform.init(project=project_id, location=location)
        vertexai.init(project=project_id, location=location)
        self._endpoint = MatchingEngineIndexEndpoint(index_endpoint_name)
        self._deployed = deployed_index_id
        self._emb = TextEmbeddingModel.from_pretrained(embed_model_id)
        self.k = k

    def search(self, query_text: str):
        qv = self._emb.get_embeddings([query_text], task_type="RETRIEVAL_QUERY")[0].values
        res = self._endpoint.find_neighbors(
            deployed_index_id=self._deployed,
            queries=[qv],
            num_neighbors=self.k,
            return_full_datapoint=True,
        )
        out = []
        for n in res[0].neighbors:
            out.append({"id": n.id, "distance": n.distance})
        return out
