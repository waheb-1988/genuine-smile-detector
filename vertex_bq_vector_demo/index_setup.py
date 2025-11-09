"""Create and deploy an EMPTY Vertex AI Vector Search index.
Run from an authenticated GCP environment (ADC), e.g. Vertex AI Workbench.

Usage (from main_test.py normally):
    from index_setup import create_and_deploy_index
"""
from typing import Tuple
from google.cloud import aiplatform

def create_and_deploy_index(
    project_id: str,
    location: str,
    index_display_name: str,
    endpoint_display_name: str,
    dimension: int = 3072,
    deployed_index_id: str = "deployed-index",
    public_endpoint_enabled: bool = True,
    distance_measure_type: str = "COSINE",
    approximate_neighbors_count: int = 150,
    machine_type: str = "e2-standard-2",
    min_replica_count: int = 1,
) -> Tuple[str, str, str]:
    """Creates an empty index, creates an endpoint, and deploys the index.
    Returns: (index_resource_name, endpoint_resource_name, deployed_index_id)
    """
    aiplatform.init(project=project_id, location=location)

    # 1) Create the index (empty for now)
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=index_display_name,
        dimensions=dimension,
        approximate_neighbors_count=approximate_neighbors_count,
        distance_measure_type=distance_measure_type,
    )
    index_resource = index.resource_name

    # 2) Create endpoint
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=endpoint_display_name,
        public_endpoint_enabled=public_endpoint_enabled,
    )
    endpoint_resource = endpoint.resource_name

    # 3) Deploy
    deployed = endpoint.deploy_index(
        index=index,
        deployed_index_id=deployed_index_id,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
    )

    return index_resource, endpoint_resource, deployed.deployed_index_id
