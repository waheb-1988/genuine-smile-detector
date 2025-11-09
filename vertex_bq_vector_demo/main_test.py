"""End-to-end smoke test.
1) Create demo BQ table with sample data
2) Create & deploy empty Vector Search index
3) Index 'comments' from BQ (with deterministic doc_id in SQL)
4) Run a small similarity search
5) Run the SQL agent for monthly averages
"""
import os
from simulate_bq import create_demo_table
from index_setup import create_and_deploy_index
from indexer import BigQueryVectorIndexer
from searcher import VectorSimilaritySearcher
from sql_agent import average_satisfaction_per_month

# --------- EDIT THESE CONSTANTS ---------
PROJECT_ID = os.environ.get("GCP_PROJECT", "your-gcp-project")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
DATASET = "demo_satisfaction"
TABLE = "satisfaction_survey"
INDEX_DISPLAY_NAME = "svy-index"
ENDPOINT_DISPLAY_NAME = "svy-index-endpoint"
DEPLOYED_INDEX_ID = "svy-deployed-index"
EMBED_MODEL = "gemini-embedding-001"
DIMENSION = 3072
# ----------------------------------------

def main():
    # 1) Create demo data
    full_table = create_demo_table(PROJECT_ID, DATASET, TABLE)
    print("Demo table:", full_table)

    # 2) Create & deploy an EMPTY index
    index_res, endpoint_res, deployed_id = create_and_deploy_index(
        project_id=PROJECT_ID,
        location=LOCATION,
        index_display_name=INDEX_DISPLAY_NAME,
        endpoint_display_name=ENDPOINT_DISPLAY_NAME,
        dimension=DIMENSION,
        deployed_index_id=DEPLOYED_INDEX_ID,
        public_endpoint_enabled=True,
    )
    print("Index:", index_res)
    print("Endpoint:", endpoint_res)
    print("Deployed Index ID:", deployed_id)

    # 3) Index comments from BigQuery with deterministic doc_id from SQL
    select_sql = f"""
    SELECT
      TO_HEX(SHA256(CONCAT(CAST(event_date AS STRING), "|", COALESCE(region,""), "|", COALESCE(comments,"")))) AS doc_id,
      comments
    FROM `{full_table}`
    WHERE comments IS NOT NULL
    """
    indexer = BigQueryVectorIndexer(
        project_id=PROJECT_ID,
        location=LOCATION,
        index_endpoint_name=endpoint_res,
        deployed_index_id=DEPLOYED_INDEX_ID,
        embed_model_id=EMBED_MODEL,
        id_col="doc_id",
        text_cols=["comments"],
        batch_size=64,
    )
    total = indexer.index_with_select(select_sql)
    print(f"Indexed rows: {total}")

    # 4) Quick similarity search
    searcher = VectorSimilaritySearcher(
        project_id=PROJECT_ID,
        location=LOCATION,
        index_endpoint_name=endpoint_res,
        deployed_index_id=DEPLOYED_INDEX_ID,
        embed_model_id=EMBED_MODEL,
        k=3,
    )
    print("Neighbors:", searcher.search("negative feedback about support"))

    # 5) SQL agent: monthly average
    df = average_satisfaction_per_month(PROJECT_ID, full_table)
    print("\nAverage satisfaction per month:")
    print(df)

if __name__ == "__main__":
    main()
