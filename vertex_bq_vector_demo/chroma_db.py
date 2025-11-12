import pandas as pd
from google.cloud import aiplatform
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel
import vertexai
import json
import math
import time

# ---------------- Configuration ----------------
PROJECT_ID = "ooredoo-oman-ai"
REGION = "me-central1"
MODEL_ID = "gemini-embedding-001"
GCS_BUCKET_NAME = "your-vector-search-bucket"  # Replace with your bucket
GCS_EMBEDDINGS_PATH = "embeddings/survey_embeddings.jsonl"
INDEX_DISPLAY_NAME = "survey-vector-index"
INDEX_ENDPOINT_DISPLAY_NAME = "survey-vector-endpoint"

# Initialize
aiplatform.init(project=PROJECT_ID, location=REGION)
vertexai.init(project=PROJECT_ID, location=REGION)
model = TextEmbeddingModel.from_pretrained(MODEL_ID)

# ---------------- Step 1: Prepare Embeddings for Vector Search ----------------
def create_embeddings_jsonl(df, output_file="embeddings.jsonl"):
    """
    Create embeddings in JSONL format required by Vertex AI Vector Search
    Format: {"id": "...", "embedding": [...], "restricts": {...}}
    """
    df["Survey_Date"] = df["Survey_Date"].astype(str)
    documents = df["Survey_Response"].tolist()
    
    BATCH_SIZE = 512
    num_batches = math.ceil(len(df) / BATCH_SIZE)
    
    print(f"⚙️ Creating embeddings for {len(df)} documents...")
    
    all_embeddings = []
    
    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, len(df))
        batch_docs = documents[start:end]
        
        # Get embeddings
        embeddings_obj = model.get_embeddings(batch_docs)
        batch_embeddings = [e.values for e in embeddings_obj]
        
        # Create JSONL entries
        for idx, (doc_idx, embedding) in enumerate(zip(range(start, end), batch_embeddings)):
            row = df.iloc[doc_idx]
            
            entry = {
                "id": str(doc_idx),
                "embedding": embedding,
                "restricts": [
                    {"namespace": "survey_category", "allow": [str(row["Survey_Category"])]},
                    {"namespace": "gender", "allow": [str(row["GENDER"])]},
                    {"namespace": "age_group", "allow": [str(row["AGE_GROUP"])]},
                    {"namespace": "customer_segment", "allow": [str(row["CUSTOMER_SEGMENT"])]},
                    {"namespace": "region", "allow": [str(row["CUSTOMER_REGION"])]}
                ]
            }
            all_embeddings.append(entry)
        
        print(f"✅ Batch {i+1}/{num_batches} processed")
        time.sleep(1)
    
    # Write to JSONL file
    with open(output_file, 'w') as f:
        for entry in all_embeddings:
            f.write(json.dumps(entry) + '\n')
    
    print(f"💾 Saved {len(all_embeddings)} embeddings to {output_file}")
    return output_file

# ---------------- Step 2: Upload to GCS ----------------
def upload_to_gcs(local_file, bucket_name, gcs_path):
    """Upload embeddings file to GCS"""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    blob.upload_from_filename(local_file)
    gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    print(f"📤 Uploaded to {gcs_uri}")
    return gcs_uri

# ---------------- Step 3: Create Vector Search Index ----------------
def create_vector_index(gcs_uri, dimensions=768):
    """
    Create Vertex AI Vector Search Index
    Note: Index creation can take 1-2 hours for large datasets
    """
    print(f"🔨 Creating Vector Search Index...")
    
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        contents_delta_uri=gcs_uri,
        dimensions=dimensions,  # 768 for text-embedding-004, check your model
        approximate_neighbors_count=10,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        description="Survey embeddings from Vertex AI",
    )
    
    print(f"✅ Index created: {index.resource_name}")
    return index

# ---------------- Step 4: Create Index Endpoint ----------------
def create_index_endpoint():
    """Create an endpoint to deploy the index"""
    print(f"🌐 Creating Index Endpoint...")
    
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=INDEX_ENDPOINT_DISPLAY_NAME,
        public_endpoint_enabled=True,
        description="Endpoint for survey vector search"
    )
    
    print(f"✅ Endpoint created: {endpoint.resource_name}")
    return endpoint

# ---------------- Step 5: Deploy Index to Endpoint ----------------
def deploy_index(index, endpoint):
    """Deploy the index to the endpoint"""
    print(f"🚀 Deploying index to endpoint...")
    
    endpoint.deploy_index(
        index=index,
        deployed_index_id="survey_deployed_index",
        display_name="Survey Vector Index Deployment",
        min_replica_count=1,
        max_replica_count=2,
    )
    
    print(f"✅ Index deployed successfully!")

# ---------------- Step 6: Query the Index ----------------
def query_vector_index(endpoint, query_text, num_neighbors=5):
    """Query the deployed vector index"""
    # Get query embedding
    query_embedding = model.get_embeddings([query_text])[0].values
    
    # Query the index
    response = endpoint.find_neighbors(
        deployed_index_id="survey_deployed_index",
        queries=[query_embedding],
        num_neighbors=num_neighbors
    )
    
    print(f"\n🔍 Query: '{query_text}'")
    print(f"Results:")
    for neighbor in response[0]:
        print(f"  - ID: {neighbor.id}, Distance: {neighbor.distance:.4f}")
    
    return response

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    
    # Step 1: Create embeddings JSONL
    embeddings_file = create_embeddings_jsonl(df)
    
    # Step 2: Upload to GCS
    gcs_uri = upload_to_gcs(embeddings_file, GCS_BUCKET_NAME, GCS_EMBEDDINGS_PATH)
    
    # Step 3: Create Vector Index (takes 1-2 hours)
    index = create_vector_index(gcs_uri, dimensions=768)
    
    # Step 4: Create Endpoint
    endpoint = create_index_endpoint()
    
    # Step 5: Deploy Index
    deploy_index(index, endpoint)
    
    # Step 6: Test Query
    query_vector_index(endpoint, "customer satisfaction with service")
    
    print(f"\n🎉 Vector Search deployment complete!")
    print(f"📍 Index: {index.resource_name}")
    print(f"📍 Endpoint: {endpoint.resource_name}")