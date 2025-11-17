# ================== Imports ==================
from typing import TypedDict, List, Optional, Any, Dict
import time
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from langgraph.graph import StateGraph
import json
import logging
from datetime import datetime
from google.cloud import bigquery
import numpy as np
import chromadb
from chromadb.config import Settings

# ================== Logging Setup ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== CONFIGURATION ==================
VECTOR_DB = "chroma"  # <– now using Chroma as vector store
PROJECT_ID = "ooredoo-oman-ai"
REGION = "us-central1"
BQ_DATASET = "cx_data"
BQ_TABLE = "survey_embeddings"

# For retrieval: how many neighbors to get from Chroma
TOP_K = 100000             # logically "all rows" for your survey
MAX_ROWS_FOR_LLM = 50      # sample for summary
MAX_ROWS_IN_TABLE = 60     # sample for display in chat

logger.info(f"🔧 Configuration: Using {VECTOR_DB.upper()} as vector database")

# ================== Vertex AI Configuration ==================
vertexai.init(project=PROJECT_ID, location=REGION)
llm_model = GenerativeModel("gemini-2.5-pro")
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# ================== BigQuery Client ==================
bq_client = bigquery.Client(project=PROJECT_ID)

# ================== Chroma Client & Globals ==================
# Persistent directory so index survives container restarts
CHROMA_PATH = "chroma_store"

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(allow_reset=True)
)
CHROMA_COLLECTION_NAME = "survey_embeddings"

collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# These will be filled in load_data_and_index()
DF_ALL: Optional[pd.DataFrame] = None
DATASET_SIZE: int = 0

# ================== State Definition ==================
class AgentState(TypedDict, total=False):
    question: str
    embedding: Optional[List[float]]
    docs: Optional[List[str]]
    metadatas: Optional[List[dict]]
    retriever_info: Optional[Dict[str, Any]]
    df: Optional[pd.DataFrame]
    refined_query: Optional[str]
    final_result: Optional[str]
    final_result_rows: Optional[int]
    df_result: Optional[pd.DataFrame]
    timing: Optional[dict]
    history: Optional[List[Dict[str, str]]]
    vector_db_used: Optional[str]
    text_column_name: Optional[str]
    is_text_analysis: Optional[bool]
    error: Optional[str]

# ================== Helper: Markdown table ==================
def df_to_markdown_table(df: pd.DataFrame, max_rows: int = MAX_ROWS_IN_TABLE) -> str:
    """
    Convert a DataFrame to a nice markdown table for chatbot.
    Uses only head(max_rows) for display but keeps full df for calculations.
    """
    if df is None or df.empty:
        return "_No rows returned._"

    display_df = df.copy()
    truncated = False
    if len(display_df) > max_rows:
        display_df = display_df.head(max_rows)
        truncated = True

    if display_df.index.name or not isinstance(display_df.index, pd.RangeIndex):
        display_df = display_df.reset_index()

    try:
        table_md = display_df.to_markdown(index=False)
    except Exception:
        text = display_df.to_string(index=False)
        table_md = f"```text\n{text}\n```"

    if truncated:
        table_md += f"\n\n_Only showing first {max_rows} of {len(df)} rows._"

    return table_md

# ================== Offline Load: BigQuery ➜ df_all & Chroma ==================
def load_data_and_index() -> None:
    """
    Read whole survey table once from BigQuery and build:
      - global DF_ALL (without embedding)
      - Chroma index with embeddings
    Called at module import/startup.
    """
    global DF_ALL, DATASET_SIZE, collection

    if DF_ALL is not None and DATASET_SIZE > 0:
        logger.info("ℹ️ DF_ALL already loaded, skipping reload.")
        return

    logger.info("⏳ Loading data from BigQuery into df_all...")
    query = f"SELECT * FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"
    df_raw = bq_client.query(query).to_dataframe()

    if "embedding" not in df_raw.columns:
        raise ValueError("❌ BigQuery table must contain an 'embedding' column.")

    # Extract embeddings and drop from df
    embeddings = df_raw["embedding"].apply(lambda v: list(v)).tolist()
    df = df_raw.drop(columns=["embedding"])

    # Create row_id to map back from Chroma ids
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(str)

    DF_ALL = df
    DATASET_SIZE = len(df)
    logger.info(f"✅ Loaded {DATASET_SIZE} rows from BigQuery.")

    # Build Chroma index only if collection is empty
    if collection.count() == 0:
        logger.info("⏳ Building Chroma index with embeddings...")
        ids = df["row_id"].tolist()
        # Minimal docs just to satisfy Chroma; real data kept in DF_ALL
        documents = [""] * len(ids)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents
        )
        logger.info(f"✅ Chroma index built with {collection.count()} vectors.")
    else:
        logger.info(f"ℹ️ Chroma collection already has {collection.count()} vectors; not rebuilding.")

# Load at startup / import
load_data_and_index()

# ================== Agent 1: Retrieve rows using Chroma ==================
def agent_retrieve_and_df(state: AgentState) -> AgentState:
    start = time.time()
    state["vector_db_used"] = VECTOR_DB

    try:
        if DF_ALL is None or DATASET_SIZE == 0:
            raise RuntimeError("DF_ALL is not loaded.")

        # ---- Question embedding ----
        emb_obj = embed_model.get_embeddings([state["question"]])[0]
        embedding_vec = emb_obj.values
        state["embedding"] = embedding_vec

        # ---- Chroma query ----
        # To mimic "all rows", we take min(TOP_K, DATASET_SIZE)
        n_results = min(TOP_K, DATASET_SIZE)
        logger.info(f"🔍 Chroma query with n_results={n_results}")

        result = collection.query(
            query_embeddings=[embedding_vec],
            n_results=n_results
        )

        id_list = result.get("ids", [[]])[0] if result.get("ids") else []

        if not id_list:
            logger.warning("⚠️ Chroma returned no ids; falling back to full DF_ALL.")
            df = DF_ALL.copy()
        else:
            # Map ids back to DF_ALL in rank order
            df = DF_ALL.set_index("row_id").loc[id_list].reset_index()

        # Detect text column (same logic as before)
        text_cols = [
            c for c in df.columns
            if c.lower() in ["survey_response", "response", "comment", "feedback", "text"]
        ]
        text_column_name = text_cols[0] if text_cols else None
        if text_column_name:
            df[text_column_name] = df[text_column_name].fillna("")

        # Optional numeric conversion
        if "Satisfaction_Level" in df.columns:
            df["Satisfaction_Level"] = pd.to_numeric(df["Satisfaction_Level"], errors="coerce")

        state["df"] = df
        state["metadatas"] = df.to_dict("records")
        state["retriever_info"] = {
            "total_retrieved": len(df),
            "top_samples": df.head(5).to_dict("records"),
            "database": VECTOR_DB,
        }
        state["text_column_name"] = text_column_name

        # Light docs for LLM
        sample_df_for_docs = df.head(200)
        state["docs"] = sample_df_for_docs.astype(str).apply(
            lambda x: " | ".join(x),
            axis=1
        ).tolist()

        logger.info(f"✅ Retrieval + DataFrame conversion done: {len(df)} rows")

    except Exception as e:
        logger.error(f"❌ Retrieval & DF conversion failed: {e}")
        state["error"] = str(e)
        state["df"] = pd.DataFrame()
        state["text_column_name"] = None

    state["timing"] = state.get("timing", {})
    state["timing"]["retrieve_and_df"] = round(time.time() - start, 2)
    return state

# ================== Agent 2: Refine Query ==================
def agent_refine_query(state: AgentState) -> AgentState:
    start = time.time()
    df = state.get("df")

    if df is None or df.empty:
        state["refined_query"] = None
        state["timing"] = state.get("timing", {})
        state["timing"]["refine_query"] = round(time.time() - start, 2)
        logger.warning("⚠️ No data available; skipping refine_query.")
        return state

    columns = list(df.columns)
    schema_description = "\n".join(
        f"- {col}: {str(df[col].dtype)}" for col in columns
    )

    prompt = f"""
You are a senior data analyst. The user asked a question about a pandas DataFrame `df`.

User question:
{state['question']}

The DataFrame `df` has the following columns and dtypes:
{schema_description}

Return ONLY a single valid pandas expression using `df` to answer the question.
Rules:
- Do NOT include backticks, quotes around the whole expression, or explanations.
- Do NOT assign to variables.
- Do NOT print().
- Use column names exactly as given.
- If you need counts, you can use `value_counts()`.
- Prefer returning a Series or DataFrame if possible.

Examples:
- df['AGE_GROUP'].value_counts()
- df.groupby('Gender')['Satisfaction_Level'].mean().sort_values(ascending=False)
- df[['AGE_GROUP', 'Satisfaction_Level']].groupby('AGE_GROUP').agg(count=('Satisfaction_Level','size'))

Now output ONLY the pandas expression.
"""

    try:
        resp = llm_model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 256, "temperature": 0.2}
        )
        expr_raw = (resp.text or "").strip()
        expr = expr_raw.splitlines()[0]
        expr = expr.replace("```python", "").replace("```", "").strip()

        state["refined_query"] = expr
        logger.info(f"🔍 Generated query: {expr}")

    except Exception as e:
        logger.error(f"❌ refine_query failed: {e}")
        state["refined_query"] = None
        state["error"] = f"refine_query failed: {e}"

    state["timing"] = state.get("timing", {})
    state["timing"]["refine_query"] = round(time.time() - start, 2)
    return state

# ================== Agent 3: Apply Query ==================
def agent_apply_query(state: AgentState) -> AgentState:
    start = time.time()
    df = state.get("df")
    expr = state.get("refined_query")

    try:
        if df is None or df.empty:
            state["final_result"] = "⚠️ No data available to run the query."
            state["final_result_rows"] = 0
            state["df_result"] = pd.DataFrame()
        elif not expr:
            state["final_result"] = "⚠️ No query generated."
            state["final_result_rows"] = 0
            state["df_result"] = pd.DataFrame()
        else:
            result = eval(
                expr,
                {"__builtins__": {}},
                {"df": df, "pd": pd, "np": np, "len": len}
            )

            if isinstance(result, pd.DataFrame):
                state["df_result"] = result
                state["final_result_rows"] = len(result)
            elif isinstance(result, pd.Series):
                state["df_result"] = result.to_frame()
                state["final_result_rows"] = len(state["df_result"])
            else:
                state["df_result"] = pd.DataFrame({"result": [result]})
                state["final_result_rows"] = 1

            logger.info("✅ Query applied successfully")

    except Exception as e:
        logger.error(f"❌ Query execution failed: {e}")
        state["error"] = str(e)
        state["final_result"] = f"⚠️ Query failed: {e}"
        state["final_result_rows"] = 0
        state["df_result"] = pd.DataFrame()

    state["timing"] = state.get("timing", {})
    state["timing"]["apply_query"] = round(time.time() - start, 2)
    return state

# ================== Agent 4: Format Answer ==================
def agent_format_answer(state: AgentState) -> AgentState:
    start = time.time()

    try:
        df_result = state.get("df_result", pd.DataFrame())
        base_answer_parts: List[str] = []

        # ---- Main result table ----
        if df_result is not None and not df_result.empty and not str(state.get("final_result", "")).startswith("⚠️"):
            table_md = df_to_markdown_table(df_result)
            base_answer_parts.append("### 📊 Query Result\n\n" + table_md)
        else:
            if state.get("final_result"):
                base_answer_parts.append(str(state["final_result"]))
            else:
                base_answer_parts.append("⚠️ No result to display.")

        # ---- LLM summary + suggestions ----
        if df_result is not None and not df_result.empty and not str(state.get("final_result", "")).startswith("⚠️"):
            sample_df = df_result.head(MAX_ROWS_FOR_LLM)
            sample_csv = sample_df.to_csv(index=False)

            prompt = f"""Analyze this query result and provide insights.

User Question: {state['question']}
Vector Database: {state.get('vector_db_used', 'unknown')}

Result sample as CSV:
{sample_csv}

Respond with valid JSON only:
{{
  "summary": "Brief summary in 15-20 words",
  "suggestions": [
    "First follow-up question",
    "Second follow-up question"
  ]
}}

No markdown, no explanations, only JSON.
"""

            try:
                resp = llm_model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": 256, "temperature": 0.25}
                )
                text = (resp.text or "").strip()
                text = text.replace("```json", "").replace("```", "").strip()

                parsed = json.loads(text)
                summary = parsed.get("summary", "")
                suggestions = parsed.get("suggestions", [])
            except Exception as parse_error:
                logger.warning(f"⚠️ JSON parsing failed in format_answer: {parse_error}, using fallback")
                raw = (resp.text or "") if "resp" in locals() else ""
                lines = [l.strip() for l in raw.splitlines() if l.strip()]
                summary = lines[0] if lines else "Analysis complete."
                suggestions = lines[1:3] if len(lines) > 1 else []

            sugg_text = "\n".join(f"- {s}" for s in suggestions[:2])
            db_info = state.get("vector_db_used", "unknown").upper()

            summary_block = (
                "\n\n---\n"
                "### 📝 Summary\n"
                f"{summary}\n\n"
                "### 🔎 Suggested Follow-up Questions\n"
                f"{sugg_text}\n\n"
                f"> 🗄️ Data source: **{db_info}**"
            )
            base_answer_parts.append(summary_block)

        # ---- Timing info ----
        timing = state.get("timing", {})
        if timing:
            timing_str = " | ".join(f"{k}: {v}s" for k, v in timing.items())
            base_answer_parts.append(f"\n<sub>⏱ {timing_str}</sub>")

        final_text = "\n".join(base_answer_parts)
        state["final_result"] = final_text

        # ---- History ----
        if "history" not in state or state["history"] is None:
            state["history"] = []

        state["history"].append({
            "question": state["question"],
            "answer": final_text,
            "database": state.get("vector_db_used", "unknown"),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error formatting answer: {e}")
        state["error"] = f"Answer formatting failed: {str(e)}"

    state["timing"] = state.get("timing", {})
    state["timing"]["format_answer"] = round(time.time() - start, 2)
    return state

# ================== Graph Setup & Compilation ==================
graph = StateGraph(AgentState)

graph.add_node("agent_retrieve_and_df", agent_retrieve_and_df)
graph.add_node("refine_query", agent_refine_query)
graph.add_node("apply_query", agent_apply_query)
graph.add_node("format_answer", agent_format_answer)

graph.add_edge("agent_retrieve_and_df", "refine_query")
graph.add_edge("refine_query", "apply_query")
graph.add_edge("apply_query", "format_answer")

graph.set_entry_point("agent_retrieve_and_df")
graph.set_finish_point("format_answer")

compiled_graph = graph.compile()
