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

# ================== Logging Setup ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== CONFIGURATION ==================
VECTOR_DB = "bigquery"
PROJECT_ID = "ooredoo-oman-ai"
REGION = "us-central1"
BQ_DATASET = "cx_data"
BQ_TABLE = "survey_embeddings"

# We KEEP all rows (as in your original: 100000)
TOP_K = 100000             # ensures we retrieve all possible rows
MAX_ROWS_FOR_LLM = 50      # sample only for summary
MAX_ROWS_IN_TABLE = 60     # sample only for display in chat

logger.info(f"🔧 Configuration: Using {VECTOR_DB.upper()} as vector database")

# ================== Vertex AI Configuration ==================
vertexai.init(project=PROJECT_ID, location=REGION)
llm_model = GenerativeModel("gemini-2.5-pro")
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# ================== BigQuery Client ==================
bq_client = bigquery.Client(project=PROJECT_ID)

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

    # Ensure index is a normal column for readability
    if display_df.index.name or not isinstance(display_df.index, pd.RangeIndex):
        display_df = display_df.reset_index()

    try:
        table_md = display_df.to_markdown(index=False)
    except Exception:
        # Fallback: plain text
        text = display_df.to_string(index=False)
        table_md = f"```text\n{text}\n```"

    if truncated:
        table_md += f"\n\n_Only showing first {max_rows} of {len(df)} rows._"

    return table_md

# ================== Agent 1: Retrieve & Convert to DF ==================
def agent_retrieve_and_df(state: AgentState) -> AgentState:
    start = time.time()
    state["vector_db_used"] = VECTOR_DB

    try:
        # ---- Embedding ----
        emb = embed_model.get_embeddings([state["question"]])
        state["embedding"] = emb[0].values

        embedding_str = "[" + ",".join(map(str, state["embedding"])) + "]"

        # 👉 Keep TOP_K=100000 so you get **all rows** for calculations
        search_query = f"""
        SELECT base.*, distance FROM VECTOR_SEARCH(
            TABLE `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`,
            'embedding',
            (SELECT {embedding_str} AS embedding),
            distance_type => 'COSINE',
            top_k => {TOP_K}
        )
        """

        logger.info(f"🔍 Running BigQuery VECTOR_SEARCH with top_k={TOP_K}")
        results_df = bq_client.query(search_query).to_dataframe()

        if results_df.empty:
            state["df"] = pd.DataFrame()
            state["docs"] = []
            state["metadatas"] = []
            state["retriever_info"] = {"total_retrieved": 0, "top_samples": []}
            state["text_column_name"] = None
        else:
            # Drop embedding + distance from df we hold in memory
            metadatas = results_df.drop(
                columns=["embedding", "distance"],
                errors="ignore"
            ).to_dict("records")

            df = pd.DataFrame(metadatas)

            # Try to detect a text column automatically
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
            state["metadatas"] = metadatas
            state["retriever_info"] = {
                "total_retrieved": len(df),
                "top_samples": df.head(5).to_dict("records"),
                "database": VECTOR_DB,
            }
            state["text_column_name"] = text_column_name

            # ⚡ Docs only sampled for LLM to save tokens
            sample_df_for_docs = df.head(200)
            state["docs"] = sample_df_for_docs.astype(str).apply(
                lambda x: " | ".join(x),
                axis=1
            ).tolist()

        logger.info(f"✅ Retrieval + DataFrame conversion done: {len(state.get('df', []))} rows")

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
    """
    Use LLM to turn the natural-language question into a pandas expression.
    If you already have your own refine_query, you can replace this.
    """
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
    """
    Applies the refined pandas query to the DataFrame and stores result.
    Uses FULL df for calc (no sampling here).
    """
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
            # Execute safely – very restricted environment
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
    """
    Formats answer nicely in Markdown and generates summary & follow-ups using LLM.
    Uses only a sample of df_result for the LLM – FULL df_result is still stored.
    """
    start = time.time()

    try:
        df_result = state.get("df_result", pd.DataFrame())
        base_answer_parts: List[str] = []

        # ---- Main result table (Markdown) ----
        if df_result is not None and not df_result.empty and not str(state.get("final_result", "")).startswith("⚠️"):
            table_md = df_to_markdown_table(df_result)
            base_answer_parts.append("### 📊 Query Result\n\n" + table_md)
        else:
            if state.get("final_result"):
                base_answer_parts.append(str(state["final_result"]))
            else:
                base_answer_parts.append("⚠️ No result to display.")

        # ---- LLM summary + suggestions (on a sample of rows) ----
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

        # ---- Maintain conversation history ----
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
