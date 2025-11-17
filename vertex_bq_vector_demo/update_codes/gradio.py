import os
import json
import uuid
import gradio as gr
import pandas as pd

from agent_cx import compiled_graph, AgentState  # adjust name if needed

JSON_PATH = "chat_history.json"

# ------------------- JSON Load/Save -------------------
def load_json():
    if not os.path.exists(JSON_PATH):
        return {}
    try:
        with open(JSON_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_json(data):
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

chat_sessions = load_json()

# ------------------- Core Chat Pipeline -------------------
def chat_pipeline(message: str, session_id: str):
    """
    Runs the agent graph and returns:
    - chat_pairs: list of {role, content} for Gradio Chatbot (Markdown)
    - session_id
    - df_result: DataFrame with full result
    - timing: dict with timings
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    state: AgentState = {
        "question": message,
        "history": chat_sessions[session_id],
    }

    result_state: AgentState = compiled_graph.invoke(state)
    final_result = result_state.get("final_result", "⚠️ No final result generated.")
    df_result = result_state.get("df_result", pd.DataFrame())
    timing = result_state.get("timing", {})

    # Persist in session JSON
    chat_sessions[session_id].append({"role": "user", "content": message})
    chat_sessions[session_id].append({"role": "assistant", "content": final_result})
    save_json(chat_sessions)

    chat_pairs = [
        {"role": e.get("role", "assistant"), "content": e.get("content", "")}
        for e in chat_sessions[session_id]
        if isinstance(e, dict) and "role" in e and "content" in e
    ]

    return chat_pairs, session_id, df_result, timing

# ------------------- New Chat -------------------
def new_chat():
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = []
    save_json(chat_sessions)
    return [], session_id, pd.DataFrame(), {}

# ------------------- Load History -------------------
def load_history(session_id):
    if session_id in chat_sessions:
        entries = chat_sessions[session_id]
        chat_pairs = [
            {"role": e["role"], "content": e["content"]}
            for e in entries
            if "role" in e and "content" in e
        ]
        return chat_pairs, session_id, pd.DataFrame(), {}
    return [], session_id, pd.DataFrame(), {}

# ------------------- Delete History -------------------
def delete_history(session_id):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        save_json(chat_sessions)
    return [], "", pd.DataFrame(), {}

# ------------------- Frontend Helpers -------------------
def echo_user(message, history):
    """
    Step 1: Immediately show the user message in chat.
    """
    if history is None:
        history = []
    history = history + [{"role": "user", "content": message}]
    return "", history

def bot_reply(history, session_id):
    """
    Step 2: Run the heavy pipeline and replace chat history with
    full conversation (including formatted result and table).
    """
    if not history:
        return history, session_id, pd.DataFrame(), {}

    last_user_msg = None
    for m in reversed(history):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    if not last_user_msg:
        return history, session_id, pd.DataFrame(), {}

    chat_pairs, session_id, df_result, timing = chat_pipeline(last_user_msg, session_id)
    return chat_pairs, session_id, df_result, timing

# ------------------- UI -------------------
CSS = """
#sidebar { width: 260px; transition: width 0.3s ease; }
#sidebar.collapsed { width: 0px !important; overflow: hidden; }
#collapse-btn { background:#1e293b; color:white; padding:4px 8px; border-radius:6px; margin-bottom:6px; cursor:pointer; }
"""

theme = gr.themes.Monochrome(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(css=CSS, theme=theme) as demo:
    default_session = str(uuid.uuid4())
    if default_session not in chat_sessions:
        chat_sessions[default_session] = []
        save_json(chat_sessions)

    session_id_state = gr.State(default_session)

    with gr.Row():
        # ------------- LEFT SIDEBAR -------------
        with gr.Column(scale=1, min_width=220, elem_id="sidebar"):
            collapse_btn = gr.Button("⮜", elem_id="collapse-btn")
            # Optional logo
            if os.path.exists("logo.png"):
                gr.Image(value="logo.png", show_label=False, height=90, width=180)
            gr.Markdown("### 🕘 Chat Controls")

            new_btn = gr.Button("✨ New Chat")
            load_btn = gr.Button("📥 Load Current Chat")
            delete_btn = gr.Button("🗑 Delete Current Chat")

        # ------------- MAIN COLUMN -------------
        with gr.Column(scale=4):
            gr.Markdown("## 🤖 **CX Agent – Customer Experience AI Assistant**")

            chatbot = gr.Chatbot(
                height=460,
                type="messages",
                show_label=False,
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Type your message here...",
                    placeholder="Ask anything about your CX survey data…",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Markdown("### 📑 Result Table (full DataFrame)")
            result_df = gr.Dataframe(
                interactive=False,
                wrap=True,
                height=220
            )

            gr.Markdown("### ⏱ Run Details")
            timing_json = gr.JSON(value={}, label="Step timing (seconds)")

    # ---------- Wiring the events ----------

    # Textbox submit: echo user, then run agent
    msg.submit(
        echo_user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        bot_reply,
        inputs=[chatbot, session_id_state],
        outputs=[chatbot, session_id_state, result_df, timing_json],
    )

    # Send button: same behavior
    send_btn.click(
        echo_user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        bot_reply,
        inputs=[chatbot, session_id_state],
        outputs=[chatbot, session_id_state, result_df, timing_json],
    )

    # New chat
    new_btn.click(
        new_chat,
        inputs=None,
        outputs=[chatbot, session_id_state, result_df, timing_json],
    )

    # Load current chat
    load_btn.click(
        load_history,
        inputs=[session_id_state],
        outputs=[chatbot, session_id_state, result_df, timing_json],
    )

    # Delete current chat
    delete_btn.click(
        delete_history,
        inputs=[session_id_state],
        outputs=[chatbot, session_id_state, result_df, timing_json],
    )

    # Sidebar collapse toggle
    collapse_btn.click(
        None,
        None,
        None,
        js="""
        () => {
            const el = document.getElementById('sidebar');
            el.classList.toggle('collapsed');
        }
        """
    )

demo.launch(debug=True, share=True)
