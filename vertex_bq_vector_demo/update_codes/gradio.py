# full_ready_to_run_gradio_app.py
import os
import json
import uuid
import time
from typing import List, Dict, Any, Tuple

import gradio as gr
import pandas as pd

# ------------------ User-editable ------------------
JSON_PATH = "chat_history.json"
USERS_JSON = "users.json"

USER_AVATAR = "icon_chat.png" if os.path.exists("icon_chat.png") else None
AGENT_AVATAR = "agent.png" if os.path.exists("agent.png") else "🤖"
# ---------------------------------------------------

# ------------------- Default Users -------------------
DEFAULT_USERS = [
    {"email": "analyst@example.com", "password": "survey123", "name": "CX Analyst"},
    {"email": "manager@example.com", "password": "insights123", "name": "CX Manager"},
    {"email": "waheb@example.com", "password": "datascience", "name": "Waheb"},
]

def ensure_default_users():
    if not os.path.exists(USERS_JSON):
        with open(USERS_JSON, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_USERS, f, indent=2, ensure_ascii=False)

def load_users() -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(USERS_JSON):
            ensure_default_users()
        with open(USERS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_USERS

def authenticate_user(email: str, password: str) -> Tuple[bool, Dict[str, Any]]:
    users = load_users()
    email = (email or "").strip().lower()
    password = (password or "").strip()
    for u in users:
        if u.get("email", "").strip().lower() == email and u.get("password", "") == password:
            return True, u
    return False, {}

# ------------------- Helpers: JSON load/save -------------------
def load_json() -> Dict[str, List[Dict[str, str]]]:
    if not os.path.exists(JSON_PATH):
        return {}
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_json(data: Dict[str, List[Dict[str, str]]]):
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

chat_sessions = load_json()

# ------------------- Get Recent Sessions (sidebar) -------------------
def get_recent_sessions(limit: int = 3) -> List[Tuple[str, str, int]]:
    """
    Returns up to `limit` sessions, treating later keys in the dict
    as "more recent" (Python dict preserves insertion order).
    """
    sessions_info = []
    # Iterate from the end of the insertion order
    for sid, messages in reversed(list(chat_sessions.items())):
        if not messages:
            continue
        preview = "New chat"
        # Use first user message as preview
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                preview = content[:50] + "..." if len(content) > 50 else content
                break
        sessions_info.append((sid, preview, len(messages)))
        if len(sessions_info) >= limit:
            break
    return sessions_info

# ------------------- Fallback compiled_graph -------------------
class _DummyCompiledGraph:
    """Fallback simple 'graph' object with .invoke(state) that returns a dict."""
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("question", "")
        time.sleep(0.2)
        df = pd.DataFrame({
            "AGE_GROUP": ["0-11 Years", "12-25 Years", "26-35 Years", "36-45 Years", "46-55 Years"],
            "count": [515, 368, 1554, 1583, 688],
        })
        return {
            "final_result": f"📊 Analysis Results\n\nHere's the breakdown of responses by age group:\n\n• 36-45 Years: 1,583 responses\n• 26-35 Years: 1,554 responses\n• 46-55 Years: 688 responses\n• 0-11 Years: 515 responses\n• 12-25 Years: 368 responses",
            "df_result": df,
            "timing": {"retrieve_and_df": 3.56, "refine_query": 3.02, "apply_query": 0.0}
        }

# If you have a real `graph`, this will use it, otherwise fallback.
try:
    compiled_graph = graph.compile()  # type: ignore
except Exception:
    if "compiled_graph" in globals():
        compiled_graph = globals()["compiled_graph"]
    else:
        compiled_graph = _DummyCompiledGraph()

# ------------------- Format Assistant Response -------------------
def format_assistant_response(result_state: Dict[str, Any]) -> str:
    final_result = result_state.get("final_result", "")
    if not final_result or "<bound method" in str(final_result):
        df_result = result_state.get("df_result")
        if df_result is not None and not df_result.empty:
            return "✅ Query executed successfully. Please check the Data Results panel below for details."
        return "⚠️ No response generated. Please check the result table below."
    
    clean_result = str(final_result)
    summary_markers = ["📝 Summary", "### 🔎 Suggested Follow-up", "🗄️ Data source:", "<sub>⏱"]
    for marker in summary_markers:
        if marker in clean_result:
            clean_result = clean_result.split(marker)[0]
    if clean_result.startswith("```") and clean_result.endswith("```"):
        lines = clean_result.split("\n")
        clean_result = "\n".join(lines[1:-1])
    if "```json" in clean_result:
        clean_result = clean_result.split("```json")[0]
    return clean_result.strip()

# ------------------- Converters -------------------
def entries_to_chatbot(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for e in entries:
        messages.append({
            "role": e.get("role", "user"),
            "content": e.get("content", "")
        })
    return messages

def last_n_turns_history(entries: List[Dict[str, str]], n_turns: int = 2) -> List[Dict[str, str]]:
    """
    Return only the last `n_turns` user/assistant pairs (so 2*n_turns messages).
    If there are fewer messages, return them all.
    """
    if not entries:
        return []
    max_msgs = n_turns * 2
    if len(entries) <= max_msgs:
        return entries
    return entries[-max_msgs:]

# ------------------- Core Chat Pipeline -------------------
def chat_pipeline(message: str, session_id: str):
    # Ensure we have a session id
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    full_history = chat_sessions[session_id]
    history_for_model = last_n_turns_history(full_history, n_turns=2)  # ✅ only last 2 turns

    state = {
        "question": message,
        "history": history_for_model,
    }

    # Call graph
    try:
        result_state = compiled_graph.invoke(state)
    except Exception as e:
        result_state = {
            "final_result": f"⚠️ Error: {str(e)[:200]}",
            "df_result": pd.DataFrame(),
            "timing": {}
        }

    assistant_message = format_assistant_response(result_state)
    df_result = result_state.get("df_result", pd.DataFrame())
    timing = result_state.get("timing", {})

    # Append NEW turn to full session history
    chat_sessions[session_id].append({"role": "user", "content": str(message)})
    chat_sessions[session_id].append({"role": "assistant", "content": assistant_message})
    save_json(chat_sessions)

    chat_messages = entries_to_chatbot(chat_sessions[session_id])
    return chat_messages, session_id, df_result, timing

# ------------------- Session Controls -------------------
def new_chat():
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = []
    save_json(chat_sessions)
    return [], session_id, pd.DataFrame(), {}, create_history_html()

def load_session(session_id: str):
    if session_id in chat_sessions:
        entries = chat_sessions[session_id]
        chat_messages = entries_to_chatbot(entries)
        return chat_messages, session_id, pd.DataFrame(), {}, create_history_html()
    return [], session_id, pd.DataFrame(), {}, create_history_html()

def delete_history(session_id):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        save_json(chat_sessions)
    new_session_id = str(uuid.uuid4())
    chat_sessions[new_session_id] = []
    save_json(chat_sessions)
    return [], new_session_id, pd.DataFrame(), {}, create_history_html()

def create_history_html() -> str:
    recent = get_recent_sessions(2)
    if not recent:
        return "<p style='color: #64748b; font-size: 0.9em; padding: 1em;'>No chat history yet</p>"
    html = "<div style='display: flex; flex-direction: column; gap: 0.5em;'>"
    for sid, preview, msg_count in recent:
        html += f"""
        <div class='history-item' style='
            padding: 0.5em; 
            background: white; 
            border: 2px solid #F09491; 
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        ' 
        onmouseover="this.style.background='#fef2f2'; this.style.borderColor='#ED1B24'; this.style.transform='translateX(4px)';"
        onmouseout="this.style.background='white'; this.style.borderColor='#F09491'; this.style.transform='translateX(0)';"
        onclick="document.querySelector('#load_session_btn').click(); 
                 document.querySelector('#session_id_input').value='{sid}';">
            <div style='font-weight: 600; color: #ED1B24; font-size: 0.9em; margin-bottom: 0.3em;'>
                💬 {preview}
            </div>
            <div style='font-size: 0.75em; color: #64748b;'>
                {msg_count} messages
            </div>
        </div>
        """
    html += "</div>"
    return html

# ------------------- Frontend Helpers -------------------
def echo_user(message: str, chatbot_history: List[Dict[str, str]]):
    if chatbot_history is None:
        chatbot_history = []
    updated = chatbot_history + [{"role": "user", "content": str(message)}]
    updated.append({"role": "assistant", "content": "🤔 Thinking..."})
    return "", updated

def bot_reply(chatbot_history: List[Dict[str, str]], session_id: str):
    if not chatbot_history:
        return chatbot_history, session_id, pd.DataFrame(), {}, create_history_html()
    last_user = None
    for msg in reversed(chatbot_history):
        if msg.get("role") == "user" and msg.get("content", "").strip():
            last_user = msg.get("content")
            break
    if not last_user:
        return chatbot_history, session_id, pd.DataFrame(), {}, create_history_html()
    chat_messages, session_id, df_result, timing = chat_pipeline(last_user, session_id)
    return chat_messages, session_id, df_result, timing, create_history_html()

# ------------------- Login Logic -------------------
def login(email: str, password: str):
    email = (email or "").strip()
    password = (password or "").strip()

    if not email or not password:
        return (
            False,  # auth_state
            {},     # current_user
            "⚠️ Please enter both email and password.",
            gr.update(visible=True),   # login_view
            gr.update(visible=False),  # app_view
            gr.update(value="", visible=False),  # welcome_html
        )

    ok, user = authenticate_user(email, password)
    if not ok:
        return (
            False,
            {},
            "❌ Invalid email or password. Please try again.",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", visible=False),
        )

    display_name = user.get("name") or user.get("email") or "User"
    welcome_msg = f"""
    <div style='padding: 0.75rem 1rem; border-radius: 999px; background: #fef2f2;
                border: 1px solid #F09491; display: inline-flex; align-items: center;
                gap: 0.5rem; margin-bottom: 0.75rem;'>
        <span>👋</span>
        <span style='font-weight: 600; color: #ED1B24;'>
            Welcome {display_name} to <strong>SurveyIQ Agent</strong>!
        </span>
    </div>
    """

    return (
        True,          # auth_state
        user,          # current_user
        "",            # login_error
        gr.update(visible=False),  # hide login
        gr.update(visible=True),   # show app
        gr.update(value=welcome_msg, visible=True),  # welcome popup
    )

# ------------------- Sidebar Toggle -------------------
def toggle_sidebar(current_visible: bool):
    new_visible = not bool(current_visible)
    new_label = "⬅️ Hide sidebar" if new_visible else "➡️ Show sidebar"
    return (
        new_visible,
        gr.update(visible=new_visible),
        gr.update(value=new_label),
    )

# ------------------- UI CSS and Theme -------------------
CSS = """
:root {
    --ooredoo-red: #ED1B24;
    --ooredoo-pink: #F09491;
    --ooredoo-dark: #0f172a;
    --ooredoo-light: #fef2f2;
}

body {
    background: radial-gradient(circle at top, #fee2e2 0, #ffffff 40%, #e5e7eb 100%);
}

/* Login layout */
#login-container {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 1.5rem;
}

#login-card {
    max-width: 420px;
    width: 100%;
    padding: 1.75rem 1.5rem;
    background: #ffffff;
    border-radius: 18px;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
    border: 1px solid #fecaca;
}

#login-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
    background: linear-gradient(135deg, var(--ooredoo-red) 0%, var(--ooredoo-pink) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#login-subtitle {
    color: #64748b;
    font-size: 0.95rem;
}

#login-error {
    color: #b91c1c;
    font-size: 0.9rem;
    min-height: 1.2rem;
}

/* App layout */
#sidebar { 
    width: 280px; 
    background: linear-gradient(180deg, var(--ooredoo-light) 0%, #ffffff 100%);
    padding: 1rem;
    border-right: 2px solid var(--ooredoo-pink);
}

.panel-section {
    margin-top: 1.5rem;
    padding: 1rem;
    background: var(--ooredoo-light);
    border-radius: 8px;
    border: 1px solid var(--ooredoo-pink);
}

#history-container {
    max-height: 400px;
    overflow-y: auto;
}

/* Chat card */
#chatbot {
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    box-shadow: 0 12px 25px rgba(15, 23, 42, 0.12);
    padding: 0.5rem;
}

/* Typing / thinking style */
@keyframes thinking {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
}

.thinking-indicator {
    animation: thinking 1.5s ease-in-out infinite;
    color: var(--ooredoo-red);
    font-style: italic;
}

.gradio-container h1 {
    background: linear-gradient(135deg, var(--ooredoo-red) 0%, var(--ooredoo-pink) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

button[variant="primary"] {
    background: var(--ooredoo-red) !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

button[variant="primary"]:hover {
    background: #c41620 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(237, 27, 36, 0.3) !important;
}

textarea.gr-text-input {
    min-height: 70px !important;
    font-size: 1rem !important;
    padding: 0.75rem !important;
}
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#fef2f2",
        c100="#fee2e2", 
        c200="#fecaca",
        c300="#fca5a5",
        c400="#F09491",
        c500="#ED1B24",
        c600="#dc2626",
        c700="#b91c1c",
        c800="#991b1b",
        c900="#7f1d1d",
        c950="#450a0a"
    ),
    secondary_hue="slate",
    neutral_hue="slate"
)

# ------------------- Build UI -------------------
ensure_default_users()

with gr.Blocks(css=CSS, theme=theme, title="SurveyIQ Agent") as demo:
    # Global states
    default_session = str(uuid.uuid4())
    if default_session not in chat_sessions:
        chat_sessions[default_session] = []
        save_json(chat_sessions)

    session_id_state = gr.State(default_session)
    session_id_input = gr.Textbox(visible=False, elem_id="session_id_input")

    auth_state = gr.State(False)
    current_user_state = gr.State({})
    sidebar_visible_state = gr.State(True)

    # ---------- LOGIN VIEW ----------
    with gr.Column(visible=True, elem_id="login-container") as login_view:
        gr.HTML("""
            <div style='text-align: center;'>
                <div id="login-title">SurveyIQ Agent</div>
                <div id="login-subtitle">
                    Secure access to your intelligent CX assistant for survey analytics.
                </div>
            </div>
        """)

        with gr.Group(elem_id="login-card"):
            login_email = gr.Textbox(
                label="Work email",
                placeholder="you@example.com",
                autofocus=True
            )
            login_password = gr.Textbox(
                label="Password",
                placeholder="Enter your password",
                type="password"
            )
            login_btn = gr.Button("Login to SurveyIQ", variant="primary")
            login_error = gr.Markdown("", elem_id="login-error")

    # ---------- MAIN APP VIEW ----------
    with gr.Column(visible=False) as app_view:
        with gr.Row():
            # Left sidebar (collapsible)
            with gr.Column(scale=1, min_width=260, elem_id="sidebar") as sidebar_col:
                if os.path.exists("logo.png"):
                    gr.Image(value="logo.png", show_label=False, height=80, width=200)
                
                gr.Markdown("### 💬 Chat Sessions")
                new_btn = gr.Button("✨ New Chat", variant="primary", size="sm")
                delete_btn = gr.Button("🗑️ Delete Current Chat", size="sm")
                
                gr.Markdown("#### 📚 Recent Chats")
                history_html = gr.HTML(create_history_html(), elem_id="history-container")
                load_session_btn = gr.Button("Load Session", visible=False, elem_id="load_session_btn")
                
                gr.Markdown("---")
                gr.Markdown("""
                    **💡About Survey Metrics:**  
                    - **Time & Category:** `Survey_Month`, `Survey_Date`, `Survey_Category`  
                    - **Customer Info:** `GENDER`, `AGE_GROUP`, `NATIONALITY`, `CUSTOMER_SEGMENT`, `PLAN_NAME`, `CUSTOMER_TENURE`, `CUSTOMER_REGION`  
                    - **Satisfaction & Feedback:** `Satisfaction_Level`, `Survey_Response`  
                """)

            # Main chat area
            with gr.Column(scale=4):
                with gr.Row():
                    gr.HTML("""
                        <div style='margin-bottom: 1rem;'>
                            <h1 style='font-size: 2.3rem; font-weight: 700; margin-bottom: 0.2rem;
                                background: linear-gradient(135deg, #ED1B24 0%, #F09491 100%);
                                -webkit-background-clip: text;
                                -webkit-text-fill-color: transparent;
                                background-clip: text;'>🤖 SurveyIQ Agent</h1>
                            <p style='color: #64748b; font-size: 1rem; font-style: italic; margin: 0;'>
                                Your Intelligent CX Assistant for Survey Insights
                            </p>
                        </div>
                    """, scale=10)
                    toggle_sidebar_btn = gr.Button("⬅️ Hide sidebar", size="sm", scale=2)

                # Welcome popup after login
                welcome_html = gr.HTML("", visible=False)

                chatbot = gr.Chatbot(
                    value=[], 
                    height=500, 
                    show_label=False,
                    type="messages",
                    avatar_images=(USER_AVATAR, AGENT_AVATAR),
                    bubble_full_width=False,
                    elem_id="chatbot",
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="🪄💭 Curious about customer satisfaction? Ask me anything!",
                        scale=9,
                        container=False
                    )
                    send_btn = gr.Button("Send 📤", variant="primary", scale=2)

        # Bottom panels
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("📊 Data Results", open=False, elem_classes="panel-section"):
                    gr.Markdown("*Complete dataset returned from your query*")
                    result_df = gr.Dataframe(interactive=False, wrap=True, max_height=300)

            with gr.Column(scale=1):
                with gr.Accordion("⚡ Performance Metrics", open=False, elem_classes="panel-section"):
                    gr.Markdown("*Query execution time breakdown*")
                    timing_json = gr.JSON(value={}, label="", show_label=False)

    # ---------- Event Handlers ----------

    # Login
    login_btn.click(
        login,
        inputs=[login_email, login_password],
        outputs=[
            auth_state,
            current_user_state,
            login_error,
            login_view,
            app_view,
            welcome_html,
        ],
    )

    # Sidebar toggle
    toggle_sidebar_btn.click(
        toggle_sidebar,
        inputs=[sidebar_visible_state],
        outputs=[sidebar_visible_state, sidebar_col, toggle_sidebar_btn],
    )

    # Chat send
    msg.submit(
        echo_user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        bot_reply,
        inputs=[chatbot, session_id_state],
        outputs=[chatbot, session_id_state, result_df, timing_json, history_html],
    )

    send_btn.click(
        echo_user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        bot_reply,
        inputs=[chatbot, session_id_state],
        outputs=[chatbot, session_id_state, result_df, timing_json, history_html],
    )

    # Session controls
    new_btn.click(
        new_chat,
        inputs=None,
        outputs=[chatbot, session_id_state, result_df, timing_json, history_html],
    )

    load_session_btn.click(
        load_session,
        inputs=[session_id_input],
        outputs=[chatbot, session_id_state, result_df, timing_json, history_html],
    )

    delete_btn.click(
        delete_history,
        inputs=[session_id_state],
        outputs=[chatbot, session_id_state, result_df, timing_json, history_html],
    )

# Launch
if __name__ == "__main__":
    demo.launch(debug=True, share=True)
