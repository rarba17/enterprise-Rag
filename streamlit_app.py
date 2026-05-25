"""
Enterprise RAG - Streamlit Frontend
A clean, modern UI for document Q&A with streaming responses.
"""

import streamlit as st
import requests
import asyncio
import websockets
import json
from typing import Generator

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom header */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }

    .app-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }

    .app-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }

    /* Chat messages */
    .chat-message {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }

    .chat-message.user {
        background: #667eea;
        margin-left: 20%;
        color: #ffffff;
    }

    .chat-message.assistant {
        background: #2d2d2d;
        margin-right: 20%;
        color: #ffffff;
        border: 1px solid #444;
    }

    .chat-message .role {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        color: rgba(255, 255, 255, 0.7);
    }

    .chat-message .content {
        font-size: 1rem;
        line-height: 1.6;
        color: #ffffff;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .sidebar-section p {
        color: #ffffff !important;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .status-badge.success {
        background: #28a745;
        color: #ffffff;
    }

    .status-badge.pending {
        background: #ffc107;
        color: #000000;
    }

    .status-badge.error {
        background: #dc3545;
        color: #ffffff;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        background: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #888 !important;
    }

    .stTextArea > div > div > textarea {
        border-radius: 8px;
        background: #2d2d2d !important;
        color: #ffffff !important;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* File uploader */
    .uploadedFile {
        border-radius: 8px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "token" not in st.session_state:
    st.session_state.token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents" not in st.session_state:
    st.session_state.documents = []

# =============================================================================
# API FUNCTIONS
# =============================================================================

def api_register(email: str, password: str) -> dict:
    """Register a new user."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/register",
            json={"email": email, "password": password}
        )
        return {"success": response.status_code == 201, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def api_login(email: str, password: str) -> dict:
    """Login and get JWT token."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            data={"username": email, "password": password}
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        return {"success": False, "error": response.json().get("detail", "Login failed")}
    except Exception as e:
        return {"success": False, "error": str(e)}


def api_upload_document(file, token: str) -> dict:
    """Upload a PDF document."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files=files,
            headers=headers
        )
        return {"success": response.status_code == 200, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def api_get_documents(token: str) -> dict:
    """Get list of uploaded documents."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{API_BASE_URL}/documents/", headers=headers)
        return {"success": response.status_code == 200, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def api_delete_document(document_id: str, token: str) -> dict:
    """Delete a document by ID."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.delete(
            f"{API_BASE_URL}/documents/{document_id}",
            headers=headers
        )
        return {"success": response.status_code == 204}
    except Exception as e:
        return {"success": False, "error": str(e)}


def api_ask_question(question: str, token: str) -> dict:
    """Ask a question (non-streaming)."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            f"{API_BASE_URL}/query/ask",
            params={"question": question},
            headers=headers
        )
        return {"success": response.status_code == 200, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def stream_response(question: str, token: str) -> Generator[str, None, None]:
    """Stream response via WebSocket."""
    async def get_stream():
        uri = f"{WS_BASE_URL}/query/ws?token={token}"
        chunks = []
        try:
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({"question": question}))
                while True:
                    response = json.loads(await ws.recv())
                    if response.get("done"):
                        break
                    if "chunk" in response:
                        chunks.append(response["chunk"])
                    if "error" in response:
                        chunks.append(f"\n\nError: {response['error']}")
                        break
        except Exception as e:
            chunks.append(f"\n\nConnection error: {str(e)}")
        return chunks

    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    chunks = loop.run_until_complete(get_stream())
    loop.close()

    for chunk in chunks:
        yield chunk

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the app header."""
    st.markdown("""
    <div class="app-header">
        <h1>🔍 Enterprise RAG</h1>
        <p>Ask questions about your documents using AI-powered search</p>
    </div>
    """, unsafe_allow_html=True)


def render_auth_page():
    """Render login/register page."""
    render_header()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Welcome Back")

        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", use_container_width=True)

                if submit:
                    if email and password:
                        with st.spinner("Logging in..."):
                            result = api_login(email, password)
                            if result["success"]:
                                st.session_state.authenticated = True
                                st.session_state.token = result["data"]["access_token"]
                                st.session_state.user_email = email
                                st.success("Login successful!")
                                st.rerun()
                            else:
                                st.error(result.get("error", "Login failed"))
                    else:
                        st.warning("Please enter email and password")

        with tab2:
            with st.form("register_form"):
                new_email = st.text_input("Email", placeholder="you@example.com", key="reg_email")
                new_password = st.text_input("Password", type="password", placeholder="Create a password", key="reg_pass")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                register = st.form_submit_button("Create Account", use_container_width=True)

                if register:
                    if new_email and new_password:
                        if new_password != confirm_password:
                            st.error("Passwords don't match")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters")
                        else:
                            with st.spinner("Creating account..."):
                                result = api_register(new_email, new_password)
                                if result["success"]:
                                    st.success("Account created! Please login.")
                                else:
                                    st.error(result.get("error", "Registration failed"))
                    else:
                        st.warning("Please fill in all fields")


def render_sidebar():
    """Render the sidebar with user info and documents."""
    with st.sidebar:
        # User info
        st.markdown(f"""
        <div class="sidebar-section">
            <p style="margin:0; font-size:0.85rem; color:#666;">Logged in as</p>
            <p style="margin:0; font-weight:600;">{st.session_state.user_email}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.token = None
            st.session_state.user_email = None
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")

        # Document upload
        st.markdown("### 📄 Documents")

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Upload a PDF document to ask questions about"
        )

        if uploaded_file:
            if st.button("📤 Upload", use_container_width=True):
                with st.spinner("Uploading..."):
                    result = api_upload_document(uploaded_file, st.session_state.token)
                    if result["success"]:
                        st.success("Document uploaded!")
                        st.balloons()
                    else:
                        st.error(result.get("error", "Upload failed"))

        # Document list
        st.markdown("#### Your Documents")

        if st.button("🔄 Refresh", use_container_width=True):
            result = api_get_documents(st.session_state.token)
            if result["success"]:
                st.session_state.documents = result["data"]

        if st.session_state.documents:
            for doc in st.session_state.documents:
                doc_id = doc.get('id', '')
                status_class = {
                    "COMPLETED": "success",
                    "PENDING": "pending",
                    "PROCESSING": "pending",
                    "FAILED": "error"
                }.get(doc.get("status", ""), "pending")

                doc_col1, doc_col2 = st.columns([5, 1])
                with doc_col1:
                    st.markdown(f"""
                    <div style="padding:0.75rem; background:rgba(255,255,255,0.1); border-radius:8px; margin-bottom:0.5rem; border:1px solid #444;">
                        <p style="margin:0; font-size:0.85rem; font-weight:500; word-break:break-all; color:#fff;">
                            📎 {doc.get('filename', 'Unknown')}
                        </p>
                        <span class="status-badge {status_class}">{doc.get('status', 'Unknown')}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with doc_col2:
                    if st.button("🗑️", key=f"del_{doc_id}", help="Delete document"):
                        result = api_delete_document(doc_id, st.session_state.token)
                        if result["success"]:
                            st.toast("Document deleted!")
                            # Refresh document list
                            refresh = api_get_documents(st.session_state.token)
                            if refresh["success"]:
                                st.session_state.documents = refresh["data"]
                            st.rerun()
                        else:
                            st.error(result.get("error", "Delete failed"))
        else:
            st.caption("No documents uploaded yet")

        st.markdown("---")

        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


def render_chat_message(role: str, content: str):
    """Render a chat message."""
    role_display = "You" if role == "user" else "Assistant"
    css_class = "user" if role == "user" else "assistant"

    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div class="role">{role_display}</div>
        <div class="content">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_main_chat():
    """Render the main chat interface."""
    render_header()

    # Chat history
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center; padding:3rem; color:#aaa;">
                <p style="font-size:3rem; margin-bottom:1rem;">💬</p>
                <h3 style="color:#fff;">Start a conversation</h3>
                <p style="color:#aaa;">Upload a document and ask questions about it.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.chat_history:
                render_chat_message(message["role"], message["content"])

    # Input area — uses chat_input for Enter-to-submit and auto-clear
    question = st.chat_input("What would you like to know about your documents?")

    # Handle send
    if question:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        # Get response with streaming
        with st.spinner("Thinking..."):
            full_response = ""
            response_placeholder = st.empty()

            for chunk in stream_response(question, st.session_state.token):
                full_response += chunk
                response_placeholder.markdown(f"""
                <div class="chat-message assistant">
                    <div class="role">Assistant</div>
                    <div class="content">{full_response}▌</div>
                </div>
                """, unsafe_allow_html=True)

            response_placeholder.empty()

        # Add assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response
        })

        st.rerun()


def render_main_page():
    """Render the main application page."""
    render_sidebar()
    render_main_chat()

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    if not st.session_state.authenticated:
        render_auth_page()
    else:
        render_main_page()


if __name__ == "__main__":
    main()
