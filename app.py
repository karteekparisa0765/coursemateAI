import tempfile
from pathlib import Path

import streamlit as st

from rag_pipeline import (
    DB_DIRECTORY,
    answer_question,
    database_exists,
    get_active_database_directory,
    ingest_pdf,
)

st.set_page_config(
    page_title="CourseMate",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Global styling
# NOTE: every custom "card" below is built with st.container(border=True)
# rather than hand-rolled <div> wrappers split across markdown calls.
# Streamlit renders each st.markdown/widget call as an independent DOM
# sibling, so opening a <div> in one call and closing it in another does
# NOT actually nest the widgets in between — it just breaks the box apart.
# Using real containers avoids that, and also inherits correct theming.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        color-scheme: light only;
        --ink: #0f172a;
        --ink-soft: #475569;
        --ink-faint: #94a3b8;
        --accent: #7c3aed;
        --card-border: rgba(15, 23, 42, 0.08);
        --card-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color-scheme: light only;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(124, 58, 237, 0.09), transparent 30%),
            radial-gradient(circle at 92% 4%, rgba(14, 165, 233, 0.11), transparent 28%),
            linear-gradient(180deg, #fbfaff 0%, #f5f6fb 100%) !important;
    }

    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1180px;
    }

    /* ---------- Hero ---------- */
    .hero-card {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.95), rgba(14, 165, 233, 0.90));
        box-shadow: 0 20px 45px rgba(76, 29, 149, 0.20);
        color: white;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.4rem;
    }
    .hero-card::after {
        content: "";
        position: absolute;
        right: -60px;
        top: -60px;
        width: 220px;
        height: 220px;
        background: rgba(255,255,255,0.10);
        border-radius: 50%;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.92);
        background: rgba(255,255,255,0.16);
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.9rem;
    }
    .hero-title { margin: 0; font-size: 2.25rem; font-weight: 800; letter-spacing: -0.02em; }
    .hero-sub {
        margin: 0.6rem 0 0;
        font-size: 1rem;
        color: rgba(255,255,255,0.94);
        max-width: 620px;
        line-height: 1.5;
    }

    /* ---------- Native containers used as cards ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border-radius: 18px !important;
        border: 1px solid var(--card-border) !important;
        box-shadow: var(--card-shadow);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] p,
    div[data-testid="stVerticalBlockBorderWrapper"] label,
    div[data-testid="stVerticalBlockBorderWrapper"] li,
    div[data-testid="stVerticalBlockBorderWrapper"] span:not(.step-chip):not(.badge),
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] {
        color: var(--ink) !important;
    }

    .stat-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }
    .stat-icon.ready { background: rgba(16, 185, 129, 0.14); }
    .stat-icon.wait { background: rgba(245, 158, 11, 0.14); }
    .stat-icon.pages { background: rgba(14, 165, 233, 0.14); }
    .stat-icon.chunks { background: rgba(124, 58, 237, 0.14); }

    .small-label {
        color: var(--ink-soft) !important;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.2rem;
    }
    .stat-value { margin: 0.1rem 0 0; color: var(--ink) !important; font-size: 1.5rem; font-weight: 800; }
    .stat-caption { margin: 0.35rem 0 0; color: var(--ink-faint) !important; font-size: 0.85rem; line-height: 1.4; }

    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .badge.green { background: rgba(16,185,129,0.15); color: #047857 !important; }
    .badge.amber { background: rgba(245,158,11,0.16); color: #92400e !important; }

    .panel-heading { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.1rem; }
    .panel-heading h3 { margin: 0; color: var(--ink) !important; font-size: 1.12rem; font-weight: 700; }
    .step-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 25px;
        height: 25px;
        border-radius: 8px;
        background: var(--accent);
        color: white !important;
        font-size: 0.8rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .panel-desc { color: var(--ink-faint) !important; font-size: 0.86rem; margin: 0.25rem 0 0.9rem; }

    .empty-state { text-align: center; padding: 1.8rem 1rem; color: var(--ink-faint) !important; }
    .empty-state .emoji { font-size: 2rem; margin-bottom: 0.4rem; }

    /* ---------- Force light appearance on native widgets ---------- */
    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(124, 58, 237, 0.03) !important;
        border: 1.5px dashed rgba(124, 58, 237, 0.35) !important;
        border-radius: 14px !important;
    }
    div[data-testid="stFileUploaderDropzone"] * { color: var(--ink-soft) !important; }
    div[data-testid="stFileUploaderDropzoneInstructions"] svg { fill: var(--accent) !important; }

    div[data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 14px !important;
    }
    div[data-testid="stChatInput"] textarea { color: var(--ink) !important; }
    div[data-testid="stChatInput"] textarea::placeholder { color: var(--ink-faint) !important; }

    div[data-testid="stChatMessage"] { background: transparent !important; }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"],
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] span {
        color: var(--ink) !important;
    }

    details {
        background: #fbfcff !important;
        border: 1px solid rgba(15, 23, 42, 0.08) !important;
        border-radius: 12px !important;
    }
    details summary,
    details p,
    details li,
    details span {
        color: var(--ink) !important;
    }

    div[data-testid="stAlertContainer"] {
        border-radius: 14px !important;
    }
    div[data-testid="stAlertContainer"] * {
        color: var(--ink) !important;
    }
    div[data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] p {
        color: var(--ink) !important;
    }

    div[data-testid="stFileUploaderFile"] *,
    div[data-testid="stFileUploaderFileName"] *,
    div[data-testid="stFileUploaderDeleteBtn"] * {
        color: var(--ink) !important;
    }

    button[kind="primary"] {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid var(--card-border);
    }
    section[data-testid="stSidebar"] * { color: var(--ink) !important; }
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: #f8f7fd !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 12px;
        padding: 0.5rem 0.7rem;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
    }

    hr { border-color: var(--card-border) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("db_ready", database_exists(DB_DIRECTORY))
    st.session_state.setdefault("last_ingestion", None)


@st.cache_data(show_spinner=False)
def get_database_snapshot(db_path: str) -> dict[str, str | int]:
    path = Path(db_path)
    if not path.exists():
        return {"files": 0, "size_mb": 0.0}

    files = [item for item in path.rglob("*") if item.is_file()]
    total_bytes = sum(item.stat().st_size for item in files)
    return {
        "files": len(files),
        "size_mb": round(total_bytes / (1024 * 1024), 2),
    }


def get_snapshot_target() -> Path:
    return get_active_database_directory() or DB_DIRECTORY


def save_uploaded_pdf(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def render_sidebar() -> None:
    st.sidebar.markdown("### 📚 CourseMate")
    st.sidebar.caption("RAG-powered PDF assistant")
    st.sidebar.divider()

    st.sidebar.markdown("**Workspace**")
    st.sidebar.write(
        "Create a fresh vector database from one PDF, then chat with grounded, cited answers."
    )

    snapshot = get_database_snapshot(str(get_snapshot_target()))
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Indexed files", snapshot["files"])
    col2.metric("DB size", f"{snapshot['size_mb']} MB")

    st.sidebar.divider()
    st.sidebar.caption("PRODUCTION NOTE")
    st.sidebar.write(
        "Each new indexing run becomes the active database without deleting an in-use Chroma store."
    )

    if st.session_state.chat_history:
        st.sidebar.divider()
        if st.sidebar.button("🗑️ Clear chat history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">📖 RAG PDF Assistant</div>
            <h1 class="hero-title">CourseMate</h1>
            <p class="hero-sub">
                Upload a PDF, build a clean vector index, and ask document-grounded
                questions with cited, page-referenced snippets.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_cards() -> None:
    info = st.session_state.get("last_ingestion") or {}
    ready = st.session_state.db_ready
    left, middle, right = st.columns(3)

    with left:
        with st.container(border=True):
            icon_class = "ready" if ready else "wait"
            icon = "✅" if ready else "⏳"
            badge = (
                '<span class="badge green">READY</span>'
                if ready
                else '<span class="badge amber">WAITING</span>'
            )
            caption = (
                "Questions are enabled." if ready else "Upload and index a PDF first."
            )
            st.markdown(
                f"""
                <div class="stat-icon {icon_class}">{icon}</div>
                <div class="small-label">Database Status</div>
                <p class="stat-value">{badge}</p>
                <p class="stat-caption">{caption}</p>
                """,
                unsafe_allow_html=True,
            )

    with middle:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="stat-icon pages">📄</div>
                <div class="small-label">Pages Indexed</div>
                <p class="stat-value">{info.get("page_count", 0)}</p>
                <p class="stat-caption">Last document: {info.get("document_name", "None")}</p>
                """,
                unsafe_allow_html=True,
            )

    with right:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="stat-icon chunks">🧩</div>
                <div class="small-label">Chunks Created</div>
                <p class="stat-value">{info.get("chunk_count", 0)}</p>
                <p class="stat-caption">MMR retrieval is enabled for answer diversity.</p>
                """,
                unsafe_allow_html=True,
            )


def render_ingestion_panel() -> None:
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-heading">
                <span class="step-chip">1</span>
                <h3>Upload &amp; Index</h3>
            </div>
            <p class="panel-desc">Pick a PDF and build a fresh, searchable vector database.</p>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose a PDF", type=["pdf"], label_visibility="collapsed"
        )

        if not uploaded_file:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="emoji">📄</div>
                    Drop a PDF above to get started
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        size_kb = round(len(uploaded_file.getbuffer()) / 1024, 1)
        st.info(f"Selected file: **{uploaded_file.name}**  ·  {size_kb} KB")

        if st.button(
            "⚡ Build Fresh Database", type="primary", use_container_width=True
        ):
            temp_path = save_uploaded_pdf(uploaded_file)
            try:
                with st.spinner(
                    "Reading, splitting, embedding, and indexing your PDF..."
                ):
                    result = ingest_pdf(
                        temp_path, uploaded_file.name, replace_existing=True
                    )

                st.session_state.last_ingestion = {
                    "document_name": result.document_name,
                    "page_count": result.page_count,
                    "chunk_count": result.chunk_count,
                }
                st.session_state.db_ready = True
                get_database_snapshot.clear()
                st.success(
                    f"Indexed **{result.document_name}** — "
                    f"{result.page_count} pages, {result.chunk_count} chunks."
                )
            except Exception as exc:
                st.session_state.db_ready = database_exists(DB_DIRECTORY)
                st.error(f"Indexing failed: {exc}")
            finally:
                Path(temp_path).unlink(missing_ok=True)


def render_chat_panel() -> None:
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-heading">
                <span class="step-chip">2</span>
                <h3>Ask Questions</h3>
            </div>
            <p class="panel-desc">Chat with your indexed document — every answer is grounded and cited.</p>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.db_ready:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="emoji">🔒</div>
                    No vector database is ready yet — index a PDF first
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        if not st.session_state.chat_history:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="emoji">💬</div>
                    Ask your first question about the indexed PDF below
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            chat_box = st.container(height=380)
            with chat_box:
                for message in st.session_state.chat_history:
                    avatar = "🧑‍🎓" if message["role"] == "user" else "📚"
                    with st.chat_message(message["role"], avatar=avatar):
                        st.markdown(message["content"])
                        if message.get("sources"):
                            with st.expander(f"📎 {len(message['sources'])} source(s)"):
                                for source in message["sources"]:
                                    st.write(source)

        question = st.chat_input("Ask something about the indexed PDF")

    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})

    try:
        with st.spinner("Finding relevant passages and drafting an answer..."):
            answer, docs = answer_question(question)

        sources = []
        if docs:
            for doc in docs:
                source = doc.metadata.get("source", "Unknown source")
                page = doc.metadata.get("page")
                preview = doc.page_content[:240].strip().replace("\n", " ")
                page_label = page + 1 if isinstance(page, int) else "n/a"
                source_line = f"{source} | page {page_label} | {preview}"
                sources.append(source_line)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()
    except Exception as exc:
        st.session_state.chat_history.pop()
        st.error(f"Question answering failed: {exc}")


init_session_state()
render_sidebar()
render_header()
render_status_cards()
st.write("")

left, right = st.columns([1, 1.25], gap="large")
with left:
    render_ingestion_panel()
with right:
    render_chat_panel()
