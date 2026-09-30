import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container */
    .main { padding-top: 1rem; }

    /* Source badge */
    .source-badge {
        background-color: #1e3a5f;
        color: #7eb3ff;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78em;
        margin: 2px;
        display: inline-block;
    }

    /* Score badge */
    .score-badge {
        background-color: #1a3a2a;
        color: #4caf50;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78em;
        margin: 2px;
        display: inline-block;
    }

    /* Info box */
    .info-box {
        background-color: #0e1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0e1117 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def initialiser_systeme():
    """Initialize all components once and cache them."""

    errors = []
    if not os.path.exists("base_documind"):
        errors.append("❌ Vector base 'base_documind' not found. Run `main.py` first.")
    if "GROQ_API_KEY" not in os.environ:
        errors.append("❌ `GROQ_API_KEY` missing from `.env` file.")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.load_local(
        "base_documind",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm_groq = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=2048
    )

    return db, llm_groq


def build_prompt(langue: str) -> ChatPromptTemplate:
    """Build system prompt based on selected language."""

    lang_map = {
        "🇫🇷 Français": "Tu DOIS répondre en Français.",
        "🇬🇧 English": "You MUST respond in English.",
        "🇸🇦 العربية": "يجب أن تجيب باللغة العربية دائماً."
    }
    lang_instruction = lang_map.get(langue, "Tu DOIS répondre en Français.")

    return ChatPromptTemplate.from_messages([
        ("system", f"""You are DocuMind, a highly specialized technical assistant \
for Linux system administration and Cisco network configuration.

══════════════ STRICT RULES ══════════════
1. Use ONLY the documentation context provided below. NEVER use external knowledge.
2. If the answer is NOT in the context, respond ONLY with:
   "Je n'ai pas cette information dans ma documentation." (or translated equivalent)
3. NEVER hallucinate or guess an answer.
4. Format responses with Markdown:
   - Use ```bash for Linux commands
   - Use ```cisco for Cisco IOS commands  
   - Use bullet points for lists
   - Use headers (##) for sections
5. {lang_instruction}
══════════════════════════════════════════

📚 Documentation Context:
{{contexte}}

📍 Sources: {{sources}}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])


def extraire_sources(resultats_avec_scores: list) -> tuple:
    """Extract context string and source list from search results."""

    contexte_parts = []
    sources_details = []

    for doc, score in resultats_avec_scores:
        contexte_parts.append(doc.page_content)

        nom_fichier = doc.metadata.get("source", "Unknown")

        nom_fichier = os.path.basename(nom_fichier)
        page = doc.metadata.get("page", "?")

        sources_details.append({
            "file": nom_fichier,
            "page": page,
            "score": round(score, 3)
        })

    contexte = "\n\n---\n\n".join(contexte_parts)
    sources_str = ", ".join([
        f"{s['file']} p.{s['page']}" for s in sources_details
    ])

    return contexte, sources_str, sources_details



with st.sidebar:
    st.markdown("## 🐧 DocuMind AI")
    st.markdown("*Linux & Cisco RAG Assistant*")
    st.divider()

    st.markdown("### ⚙️ Settings")

    langue = st.selectbox(
        "🌍 Response language",
        ["🇫🇷 Français", "🇬🇧 English", "🇸🇦 العربية"],
        index=0
    )

    k_sources = st.slider(
        "📚 Sources to retrieve (k)",
        min_value=1, max_value=6, value=3,
        help="More sources = more context but slower response"
    )

    score_threshold = st.slider(
        "🎯 Relevance threshold",
        min_value=0.5, max_value=2.0, value=1.5, step=0.1,
        help="Lower = stricter filtering. Raise if you get 'no info' too often."
    )

    st.divider()

    st.markdown("### 📖 Knowledge Base")
    st.markdown("""
- 🐧 **The Linux Command Line**
- 🌐 **CCNA Course Support**
- 🔧 **Cisco Configuration Guide**
    """)

    st.divider()

    st.markdown("### 💬 Conversation")

    nb_messages = len(st.session_state.get("messages", []))
    nb_echanges = nb_messages // 2

    col1, col2 = st.columns(2)
    col1.metric("Messages", nb_messages)
    col2.metric("Exchanges", nb_echanges)

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("🎓 Academic Project - Generative AI Module")
    st.caption("Built with LangChain + Groq + FAISS")


with st.spinner("🔄 Loading DocuMind system..."):
    vectorstore, llm = initialiser_systeme()

st.markdown("""
<div class="welcome-card">
    <h2>🐧 DocuMind AI Assistant</h2>
    <p>Expert in <strong>Linux Administration</strong> & <strong>Cisco Networking</strong></p>
    <p style="color: #888; font-size: 0.85em;">
        Answers based exclusively on official documentation
    </p>
</div>
""", unsafe_allow_html=True)

# ── Example Questions (shown only when chat is empty) ──────────
EXEMPLES = {
    "🐧 Linux": [
        "How to navigate directories and list files?",
        "How to manage users and permissions in Linux?",
        "How to use grep to search in files?",
        "How to set up a cron job?",
    ],
    "🔧 Cisco": [
        "How to configure a static route on Cisco?",
        "How to set up VLANs on a Cisco switch?",
        "How to configure OSPF on a Cisco router?",
        "How to enable SSH on a Cisco device?",
    ]
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if not st.session_state.messages:
    st.markdown("#### 💡 Example Questions")

    for categorie, questions in EXEMPLES.items():
        st.markdown(f"**{categorie}**")
        cols = st.columns(2)
        for i, q in enumerate(questions):
            if cols[i % 2].button(q, use_container_width=True, key=f"ex_{q}"):
                st.session_state.pending_question = q
                st.rerun()

    st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 View Sources & Relevance Scores"):
                for src in msg["sources"]:
                    relevance = max(0, 100 - int(src['score'] * 40))
                    st.markdown(
                        f'<span class="source-badge">📄 {src["file"]}</span>'
                        f'<span class="source-badge">📃 Page {src["page"]}</span>'
                        f'<span class="score-badge">🎯 Relevance: {relevance}%</span>',
                        unsafe_allow_html=True
                    )

question_input = st.chat_input(
    "Ask about Linux or Cisco networking... (e.g. 'How to configure OSPF?')"
)

question = question_input or st.session_state.pending_question
if st.session_state.pending_question and question_input is None:
    st.session_state.pending_question = None

MAX_HISTORY = 8

if question:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documentation..."):

            resultats_avec_scores = vectorstore.similarity_search_with_score(
                question, k=k_sources
            )

            resultats_filtres = [
                (doc, score)
                for doc, score in resultats_avec_scores
                if score < score_threshold
            ]

            if resultats_filtres:
                contexte, sources_str, sources_details = extraire_sources(
                    resultats_filtres
                )
                no_context = False
            else:
                contexte = "No relevant documentation found for this question."
                sources_str = "None"
                sources_details = []
                no_context = True

            history_trimmed = st.session_state.chat_history[-MAX_HISTORY:]

            prompt = build_prompt(langue)
            chain = prompt | llm

            reponse = chain.invoke({
                "contexte": contexte,
                "sources": sources_str,
                "question": question,
                "chat_history": history_trimmed,
            })
            reponse_ia = reponse.content

        st.markdown(reponse_ia)

        if sources_details and not no_context:
            with st.expander("📚 View Sources & Relevance Scores"):
                for src in sources_details:
                    relevance = max(0, 100 - int(src['score'] * 40))
                    st.markdown(
                        f'<span class="source-badge">📄 {src["file"]}</span>'
                        f'<span class="source-badge">📃 Page {src["page"]}</span>'
                        f'<span class="score-badge">🎯 Relevance: {relevance}%</span>',
                        unsafe_allow_html=True
                    )
        elif no_context:
            st.warning(
                "⚠️ No sufficiently relevant chunks found in the documentation. "
                "Try rephrasing or adjusting the relevance threshold in the sidebar."
            )

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=reponse_ia))

    st.session_state.messages.append({
        "role": "assistant",
        "content": reponse_ia,
        "sources": sources_details
    })