import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from datetime import datetime

load_dotenv()

st.set_page_config(
    page_title="KB Document Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #f0f2f5; }

    [data-testid="stSidebar"] {
        background-color: #0a1628;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] * { color: #c8d6e5 !important; }

    .logo-area {
        padding: 24px 20px 16px;
        border-bottom: 1px solid #1e3a5f;
        margin-bottom: 24px;
    }
    .logo-text {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff !important;
        letter-spacing: 0.5px;
    }
    .logo-sub {
        font-size: 10px;
        color: #4a6fa5 !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 2px;
    }

    .section-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #4a6fa5 !important;
        padding: 0 20px;
        margin-bottom: 8px;
    }

    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 2px;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .status-ready {
        background-color: #0d3d2e;
        color: #2ecc71 !important;
    }
    .status-waiting {
        background-color: #1e2a3a;
        color: #7f8c9a !important;
    }

    .top-bar {
        background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 100%);
        padding: 20px 32px;
        margin: -24px -24px 0 -24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #c9a84c;
    }
    .top-bar-title {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: 0.3px;
    }
    .top-bar-sub {
        font-size: 11px;
        color: #7f9abd;
        margin-top: 3px;
        letter-spacing: 0.5px;
    }
    .top-bar-right {
        text-align: right;
    }
    .top-bar-date {
        font-size: 11px;
        color: #7f9abd;
    }
    .gold-line {
        height: 2px;
        background: linear-gradient(90deg, #c9a84c, #e8d48b, #c9a84c);
        margin: 0 -24px 24px -24px;
    }

    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #0a1628;
        border-radius: 2px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .info-card-title {
        font-size: 11px;
        font-weight: 600;
        color: #7f8c9a;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .info-card-value {
        font-size: 14px;
        font-weight: 500;
        color: #1a2b4a;
    }

    .chat-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 2px;
        min-height: 400px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .empty-state {
        text-align: center;
        padding: 60px 0;
    }
    .empty-state-title {
        font-size: 14px;
        font-weight: 500;
        color: #4a5568;
        margin-bottom: 8px;
    }
    .empty-state-sub {
        font-size: 12px;
        color: #a0aec0;
    }

    [data-testid="stChatMessage"] {
        background: #f8fafc;
        border: 1px solid #e8ecf0;
        border-radius: 2px;
        padding: 16px;
        margin-bottom: 8px;
    }

    .stChatInput > div {
        border: 1px solid #cbd5e0;
        border-radius: 2px;
        background: white;
    }

    .disclaimer {
        font-size: 10px;
        color: #a0aec0;
        padding: 12px 0;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)

# 상단 바
now = datetime.now().strftime("%Y.%m.%d  %H:%M")
st.markdown(f"""
<div class="top-bar">
    <div>
        <div class="top-bar-title">Document Intelligence System</div>
        <div class="top-bar-sub">Powered by Gemini 1.5 &nbsp;|&nbsp; RAG Architecture</div>
    </div>
    <div class="top-bar-right">
        <div class="top-bar-date">{now}</div>
        <div class="top-bar-date" style="margin-top:3px; color:#c9a84c; font-weight:500;">PRIVATE &amp; CONFIDENTIAL</div>
    </div>
</div>
<div class="gold-line"></div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore, len(chunks), len(documents)

# 사이드바
with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <div class="logo-text">FinDoc AI</div>
        <div class="logo-sub">Document Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("Processing..."):
            vectorstore, chunks, pages = process_pdf(uploaded_file)
            st.session_state['vectorstore'] = vectorstore
            st.session_state['doc_name'] = uploaded_file.name
            st.session_state['chunks'] = chunks
            st.session_state['pages'] = pages

    st.markdown("<br>", unsafe_allow_html=True)

    if 'doc_name' in st.session_state:
        st.markdown(f"""
        <div style="padding: 0 4px;">
            <span class="status-badge status-ready">Ready</span>
            <div style="margin-top:10px; font-size:12px; color:#7f9abd !important;">{st.session_state['doc_name']}</div>
            <div style="margin-top:6px; font-size:11px; color:#4a6fa5 !important;">
                {st.session_state['pages']} pages &nbsp;|&nbsp; {st.session_state['chunks']} chunks
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding: 0 4px;">
            <span class="status-badge status-waiting">No Document</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Sample Queries</div>', unsafe_allow_html=True)

    queries = [
        "이 상품의 금리 조건은?",
        "가입 대상 및 조건",
        "중도해지 패널티",
        "만기 후 처리 방법",
        "What is the interest rate?",
    ]
    for q in queries:
        st.markdown(f'<div style="font-size:12px; padding: 7px 4px; border-bottom: 1px solid #1e3a5f;">{q}</div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px; padding: 0 4px; line-height: 2;">
        <div>Model &nbsp;&nbsp; Gemini 1.5 Flash</div>
        <div>Vector DB &nbsp;&nbsp; ChromaDB</div>
        <div>Framework &nbsp;&nbsp; LangChain</div>
    </div>
    """, unsafe_allow_html=True)

# 메인 채팅
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">Status</div>
        <div class="info-card-value">{"Document Loaded" if 'doc_name' in st.session_state else "Awaiting Document"}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">Queries</div>
        <div class="info-card-value">{len(st.session_state.get('messages', [])) // 2} processed</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">Engine</div>
        <div class="info-card-value">RAG + Gemini 1.5</div>
    </div>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-title">No conversation yet</div>
        <div class="empty-state-sub">Upload a PDF document and ask your first question</div>
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if 'vectorstore' not in st.session_state:
            response = "No document loaded. Please upload a PDF document to proceed."
        else:
            with st.spinner("Retrieving relevant context..."):
                llm = get_llm()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=st.session_state['vectorstore'].as_retriever(
                        search_kwargs={"k": 3}
                    )
                )
                result = qa_chain.invoke({"query": prompt})
                response = result["result"]

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("""
<div class="disclaimer">
    This system is for internal use only. Responses are AI-generated and should be verified against official documents.
</div>
""", unsafe_allow_html=True)