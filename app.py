import os
import os.path as osp
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# ---------- PAGE LAYOUT ----------
st.set_page_config(page_title="Company Knowledge Base Agent", page_icon="🤖")

st.title("🤖 Company Knowledge Base Agent")
st.write("Ask any question related to company policies, leave, benefits, or work rules.")

# ---------- CHECK GROQ API KEY ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is not set.\n\n"
        "In PowerShell, run:\n\n"
        '    $env:GROQ_API_KEY = "gsk_your_real_key_here"\n\n'
        "Then restart the app:\n\n"
        "    python -m streamlit run app.py"
    )
    st.stop()

# ---------- CHECK DOCUMENT ----------
DOC_PATH = "data/company_policies.txt"

if not osp.exists(DOC_PATH):
    st.error(
        f"Could not find the document '{DOC_PATH}'.\n\n"
        "Create a folder named 'data' next to app.py and put a file\n"
        "called 'company_policies.txt' inside it."
    )
    st.stop()


# ---------- BUILD VECTOR STORE + LLM ----------
@st.cache_resource
def build_qa_system(doc_path: str):
    # 1. Load document
    loader = TextLoader(doc_path, encoding="utf-8")
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)

    # 3. Local embeddings (no paid API, uses sentence-transformers)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Build Chroma vector store
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)

    # 5. Groq Llama 3.1 model
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",  # fast, good quality :contentReference[oaicite:1]{index=1}
        temperature=0.1,
    )

    return vectordb, llm


st.info("Loading knowledge base for the first time. Please wait a moment...")

try:
    vectordb, llm = build_qa_system(DOC_PATH)
except Exception as e:
    st.error(f"Error while building knowledge base:\n\n{e}")
    st.stop()


# ---------- SIMPLE RAG PIPELINE ----------
def answer_question(question: str):
    # 1. Retrieve top-k similar chunks
    docs = vectordb.similarity_search(question, k=3)

    # 2. Build context string
    context = "\n\n".join(d.page_content for d in docs)

    # 3. Prompt for the LLM
    prompt = (
        "You are a helpful HR assistant for a company.\n"
        "Use ONLY the information in the context below to answer the user's question.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and concisely."
    )

    resp = llm.invoke(prompt)
    answer = resp.content
    return answer, docs


# ---------- MAIN UI ----------
question = st.text_input("💬 Enter your question about company policies:")

if question:
    with st.spinner("Thinking..."):
        try:
            answer, sources = answer_question(question)
        except Exception as e:
            st.error(f"Something went wrong while answering your question:\n\n{e}")
        else:
            st.subheader("✅ Answer")
            st.write(answer)

            if sources:
                with st.expander("📄 Sources used"):
                    for i, doc in enumerate(sources, start=1):
                        st.markdown(f"**Source {i}:**")
                        st.write(doc.page_content)

