# utils.py
import os
import tempfile
from typing import List, Dict, TypedDict

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
import lancedb

from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq

# -------------------------
# PDF loader
# -------------------------
def load_pdf(uploaded_file) -> List[Document]:
    """
    Loads a PDF from a Streamlit UploadedFile and returns a list of LangChain Document objects.
    """
    if uploaded_file is None:
        return []

    # Write to temp file because PyPDFLoader expects a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        # remove the temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return documents


# -------------------------
# Text splitter
# -------------------------
def split_text(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller chunks suitable for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# -------------------------
# Create LanceDB vector DB
# -------------------------
def create_vector_db(docs: List[Document], embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", table_name: str = "fitness_capstone"):
    """
    Create (or open) a LanceDB-backed LangChain vectorstore from documents.
    Returns a LangChain-compatible vectorstore instance (LanceDB).
    """
    if not docs:
        raise ValueError("No documents provided to create_vector_db.")

    # 1) Embeddings (we use sentence-transformers / HuggingFace here)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # 2) Prepare LanceDB connection path (ephemeral on Streamlit Cloud - consider persistent storage for production)
    lance_path = os.environ.get("LANCEDB_PATH", "/tmp/lancedb")
    os.makedirs(lance_path, exist_ok=True)

    # 3) Connect to LanceDB and use LangChain community LanceDB wrapper
    conn = lancedb.connect(lance_path)

    # 4) Build / store embeddings in LanceDB (from_documents returns a LanceDB vectorstore instance)
    db = LanceDB.from_documents(docs, embeddings, connection=conn, table_name=table_name)

    return db


# -------------------------
# LangGraph + Groq LLM RAG chain
# -------------------------
class GraphState(TypedDict):
    question: str
    context: str
    chat_history: List[Dict]
    response: str


def retrieve_context_node(state: GraphState, vector_db) -> Dict:
    """
    Node that retrieves relevant passages from the vectorstore.
    `vector_db` is expected to be a LangChain vectorstore (LanceDB).
    Returns dict with 'context' string.
    """
    question = state.get("question", "")
    # Many LangChain vectorstores expose similarity_search; LanceDB wrapper supports .similarity_search
    docs = vector_db.similarity_search(question, k=4)
    context = "\n\n".join([getattr(d, "page_content", "") for d in docs])
    return {"context": context}


def generate_response_node(state: GraphState) -> Dict:
    """
    Node that calls Groq Chat LLM to generate an answer given question + context.
    Expects environment variable GROQ_API_KEY to be set (app sets it from Streamlit secrets).
    """
    question = state.get("question", "")
    context = state.get("context", "")
    chat_history = state.get("chat_history", [])

    # Instantiate the Groq Chat LLM (uses your Groq API key from env)
    # Adjust model id if you prefer a different Groq model (e.g., "llama3-8b-8192" etc.)
    llm = ChatGroq(model_name=os.environ.get("GROQ_MODEL", "llama3-70b-8192"))

    # Build conversation messages
    system_prompt = (
        "You are a friendly and encouraging Fitness Assistant. Provide accurate answers based ONLY on the provided context. "
        "If the answer is not in the context, say you cannot answer based on the document."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Append prior chat_history items (they should be list of {"role": "...", "content": "..."})
    if chat_history:
        messages.extend(chat_history)

    # Append user message containing question + context
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
    messages.append({"role": "user", "content": user_content})

    # Call Groq LLM. ChatGroq returns a response-like object; the exact return shape may vary by version.
    # We use .invoke(...) pattern if available, otherwise call the object directly.
    try:
        # prefer invoke if available (your environment earlier used .invoke)
        llm_response = llm.invoke(messages)
        text = getattr(llm_response, "content", None) or getattr(llm_response, "text", None) or str(llm_response)
    except Exception:
        # fallback to direct call
        llm_response = llm(messages)  # many LangChain LLMs support __call__
        # __call__ commonly returns a string
        if isinstance(llm_response, str):
            text = llm_response
        else:
            text = getattr(llm_response, "content", None) or str(llm_response)

    return {"response": text}


def create_conversational_rag_chain(vector_db):
    """
    Creates a LangGraph StateGraph with two nodes:
      1) retrieve_context (vector search)
      2) generate_response (Groq LLM)
    Returns the compiled workflow (callable with .invoke(payload_dict))
    """
    workflow = StateGraph(GraphState)

    # Add nodes. retrieve_context needs the vector_db so we pass it via lambda
    workflow.add_node("retrieve_context", lambda state: retrieve_context_node(state, vector_db))
    workflow.add_node("generate_response", generate_response_node)

    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)

    # compile -> returns a runnable object supporting .invoke
    return workflow.compile()
