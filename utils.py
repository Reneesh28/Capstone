import os
import tempfile
from typing import List, Dict, TypedDict

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# Import the base chromadb library
import chromadb
# Chroma is already imported from langchain_community
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- Vector DB Functions (Updated for Persistence) ---

def create_vector_db(_chunks: List[Document]):
    """
    Creates or loads a persistent Chroma vector store from document chunks.
    """
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Define a persistent directory to store the database
    persist_directory = "chroma_db"

    # Create a persistent Chroma client
    # This will create the directory if it doesn't exist
    client = chromadb.PersistentClient(path=persist_directory)

    # Create or load the vector store from the persistent directory
    vector_db = Chroma(
        collection_name="fitness_documents",
        embedding_function=embeddings,
        persist_directory=persist_directory,
        client=client
    )

    # Add the new documents to the existing collection.
    # Chroma handles deduplication based on document content and metadata.
    vector_db.add_documents(_chunks)
    
    # The Chroma instance is now connected to the persistent database
    return vector_db

# --- Standard PDF Processing ---

def load_pdf(file) -> List[Document]:
    """Loads a PDF from a Streamlit UploadedFile object."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    # Clean up the temporary file after loading
    documents = loader.load()
    os.remove(tmp_path)
    return documents

def split_text(docs: List[Document]) -> List[Document]:
    """Splits loaded documents into smaller chunks for the vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# --- LangGraph Implementation using Groq API ---

class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    context: str
    chat_history: List[Dict]
    response: str

def retrieve_context_node(state: GraphState, vector_db: Chroma) -> Dict:
    """Node that retrieves relevant documents from the vector store."""
    question = state["question"]
    # Chroma uses similarity_search by default
    docs = vector_db.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

def generate_response_node(state: GraphState) -> Dict:
    """Node that generates a response using the Groq API."""
    question = state["question"]
    context = state["context"]
    chat_history = state["chat_history"]

    # Initialize the Groq Chat client with the Llama 3 70B model
    llm = ChatGroq(model_name="llama3-70b-8192")

    system_prompt = (
        "You are a friendly and encouraging Fitness Assistant. Your goal is to provide detailed, "
        "accurate, and supportive answers based ONLY on the context provided. "
        "If the answer is not in the context, politely state that you cannot provide an answer based on the document."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"})

    # The LangChain client handles the API call
    response = llm.invoke(messages)
    
    # The response object has a 'content' attribute
    return {"response": response.content}

def create_conversational_rag_chain(vector_db: Chroma):
    """Creates and compiles the LangGraph for the API-based RAG chain."""
    workflow = StateGraph(GraphState)
    
    # The lambda function now correctly passes the vector_db to the node
    workflow.add_node("retrieve_context", lambda state: retrieve_context_node(state, vector_db))
    workflow.add_node("generate_response", generate_response_node)
    
    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()