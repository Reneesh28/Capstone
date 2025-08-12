import streamlit as st
from dotenv import load_dotenv
from utils import (
    load_pdf,
    split_text,
    create_vector_db,
    create_conversational_rag_chain,
)
import os

# Load environment variables for local development (ignored on Streamlit Cloud)
load_dotenv()

st.set_page_config(page_title="Fitness Assistant AI", layout="wide")
st.title("Fitness Assistant AI (Powered by Groq Llama 3)")

# --- Session State Initialization ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- API Key and Setup Validation for Streamlit Cloud ---
api_key_is_set = False
try:
    # Use Streamlit secrets on cloud
    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = groq_api_key
    api_key_is_set = True
except Exception:
    # Local .env fallback
    if os.environ.get("GROQ_API_KEY"):
        api_key_is_set = True

# --- Sidebar for Setup ---
with st.sidebar:
    st.header("Setup")
    if api_key_is_set:
        st.success("Groq API Key is configured.")
    else:
        st.error("Groq API Key not found. Please add it to your Streamlit secrets or .env file.")

    uploaded_file = st.file_uploader("Upload your fitness or medical PDF", type=["pdf"])

    if st.button("Process Document", disabled=not api_key_is_set or not uploaded_file):
        with st.spinner("Processing PDF... This may take a moment."):
            try:
                docs = load_pdf(uploaded_file)
                chunks = split_text(docs)

                # Create a LanceDB-based vector DB (create_vector_db should return a LangChain-compatible vectorstore)
                st.session_state.vector_db = create_vector_db(chunks)

                # Create the RAG/QA chain (should accept the vectorstore and return an object with .invoke callable)
                st.session_state.rag_chain = create_conversational_rag_chain(st.session_state.vector_db)

                st.session_state.chat_history = []  # Reset chat history
                st.success("Setup complete! You can now start chatting.")
            except Exception as e:
                st.error(f"Failed during processing: {e}")
                raise

# --- Main Chat Interface ---
st.header("Chat with your AI Fitness Assistant")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if st.session_state.rag_chain:
    if prompt := st.chat_input("Ask a question about your fitness document"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("The AI is thinking..."):
            # Prepare input for the chain - keep shape expected by your chain
            graph_input = {"question": prompt, "chat_history": st.session_state.chat_history}
            final_state = st.session_state.rag_chain.invoke(graph_input)
            response = final_state.get("response") if isinstance(final_state, dict) else final_state

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.chat_history.append({"role": "assistant", "content": response})
else:
    # Initial state messages
    if not api_key_is_set:
        st.warning("Please configure your Groq API key in the app settings to begin.")
    else:
        st.warning("Please upload a document and click 'Process Document' in the sidebar to activate the chat.")
