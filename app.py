import streamlit as st
from dotenv import load_dotenv
from utils import (
    load_pdf,
    split_text,
    create_vector_db,
    create_conversational_rag_chain,
)
import os

# Load environment variables from the .env file
load_dotenv()

st.set_page_config(page_title="Fitness Assistant AI", layout="wide")
st.title("Fitness Assistant AI (Powered by Groq Llama 3)")


# --- THIS IS THE FIX: Initialize session state at the top ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
# --- END OF FIX ---


# Check if the Groq API key is available in the environment
api_key_is_set = os.environ.get("GROQ_API_KEY") is not None

# --- Sidebar ---
with st.sidebar:
    st.header("Setup")
    
    # Display a status message about the API key
    if api_key_is_set:
        st.success("Groq API Key loaded successfully from .env file.")
    else:
        st.error("Groq API Key not found. Please create a .env file.")

    uploaded_file = st.file_uploader("Upload your fitness or medical PDF", type=["pdf"])

    # The button is disabled if the API key is not set
    if st.button("Process Document", disabled=not api_key_is_set) and uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            docs = load_pdf(uploaded_file)
            chunks = split_text(docs)
            st.session_state.vector_db = create_vector_db(chunks)
            st.session_state.rag_chain = create_conversational_rag_chain(st.session_state.vector_db)
            # Reset chat history for the new document
            st.session_state.chat_history = []
            st.success("Setup complete! You can now start chatting.")

# --- Main Chat Interface ---
st.header("Chat with your AI Fitness Assistant")

# This loop will now safely run on an empty list on the first load
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Only show the chat input if the RAG chain has been created
if st.session_state.rag_chain:
    if prompt := st.chat_input("Ask a question about your fitness document"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("The AI is thinking..."):
            graph_input = {"question": prompt, "chat_history": st.session_state.chat_history}
            final_state = st.session_state.rag_chain.invoke(graph_input)
            response = final_state["response"]
            
            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
else:
    # Guide the user on what to do next
    if not api_key_is_set:
         st.warning("Please set up your Groq API key in a .env file to begin.")
    else:
        st.warning("Please upload a document and click 'Process Document' in the sidebar to activate the chat.")    