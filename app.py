import streamlit as st
import os
import dotenv
import uuid
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from rag_methods import (
    stream_llm_response,
    load_doc_to_db,
    load_url_to_db,
    stream_llm_rag_response
)

dotenv.load_dotenv()

MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]

st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# HEADER
st.html("""<h1 style="text-align: center;">🦾🤖 <i> RAGbot </i> 📚🔎</h1> """)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    }]

with st.sidebar:
    default_openai_api_key = os.getenv("OPENAI_API_KEY")
    st.selectbox(
        "🤖 Select a Model",
        [model for model in MODELS],
        key = "model",
    )
    st.divider()
    cols0 = st.columns(2)
    with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "Use RAG", 
                value=is_vector_db_loaded, 
                key="use_rag", 
                disabled=not is_vector_db_loaded,
            )
    with cols0[1]:
        st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")
    
    st.header("📚 RAG Sources")

    # File Upload
    st.file_uploader(
        "Upload Documents",
        type = ["pdf", "docx", "txt", "md", "doc"],
        accept_multiple_files = True,
        on_change = load_doc_to_db,
        key = "rag_docs",
    )

    # URL Input
    st.text_input(
        "Enter URL to scrape",
        placeholder = "https://example.com",
        on_change = load_url_to_db,
        key = "rag_url",
    )

    with st.expander(f"Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.vector_db.get()['metadatas'])})"):
        st.write([] if not is_vector_db_loaded else [meta["source"] for meta in st.session_state.vector_db.get()["metadatas"]])



# Chat App portion
model_provider = st.session_state.model.split("/")[0]
if model_provider == "openai":
    llm_stream = ChatOpenAI(
        model_name = st.session_state.model.split("/")[-1],
        temperature = 0.3,
        streaming = True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
        
        if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
        else:
            st.write_stream(stream_llm_rag_response(llm_stream, messages))