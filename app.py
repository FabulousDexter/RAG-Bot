import streamlit as st
from query_data import query_rag
from populate_database import load_documents, split_documents, add_to_chroma
import os

st.title("📚 RAG Document Assistant")
st.sidebar.header("Document Management")

# File upload
uploaded_files = st.sidebar.file_uploader(
    "Upload documents", type=["pdf", "txt", "md"], accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        # Save uploaded files
        with open(os.path.join("data", file.name), "wb") as f:
            f.write(file.getbuffer())
    if st.sidebar.button("Process Documents"):
        # Run population logic
        st.success("Documents processed!")

# Query interface
query = st.text_input("Ask a question about your documents:")
col1, col2 = st.columns(2)
filter_path = col1.text_input("Filter by path (optional):")
num_chunks = col2.slider("Number of chunks:", 1, 10, 5)

if st.button("Search"):
    with st.spinner("Searching..."):
        response = query_rag(query, filter_path=filter_path, k=num_chunks)
        st.write(response)
