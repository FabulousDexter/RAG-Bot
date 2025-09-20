"""
RAG Bot Database Population Script

This script processes PDF documents and populates a ChromaDB vector database
for use with the RAG (Retrieval-Augmented Generation) system.

Features:
- Loads PDF documents from a specified directory
- Splits documents into manageable chunks with overlap
- Generates embeddings using Ollama's nomic-embed-text model
- Stores embeddings in ChromaDB for similarity search
- Supports incremental updates (only processes new documents)
- Provides database reset functionality

Usage:
    python populate_database.py                 # Process new documents
    python populate_database.py --reset         # Reset and reprocess all documents
"""

import argparse
import os
import shutil
import hashlib
from typing import List
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

# Configuration constants
CHROMA_PATH = "chroma"  # Directory where ChromaDB stores vector embeddings
DATA_PATH = "data"  # Directory containing PDF documents to process


def main():
    """
    Main entry point for the database population script.

    Parses command line arguments and orchestrates the document processing pipeline:
    1. Optionally resets the database if --reset flag is provided
    2. Loads PDF documents from the data directory
    3. Splits documents into chunks for processing
    4. Adds new chunks to the ChromaDB vector store
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    # Clear existing database if reset flag is provided
    if args.reset:
        print("Clearing Database")
        clear_database()

    # Execute the document processing pipeline
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    """Load all PDF, TXT, and Markdown files from the data directory and subdirectories
    Supports organized folder structures like:
    data/
    ├── technical/
    │   ├── manual.pdf
    │   └── specs.txt
    ├── reports/
    │   ├── quarterly.pdf
    │   └── summary.md
    └── notes/
        └── meeting_notes.txt"""

    all_documents = []

    print(f"📁 Scanning '{DATA_PATH}' directory and all subdirectories...")

    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    try:
        pdf_documents = pdf_loader.load()
        if pdf_documents:
            for doc in pdf_documents:
                source_path = doc.metadata.get("source", "")
                doc.metadata["file_type"] = "PDF"
                doc.metadata["relative_path"] = os.path.relpath(source_path, DATA_PATH)

            print(f"📄 Loaded {len(pdf_documents)} PDF pages from:")

            pdf_sources = set([doc.metadata.get("source", "") for doc in pdf_documents])
            for source in sorted(pdf_sources):
                rel_path = os.path.relpath(source, DATA_PATH)
                print(f"    - {rel_path}")
            all_documents.extend(pdf_documents)
    except Exception as e:
        print(f"    No PDFS found or error loading: {e}")

    txt_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
        show_progress=True,
        use_multithreading=True,
    )
    try:
        txt_documents = txt_loader.load()
        if txt_documents:
            # Add metadata for better tracking
            for doc in txt_documents:
                doc.metadata["file_type"] = "TXT"
                source_path = doc.metadata.get("source", "")
                doc.metadata["relative_path"] = os.path.relpath(source_path, DATA_PATH)

            print(f"📝 Loaded {len(txt_documents)} TXT files from:")
            # Show unique TXT files loaded
            txt_sources = set([doc.metadata.get("source", "") for doc in txt_documents])
            for source in sorted(txt_sources):
                rel_path = os.path.relpath(source, DATA_PATH)
                print(f"   - {rel_path}")
            all_documents.extend(txt_documents)
    except Exception as e:
        print(f"   No TXT files found or error loading: {e}")

    md_loader = DirectoryLoader(
        DATA_PATH,  # Base directory
        glob="**/*.md",  # ** means any subdirectory, *.md means any markdown file
        loader_cls=UnstructuredMarkdownLoader,
        loader_kwargs={"mode": "single"},  # 'single' mode keeps document together
        show_progress=True,
        use_multithreading=True,
    )
    try:
        md_documents = md_loader.load()
        if md_documents:
            # Add metadata for better tracking
            for doc in md_documents:
                doc.metadata["file_type"] = "MD"
                source_path = doc.metadata.get("source", "")
                doc.metadata["relative_path"] = os.path.relpath(source_path, DATA_PATH)

            print(f"📘 Loaded {len(md_documents)} Markdown files from:")
            # Show unique MD files loaded
            md_sources = set([doc.metadata.get("source", "") for doc in md_documents])
            for source in sorted(md_sources):
                rel_path = os.path.relpath(source, DATA_PATH)
                print(f"   - {rel_path}")
            all_documents.extend(md_documents)
    except Exception as e:
        print(f"   No Markdown files found or error loading: {e}")

    if all_documents:
        print(f"\n📊 Summary by folder:")
        folder_counts = {}
        for doc in all_documents:
            rel_path = doc.metadata.get("relative_path", "")
            folder = os.path.dirname(rel_path) if os.path.dirname(rel_path) else "root"
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

        for folder, count in sorted(folder_counts.items()):
            print(f"   📁 {folder}/: {count} chunks")

    print(f"\n✅ Total documents loaded: {len(all_documents)}")

    if len(all_documents) == 0:
        print(
            "⚠️  No documents found. Please add PDF, TXT, or MD files to the 'data' folder or its subdirectories."
        )
        print("   Example structure:")
        print("   data/")
        print("   ├── reports/")
        print("   │   └── annual_report.pdf")
        print("   ├── notes/")
        print("   │   └── meeting_notes.txt")
        print("   └── documentation/")
        print("       └── readme.md")

    return all_documents


def split_documents(documents: list[Document]):
    """
    Split large documents into smaller, manageable chunks for processing.

    Uses RecursiveCharacterTextSplitter to break documents into chunks while
    preserving context through overlap. This ensures that related information
    isn't separated across chunk boundaries.

    Args:
        documents (list[Document]): List of Document objects to split

    Returns:
        list[Document]: List of document chunks with preserved metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Maximum characters per chunk
        chunk_overlap=80,  # Overlap between chunks to preserve context
        length_function=len,  # Function to measure text length
        is_separator_regex=False,  # Use simple string separators, not regex
    )

    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    """
    Add document chunks to the ChromaDB vector store.

    This function handles the core vector database operations:
    1. Connects to existing ChromaDB instance or creates new one
    2. Generates unique IDs for each chunk
    3. Checks for existing documents to avoid duplicates
    4. Adds only new chunks to the database
    5. Persists the database to disk

    Args:
        chunks (list[Document]): Document chunks to add to the database
    """
    # Initialize ChromaDB with persistence and embedding function
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Generate unique IDs for chunks and check for existing documents
    chunks_with_ids = calculate_chunks_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    existing_hashes = {}
    for i, doc_id in enumerate(existing_items["ids"]):
        metadata = existing_items["metadatas"][i]
        if metadata and "content_hash" in metadata:
            existing_hashes[doc_id] = metadata["content_hash"]
    print(f"Number of existing documents in DB: {len(existing_ids)}\n")

    # Filter out chunks that already exist in the database
    new_chunks = []
    updated_chunks = []
    unchanged_chunks = []

    for chunk in chunks_with_ids:
        chunk_id = chunk.metadata["id"]
        chunk_hash = chunk.metadata["content_hash"]

        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
        elif existing_hashes.get(chunk_id) != chunk_hash:
            updated_chunks.append(chunk)
            print(f"📝 Content changed for chunk: {chunk_id}")
        else:
            unchanged_chunks.append(chunk)

    # Add new chunks to database if any exist
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunks_ids)

    if len(updated_chunks):
        print(f"🔄 Updating changed documents: {len(updated_chunks)}")
        for chunk in updated_chunks:
            db.delete(ids=[chunk.metadata["id"]])
            db.add_documents([chunk], ids=[chunk.metadata["id"]])

    print(f"✅ Unchanged documents: {len(unchanged_chunks)}")
    print(f"🔄 Updating changed documents: {len(updated_chunks)}")
    db.persist()


def calculate_chunks_ids(chunks: list[Document]):
    """
    Generate unique identifiers for document chunks.

    Creates hierarchical IDs in the format: "source_file:page_number:chunk_index"
    This allows for easy tracking of which specific part of which document
    each chunk originated from.

    Args:
        chunks (list[Document]): Document chunks needing unique IDs

    Returns:
        list[Document]: Same chunks with added 'id' field in metadata
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # Extract source file and page information from metadata
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # Reset chunk index for new pages, increment for same page
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Generate unique chunk ID and add to metadata
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk_content = chunk.page_content
        content_hash = hashlib.md5(chunk_content.encode()).hexdigest()
        chunk.metadata["id"] = chunk_id
        chunk.metadata["content_hash"] = content_hash

    return chunks


def clear_database():
    """
    Remove the entire ChromaDB directory and all stored vectors.

    This function provides a clean slate for database repopulation.
    Use with caution as it permanently deletes all existing embeddings.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🗑️  Database cleared successfully!")


if __name__ == "__main__":
    main()
