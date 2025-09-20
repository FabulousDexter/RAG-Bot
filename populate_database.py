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
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
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
    """
    Load all PDF documents from the data directory.

    Uses LangChain's PyPDFDirectoryLoader to automatically discover and load
    all PDF files in the DATA_PATH directory. Each PDF is converted into
    Document objects containing the text content and metadata.

    Returns:
        list[Document]: List of Document objects loaded from PDF files
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


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
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Filter out chunks that already exist in the database
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # Add new chunks to database if any exist
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunks_ids)
        db.persist()  # Save changes to disk
        print("✅ Database update completed successfully!")
    else:
        print("No new documents to add")


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
        chunk.metadata["id"] = chunk_id

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
