"""
Embedding Function Configuration

This module configures and provides the embedding function used throughout the RAG system.
Embeddings convert text into high-dimensional vectors that capture semantic meaning,
enabling similarity search across document chunks.

The module uses Ollama's nomic-embed-text model, which provides:
- High-quality semantic embeddings
- Local processing (no external API calls)
- Consistent vector dimensions (768-dimensional)
- Good performance for document retrieval tasks
"""

# Import the OpenAIEmbeddings class from langchain.embeddings for reference
from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
    """
    Initialize and return the embedding function for the RAG system.

    This function creates an OllamaEmbeddings instance configured with the
    nomic-embed-text model. This model is specifically designed for creating
    high-quality embeddings suitable for retrieval tasks.

    Returns:
        OllamaEmbeddings: Configured embedding function ready for use

    Note:
        Requires Ollama to be running locally with the nomic-embed-text model installed.
        Install the model with: ollama pull nomic-embed-text
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


if __name__ == "__main__":
    """
    Test script to verify the embedding function is working correctly.

    This section runs when the script is executed directly (not imported).
    It performs a simple test to ensure:
    - The embedding function can be initialized
    - Text can be successfully converted to embeddings
    - The resulting vector has the expected dimensions
    """
    print("🧪 Testing embedding function...")

    # Initialize the embedding function
    embeddings = get_embedding_function()

    # Test with sample text
    test_text = "Hello, this is a test"
    result = embeddings.embed_query(test_text)

    # Display results
    print(f"✅ Embedding function works! Vector dimension: {len(result)}")
    print(f"📊 Sample embedding (first 5 values): {result[:5]}")
