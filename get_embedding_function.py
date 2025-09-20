# Example: import the OpenAIEmbeddings class from langchain.embeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


if __name__ == "__main__":
    embeddings = get_embedding_function()
    test_text = "Hello, this is a test"
    result = embeddings.embed_query(test_text)
    print(f"Embedding function works! Vector dimension: {len(result)}")
