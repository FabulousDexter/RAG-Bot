from get_embedding_function import get_embedding_function

embeddings = get_embedding_function()
test_text = "Hello, this is a test"
result = embeddings.embed_query(test_text)
print(f"Embedding functions works! Vector dimension: {len(result)}")
