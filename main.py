from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Configure local embedding model (small and efficient)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"  # Only ~33MB, great for Jetson Nano
)

# 2. Load documents from a directory
documents = SimpleDirectoryReader("data").load_data()

# 3. Create an index (embeddings run locally now!)
index = VectorStoreIndex.from_documents(documents)

# 4. Query the index (still needs an LLM - uses OpenAI by default)
query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
print(response)