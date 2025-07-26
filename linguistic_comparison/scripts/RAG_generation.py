from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil

# Step 1: Save your RAG file in a folder
os.makedirs("rag_docs", exist_ok=True)
shutil.copy("/rag_input.txt", "rag_input.txt")

# Step 2: Load the file
documents = SimpleDirectoryReader(input_dir="rag_docs").load_data()

# Step 3: Build the vector index
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Step 4: Set up LLaMA 3 70B via Together
llm = TogetherLLM(
    model="meta-llama/Llama-3-70b-chat-hf",
    api_key="",  # Replace with YOUR API JEY
)

# Step 5: Create query engine
query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

# Step 6: Define your query
query = (
    "Write a 20 word notification to motivate someone in the maintenance stage of change to integrate more exercise in their day. "
    "In this stage, people do not intend to take action in the foreseeable future (defined as within the next 6 months). People are often unaware that their behavior is problematic or produces negative consequences. "
    "Write 4 message variants with different motivational techniques."
)

# Step 7: Show retrieved context chunks before generating response
print("\nRetrieved context chunks for this query:\n")
retrieved_nodes = query_engine.retrieve(query)

for i, node in enumerate(retrieved_nodes):
    print(f"--- Retrieved Chunk {i+1} ---")
    print(node.text.strip(), "\n")

# Step 8: Generate the final answer using the LLM + retrieved context
print("\nLLaMA 3's Response:\n")
response = query_engine.query(query)
print(response)

