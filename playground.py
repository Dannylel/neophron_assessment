import ollama

chunks = [
    "GPU memory limits affect inference.",
    "KV cache grows with sequence length.",
    "Batching can improve throughput.",
]

response = ollama.embed(
    model="qwen3-embedding:0.6b",
    input=chunks
)

embeddings = response["embeddings"]

print(len(embeddings))        # 3: one vector per chunk
print(len(embeddings[0]))     # Dimensions of the first vector