import ollama
r = ollama.embed(model="qwen3-embedding:0.6b", input=["the cat sat", "a dog ran"])
len(r.embeddings)
len(r.embeddings[0])
r.embeddings[0][:5]



