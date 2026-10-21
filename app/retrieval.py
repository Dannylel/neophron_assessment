"""
Task 1 -- Permission-aware retrieval.

Chunk documents/*.txt, embed each chunk with a REAL embedding model
(your choice -- sentence-transformers, an API, etc. -- no keyword/TF-IDF
substitutes), attach ACL metadata from documents/manifest.json, and
implement retrieve() below.

Hard rule: an unauthorized chunk must never be constructible from the
return value of retrieve(). Filtering happens INSIDE this function,
before anything is returned -- never filter after the fact.
"""

from dataclasses import dataclass, field, replace
from typing import List
import math
import json 
from pathlib import Path 

import ollama


EMBED_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL  = "qwen2.5:7b"
_CACHE = None

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    acl: List[str]
    embedding: List[float] = field(default_factory=list)
    score: float = 0.0


def load_documents(documents_dir: str = "documents") -> List[Chunk]:
    """Load and chunk all documents, attaching ACL metadata from manifest.json.

    Embeddings should be computed here (or lazily) using a real embedding
    model. Paragraph-level chunking is fine given how short these documents
    are.    
    """
    

    folder = Path(documents_dir)
    tags = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    chunks=[]
    for path in sorted(folder.glob("*.txt")):
        # Normalizing line endings  (\r\n) 
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        
        # Split paragraphs and filter out empty ones
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunk_size = 20
        chunk_id = 0 
        # chunks: List[Chunk] = []

        # Handle completely empty files gracefully
        if not paragraphs:
            print(f"Skipping {path.name}: File is empty.")
            continue

        # Safe lookup for manifest tags to prevent mid-script crashes
        file_acl = tags.get(path.name, {}).get("acl", [])

        for para in paragraphs:
            # for start in range(0, len(para), chunk_size):
            #     chunk_next = para[start:start + chunk_size]       
                
                chunks.append(
                    Chunk(
                        doc_id=path.name,
                        text=para,
                        acl=file_acl.copy(),
                        chunk_id=f"{path.name}#{chunk_id}",
                    )
                )
                chunk_id += 1
    texts = [chunk.text for chunk in chunks]     # (b) a Chunk object (comprehension-local)
        
        # print("Number of chunks:", len(texts))
        # print("First text type:", type(texts[0]))
           
    r = ollama.embed(model=EMBED_MODEL,input=texts)
    for chunk, vector in zip(chunks, r.embeddings):       # (c) a Chunk object again
        chunk.embedding = vector
    
    return chunks
    raise NotImplementedError("Implement chunking + embedding + ACL attachment")


def is_authorized(chunk_acl: List[str], user_groups: List[str]) -> bool:
    """A chunk is authorized for a user if 'all' is in its acl, or if any
    of the user's groups intersects the chunk's acl.
    """
    return 'all' in chunk_acl or any(
        item in chunk_acl for item in user_groups)
    raise NotImplementedError("Implement the authorization check")


def get_chunks() -> List[Chunk]:

    global _CACHE 
    if _CACHE is None: 
        _CACHE = load_documents()
    return _CACHE 

def _cosine(a, b) -> float:
    if len(a)!=len(b):
        raise ValueError('vectors must be same size')
    
    dot_prod = sum(x*y for x, y in zip(a,b))
    magnitude_a = math.sqrt(sum(x**2 for x in a))
    magnitude_b = math.sqrt(sum(y**2 for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")

    
    return dot_prod / (magnitude_a * magnitude_b)

def retrieve(query: str, user_groups: List[str], top_k: int = 3) -> List[Chunk]:
    """Return only chunks the given user_groups are authorized to see,
    ranked by embedding similarity to `query`.
    """
    chunks=get_chunks()
    matches=[]
    
    q = ollama.embed(model=EMBED_MODEL,input=query)
    
    for ch in chunks: 
        auth=is_authorized(ch.acl,user_groups)
        if auth==False: 
            continue    #skip unauthorizedd chunks 
        else: 
            scored_chunks=replace(ch, score=_cosine(q.embeddings[0],ch.embedding))
            matches.append(scored_chunks)
    results=sorted(matches, key=lambda m: m.score, reverse=True)[:top_k]
    # print(type(ch.embedding))
    # print(type(ch.embedding[0]))
    # print(ch.embedding[0])
    return results
    raise NotImplementedError("Implement permission-filtered, ranked retrieval")
