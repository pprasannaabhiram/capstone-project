
import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_store")

def build_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection("zepto_policies")
    except Exception:
        pass
    collection = client.create_collection("zepto_policies")

    doc_ids, texts, metadatas = [], [], []
    for fname in sorted(os.listdir(DOCS_DIR)):
        if fname.endswith(".txt"):
            path = os.path.join(DOCS_DIR, fname)
            with open(path, "r") as f:
                text = f.read().strip()
            doc_id = fname.replace(".txt", "")
            doc_ids.append(doc_id)
            texts.append(text)
            metadatas.append({"source": fname})

    embeddings = model.encode(texts).tolist()
    collection.add(ids=doc_ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed {len(doc_ids)} documents into ChromaDB collection 'zepto_policies'.")
    return collection

if __name__ == "__main__":
    build_index()
