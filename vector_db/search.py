import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load saved data
vectors = np.load("../embeddings.npy").astype("float32")

with open("../texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Rebuild FAISS index
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -------- SEARCH FUNCTION --------
def search_medical(query, top_k=3):

    # Embed query
    q_vec = model.encode([query]).astype("float32")

    # Search FAISS
    distances, indices = index.search(q_vec, top_k)

    # Retrieve text
    results = []
    for idx in indices[0]:
        results.append(texts[idx])

    return results


# -------- TEST --------
if __name__ == "__main__":

    query = input("Enter medical query: ")

    results = search_medical(query)

    print("\nTop Relevant Chunks:\n")
    for i, r in enumerate(results):
        print(f"\nResult {i+1}")
        print("-"*40)
        print(r[:500])
