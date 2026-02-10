import numpy as np
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
vectors = np.load("embeddings.npy").astype("float32")

with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)


# -------------------------------------------------
# REBUILD FAISS INDEX
# -------------------------------------------------
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load model properly
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

generator = pipeline(
    "text-generation",     # safe task name
    model=model,
    tokenizer=tokenizer
)




# -------------------------------------------------
# RETRIEVAL FUNCTION
# -------------------------------------------------
def retrieve_chunks(query, top_k=3):
    
    q_vec = embed_model.encode([query]).astype("float32")

    distances, indices = index.search(q_vec, top_k)

    results = [texts[i] for i in indices[0]]

    print("\n--- DEBUG RETRIEVED CHUNKS ---")
    for r in results:
        print(r[:200])
        print("-----")

    return results

# -------------------------------------------------
# GENERATION FUNCTION
# -------------------------------------------------
def generate_answer(query):

    chunks = retrieve_chunks(query)

    context = "\n\n".join(chunks)

    prompt = f"""
Answer the medical question using the context.

Context:
{context}

Question:
{query}

Provide a clear medical summary:
"""


    result = generator(prompt, max_length=300)

    return result[0]["generated_text"]


# -------------------------------------------------
# INTERACTIVE LOOP
# -------------------------------------------------
if __name__ == "__main__":

    while True:
        query = input("\nAsk medical question (or quit): ")

        if query.lower() == "quit":
            break

        answer = generate_answer(query)

        print("\n===== AI RESPONSE =====\n")
        print(answer)
