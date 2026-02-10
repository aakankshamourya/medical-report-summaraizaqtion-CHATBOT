from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# -------- Extract text from PDF --------
reader = PdfReader("s41597-022-01899-x.pdf")

full_text = ""
for page in reader.pages:
    t = page.extract_text()
    if t:
        full_text += t + "\n"

print("Text length:", len(full_text))


# -------- Chunk properly --------
chunk_size = 500

chunks = [
    full_text[i:i+chunk_size]
    for i in range(0, len(full_text), chunk_size)
]

print("Total chunks:", len(chunks))
print("\nSample chunk:\n", chunks[0][:200])


# -------- Embed --------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    chunks,
    batch_size=32,
    show_progress_bar=True
)

# -------- Save --------
np.save("embeddings.npy", embeddings)

with open("texts.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("\nSaved clean embeddings + texts")
