import pickle
from load_medical_data import texts

# Convert to list to avoid HF lazy object issues
texts = list(texts)

with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("Texts saved:", len(texts))
