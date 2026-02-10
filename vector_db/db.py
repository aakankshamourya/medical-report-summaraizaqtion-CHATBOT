import numpy as np
import faiss
from load_medical_data import embeddings
#converting embeddting to vector fload32
vectors=np.array(embeddings).astype('float32')
# getting dimensions
dimensions=vectors.shape[1]

#craete iss index
index = faiss.IndexFlatL2(dimensions)
index.add(vectors)
print("Vectors stored:", index.ntotal)

