# Medical Report Summarization Chatbot

An end-to-end **Retrieval Augmented Generation (RAG)** based chatbot for summarizing and answering questions about medical reports (PDF documents).
This project demonstrates a production-style NLP pipeline combining embeddings, vector search, and LLM generation.

---

##  Features

*  Upload and process medical PDF reports
*  Intelligent text chunking
*  Semantic search using vector embeddings
* FAISS vector database for fast retrieval
*  LLM-powered summarization & Q/A chatbot
*  End-to-end RAG architecture implementation

---

## 🧱 System Architecture

```
PDF Report
    ↓
Text Extraction
    ↓
Chunking
    ↓
Sentence Embeddings
    ↓
FAISS Vector Index
    ↓
User Query
    ↓
Retrieve Relevant Chunks
    ↓
LLM Prompt Injection
    ↓
Generated Answer
```

---

## 🛠️ Tech Stack

| Component     | Tool                 |
| ------------- | -------------------- |
| Language      | Python               |
| PDF Parsing   | PyPDF                |
| Embeddings    | SentenceTransformers |
| Vector DB     | FAISS                |
| LLM           | FLAN-T5              |
| Data Handling | NumPy, Pickle        |

---

## 📁 Project Structure

```
medical-report-summarization-chatbot/
│
├── load_medical_data.py      # PDF extraction + embeddings
├── chunking.py               # Text chunk processing
├── rag_pipeline.py           # Full RAG pipeline
├── script.py                 # Summarization utilities
├── vector_db/
│   ├── db.py                 # FAISS index creation
│   └── search.py             # Semantic retrieval
│
├── embeddings.npy
├── texts.pkl
├── requirements.txt
└── sample_report.pdf
```

---

## ⚙️ Installation

### Clone Repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### Create Environment

```bash
python -m venv ENV
ENV\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

### Step 1 — Generate Embeddings

```bash
python load_medical_data.py
```

Creates:

```
embeddings.npy
texts.pkl
```

---

### Step 2 — Run Chatbot

```bash
python rag_pipeline.py
```

Ask questions like:

```
What diagnosis is mentioned?
Summarize patient condition
Describe treatment timeline
```

---

## 🧠 How It Works

1. Extract text from medical PDF
2. Split into semantic chunks
3. Generate vector embeddings
4. Store embeddings in FAISS
5. Embed user query
6. Retrieve relevant chunks
7. Inject context into LLM prompt
8. Generate response

---

## 📌 Learning Goals

This project demonstrates:

* Retrieval Augmented Generation
* Vector databases
* Semantic search
* Prompt engineering
* LLM integration
* Real-world AI system architecture

---

## ⚠️ Disclaimer

This system is for **research and educational purposes only**.
It is **NOT a medical diagnostic tool**.



## ⭐ Future Improvements

* Web UI (Streamlit/React)
* Cloud vector DB (Pinecone/Qdrant)
* Better medical embedding models
* Response evaluation metrics
* Multi-document support
* Deployment via Docker/Kubernetes

---

## 📜 License

MIT License
