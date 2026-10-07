\# AI Technical Knowledge Assistant



A small Retrieval-Augmented Generation (RAG) project that answers technical questions using a custom knowledge base.



The project combines \*\*Multi-Query Retrieval, FAISS, BM25, and Reciprocal Rank Fusion (RRF)\*\* to improve the retrieval of relevant information before generating an answer using Gemini.



\## 🚀 Project Overview



The system takes a user's technical question and follows this pipeline:



User Question

↓

Multi-Query Generation

↓

Multiple Search Queries

↓

┌───────────────┐

│               │

FAISS          BM25

│               │

└───────┬───────┘

&#x20;       ↓

Reciprocal Rank Fusion (RRF)

&#x20;       ↓

Top Relevant Chunks

&#x20;       ↓

Retrieved Context

&#x20;       ↓

Gemini

&#x20;       ↓

Final Answer



\## 🧠 How It Works



\### 1. Document Loading



Technical knowledge is stored as `.txt` files inside the `data/` directory.



The application loads these documents using Python's `pathlib`.



\### 2. Document Chunking



The documents are split into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.



\- Chunk size: 500

\- Chunk overlap: 100



Each chunk receives a unique ID and metadata such as its source document.



\### 3. Embeddings



Each chunk is converted into a numerical vector using:



\*\*Sentence Transformers – `all-MiniLM-L6-v2`\*\*



These embeddings are used for semantic similarity search.



\### 4. ChromaDB



The project uses ChromaDB for storing document chunks and their metadata.



\### 5. Multi-Query Retrieval



Instead of searching using only the original question, Gemini generates multiple alternative versions of the query.



For example:



Original:



"What is BM25?"



Generated queries:



\- BM25 ranking function algorithm explained

\- Okapi BM25 information retrieval formula

\- How does the Best Matching 25 algorithm work



This helps retrieve relevant information using different formulations of the same question.



\### 6. FAISS Retrieval



FAISS is used for semantic similarity search.



The project uses:



\*\*FAISS IndexFlatIP\*\*



The embeddings are normalized before searching, allowing inner product similarity to work as cosine similarity.



\### 7. BM25 Retrieval



BM25 provides keyword-based retrieval.



Unlike FAISS, BM25 does not use embeddings.



It ranks documents based on factors including:



\- Term Frequency

\- Inverse Document Frequency

\- Document Length



\### 8. Hybrid Retrieval with RRF



The results from FAISS and BM25 are combined using \*\*Reciprocal Rank Fusion (RRF)\*\*.



The RRF score is calculated using:



RRF Score = 1 / (k + rank)



Chunks that appear highly ranked across the retrieval results receive higher combined scores.



\### 9. Context Generation



The highest-ranked chunks are selected and combined into a single context.



This context is provided to Gemini.



\### 10. Answer Generation



Gemini generates the final answer using the retrieved context.



The model is instructed to answer only using the information available in the retrieved knowledge base.



\## 🛠️ Technologies Used



\- Python

\- LangChain

\- Sentence Transformers

\- ChromaDB

\- FAISS

\- BM25

\- Google Gemini

\- NumPy



\## 📁 Project Structure



```text

ai-technical-knowledge-assistant/

│

├── data/

│   ├── bm25.txt

│   ├── embeddings.txt

│   ├── Faiss.txt

│   ├── machine\_learning.txt

│   ├── python.txt

│   ├── rag.txt

│   └── vector\_database.txt

│

├── main.py

├── requirements.txt

├── README.md

└── .gitignore

