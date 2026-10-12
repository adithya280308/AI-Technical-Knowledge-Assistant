# AI Technical Knowledge Assistant

An AI-powered technical knowledge assistant that uses **Advanced Retrieval-Augmented Generation (RAG)** to answer technical questions using information from a local document knowledge base.

The system combines **Multi-Query Retrieval, FAISS, BM25, and Reciprocal Rank Fusion (RRF)** to retrieve relevant information and generate context-grounded answers using Google Gemini.

## Overview

Traditional vector search can miss relevant information when a user's query uses different terminology from the source documents. This project addresses that limitation by expanding queries, combining semantic and keyword-based retrieval, and fusing their rankings before generating an answer.

The goal is to improve retrieval quality and provide answers based on the available technical documents rather than relying solely on the language model's internal knowledge.

## Key Features

- **Document Loading:** Loads `.txt` files from a local knowledge base.
- **Intelligent Text Chunking:** Uses `RecursiveCharacterTextSplitter` with configurable chunk size and overlap.
- **Semantic Embeddings:** Generates text embeddings using Sentence Transformers and the `all-MiniLM-L6-v2` model.
- **Multi-Query Retrieval:** Uses Google Gemini to generate alternative search queries while preserving the original question.
- **FAISS Vector Search:** Performs semantic similarity search using normalized embeddings and a Flat Inner Product index.
- **BM25 Keyword Search:** Retrieves documents based on keyword relevance.
- **Hybrid Retrieval:** Combines semantic and keyword-based retrieval results.
- **Reciprocal Rank Fusion:** Merges rankings from FAISS and BM25 to prioritize chunks that rank well across retrieval results.
- **Context Construction:** Selects the highest-ranked chunks to construct the context supplied to the LLM.
- **Context-Grounded Generation:** Uses Google Gemini to generate answers based on the retrieved context.
- **Persistent Vector Storage:** Uses ChromaDB to store document chunks and their metadata.

## System Architecture

**Document Ingestion and Indexing**

1. Load technical documents from the `data/` directory.
2. Split documents into overlapping text chunks.
3. Generate embeddings using Sentence Transformers.
4. Build a FAISS index and a BM25 keyword index.
5. Store document chunks and metadata in ChromaDB.

**Query Processing and Answer Generation**

1. Receive a user's technical question.
2. Generate alternative queries using Google Gemini.
3. Retrieve candidate chunks using FAISS and BM25 for each query.
4. Fuse the retrieval rankings using Reciprocal Rank Fusion.
5. Select the top-ranked chunks and construct the context.
6. Send the original question and retrieved context to Google Gemini.
7. Return the generated answer.

## Architecture Diagram

```text
             Technical Documents
                     |
                     v
              Document Loading
                     |
                     v
                Text Chunking
                     |
                     v
             Sentence Transformers
               Embedding Model
                     |
          +----------+----------+
          |                     |
          v                     v
     FAISS Index           BM25 Index
   Semantic Search       Keyword Search
          |                     |
          +----------+----------+
                     |
                     v
             Multi-Query Retrieval
           Original + Expanded Queries
                     |
                     v
        Reciprocal Rank Fusion (RRF)
                     |
                     v
           Top-Ranked Text Chunks
                     |
                     v
              Context Assembly
                     |
                     v
              Google Gemini
                     |
                     v
              Final Answer
```

*Note: ChromaDB is currently used for persistent document and metadata storage. The retrieval flow shown above uses FAISS and BM25.*

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| LangChain Text Splitters | Recursive document chunking |
| Sentence Transformers | Text embedding generation |
| FAISS | Semantic vector search |
| BM25 / `rank-bm25` | Keyword-based retrieval |
| Reciprocal Rank Fusion | Combining retrieval rankings |
| Google Gemini | Query expansion and answer generation |
| ChromaDB | Persistent vector-store and document metadata storage |
| NumPy | Numerical operations on embeddings |
| Regular Expressions | Query and document tokenization |

## Project Structure

```text
AI-Technical-Knowledge-Assistant/
│
├── data/
│   ├── python.txt
│   ├── rag.txt
│   └── other_technical_documents.txt
│
├── chroma.sqlite3
├── main.py
├── requirements.txt
└── README.md
```

*The structure above is illustrative. Adjust the filenames to match your actual repository. ChromaDB may also create its database files inside a configured persistence directory.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/adithya280308/AI-Technical-Knowledge-Assistant.git
cd AI-Technical-Knowledge-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install the required libraries:

```bash
pip install langchain-text-splitters sentence-transformers chromadb langchain-google-genai faiss-cpu numpy rank-bm25
```

### 4. Configure your Google Gemini API key

Create an API key through [Google AI Studio](https://aistudio.google.com/).

Set the environment variable before running the application.

**Windows PowerShell:**

```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

Keep your API key private. Never commit API keys or other secrets to GitHub.

### 5. Add your technical documents

Create a `data/` directory in the project root and add your `.txt` knowledge-base documents.

### 6. Run the application

```bash
python main.py
```

The application loads the documents, prepares retrieval indexes, expands the query, retrieves relevant chunks, combines their rankings, and generates an answer using the retrieved context.

## Retrieval Techniques Explained

### Multi-Query Retrieval

Generates alternative formulations of the original question to improve the chances of finding relevant information despite differences in wording.

### FAISS Semantic Search

Uses vector embeddings and similarity search to identify chunks that are semantically related to the user's question. The implementation uses normalized embeddings with `IndexFlatIP`.

### BM25 Keyword Retrieval

Ranks chunks based on lexical overlap and term importance, helping retrieve content containing relevant technical terms, acronyms, and keywords.

### Reciprocal Rank Fusion

Combines rankings from multiple retrieval results using rank-based scoring. It allows semantic and keyword retrievers to contribute to a shared ranking without requiring their raw scores to be directly comparable.

### Context-Grounded Generation

Supplies the highest-ranked retrieved chunks alongside the original question. The prompt instructs Gemini to use the provided context and report when the answer is unavailable in the knowledge base.

## Current Scope

The current implementation focuses on document ingestion, query expansion, hybrid retrieval, rank fusion, and context-grounded answer generation.

Potential future improvements include:

- Cross-encoder reranking for improved candidate ordering.
- Evaluation using TruLens and the RAG Triad: context relevance, groundedness, and answer relevance.
- Automated retrieval and answer-quality testing.
- A FastAPI backend and interactive user interface.
- Support for additional document formats such as PDF and Markdown.
- Improved indexing workflows for updating documents without duplicating stored records.

## Learning Outcomes

This project demonstrates practical implementation of:

- Advanced RAG pipeline design.
- Semantic and lexical information retrieval.
- Embedding generation and vector similarity search.
- Query expansion and hybrid retrieval.
- Rank fusion and context construction.
- LLM-based answer generation with retrieved evidence.
- Integration of multiple Python libraries into a modular AI application.

## Author

**Adithya**

Computer Science and Engineering Student | Aspiring AI Engineer

GitHub: [@adithya280308](https://github.com/adithya280308)

---

*Built as a hands-on project to explore Advanced RAG, hybrid information retrieval, and AI engineering.*



