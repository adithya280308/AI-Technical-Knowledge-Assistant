#Week 3 Rag Journey Mini Project

#importing necessary libraries
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI 
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import re

#loading documents from the data directory 
def load_documents():
    Data_Dir=Path("data")
    documents=[]
    for file_path in Data_Dir.glob("*.txt"):
        text=file_path.read_text(encoding="utf-8")
        documents.append(
            {
            "id":file_path.stem,
            "text":text,
            "source":file_path.name
            }
        )
    return documents

#creating chunks for the loaded documents using RecursiveCharacterTextSplitter
def create_chunks(documents):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks=[]
    for document in documents:
        document_chunks=splitter.split_text(document["text"])
        for index,chunk in enumerate(document_chunks):
            chunks.append(
                {
                "id":f"{document["id"]}_{index}",
                "text":chunk,
                "source":document["source"],
                "parent_id":document["id"]
                }
                )
    return chunks

#Creating embeddings for the chunks using Sentence Transformer
def create_embeddings(chunks):
    model=SentenceTransformer("all-MiniLM-L6-v2")
    texts=[chunk["text"] for chunk in chunks]
    embeddings=model.encode(texts)
    return model,embeddings

#Storing the emmbeddings in ChrmaDb
def store_embeddings(chunks,embeddings):
    client=chromadb.PersistentClient()

    collection=client.get_or_create_collection(
        name="Technical_Assistant"
    )

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{
            "source":chunk["source"],
            "parent_id":chunk["parent_id"]
        }for chunk in chunks
        ]
    )
    print("I stored the embeddings in ChromaDb successfully")

#Multi Query Retriever
def create_query_generator():
    llm=ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        timeout=30,
        max_retries=3
    )

    return llm

def create_multiple_versionsofquery(query,llm):
    prompt=f"""
You are a query expansion system for a technical knowledge base.

Generate 3 different search queries for the user's question.

The queries should:
- preserve the original meaning
- use different wording
- focus on important technical terms
- be useful for retrieving relevant documents

User question:
{query}

Return only the 3 queries, one per line.

"""
    print("Sending request to Gemini...")
    response=llm.invoke(prompt)

    print("Gemini response received!")
    print(response.content)
    
    queries = response.content[0]["text"].strip().splitlines()

    queries=[
        q.strip()
        for q in queries
        if q.strip()
    ]
    queries=[query]+queries
    return queries



#Faiss Algorithm 
#a]Creating Faiss index for the embeddings
def create_faiss_index(embeddings):
    print("Embeddings:", embeddings)
    print("Number of embeddings:", len(embeddings))

    for i, embedding in enumerate(embeddings):
        print(i, type(embedding), len(embedding))
    embeddings=np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    dimensions=embeddings.shape[1]
    index=faiss.IndexFlatIP(dimensions)
    index.add(embeddings)
    return index

#b]Performing similarity search using Faiss index for one version of query
def Retrieve_OneQueryResult_using_Faiss(query,model,index,chunks,k=3):
    query_embeddings=model.encode([query])
    query_embeddings=np.array(query_embeddings).astype("float32")
    faiss.normalize_L2(query_embeddings)
    scores,indices=index.search(query_embeddings,k)
    results=[]
    for score,result_index in zip(scores[0],indices[0]):
        if result_index!=-1:
            results.append(
                {
                "chunk":chunks[result_index],
                "score":float(score)
            }
            )
    return results
#c]Now Extending the Faiss Result for all versions of queries in final_result and returning it.....
def FinalResult_for_allversions_usingFaiss(queries,model,index,chunks,k=3):
    final_faiss_result=[]

    for query in queries:
        result=Retrieve_OneQueryResult_using_Faiss(query,model,index,chunks,k=k)
        final_faiss_result.extend(result)

    return final_faiss_result


#BM25 Algorithm(keyword based retrieval)
def create_bm25(chunks):
    tokenized_texts = [
    re.findall(r"\w+", chunk["text"].lower())
    for chunk in chunks
]

    bm25=BM25Okapi(tokenized_texts)
    return bm25

def  Retrieve_OneQueryResult_using_BM25(query,bm25,chunks,k=3):
    tokenized_query = re.findall(r"\w+", query.lower())
    scores=bm25.get_scores(tokenized_query)

    indices=scores.argsort()[::-1][:k]
    bm25_results=[]

    for index in indices:
        bm25_results.append(
            {
                "chunk":chunks[index],
                "score":scores[index]
            }
        )
    return bm25_results

#Cover for BM25
def FinalResult_for_allversions_usingBM25(queries,bm25,chunks,k=3):
    final_bm25_result=[]

    for query in queries:
        result=Retrieve_OneQueryResult_using_BM25(query,bm25,chunks,k=k)
        final_bm25_result.extend(result)

    return final_bm25_result


#Reciprocal Rank Fusion to get Final Result of Retrieved Chunks(Hybrid Retrieval)
def reciprocal_rank_fusion(final_bm25_results,final_faiss_results,k=60):
    fused_scores={}

    for rank,result in enumerate(final_bm25_results,start=1):
        chunk_id=result["chunk"]["id"]
        score=1/(k+rank)

        if(chunk_id not in fused_scores):
            fused_scores[chunk_id]=0

        fused_scores[chunk_id]+=score

    for rank,result in enumerate(final_faiss_results,start=1):
        chunk_id=result["chunk"]["id"]
        score=1/(k+rank)

        if(chunk_id not in fused_scores):
            fused_scores[chunk_id]=0

        fused_scores[chunk_id]+=score

    final_efficient_retrieved_chunks=sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return final_efficient_retrieved_chunks

#Getting the text from the retrieved chunks to Generated Augmented text

def create_retrieved_text(final_retrieved_chunks,chunks,top_k=3):
    context_chunks=[]

    for chunk_id,score in final_retrieved_chunks[:top_k]:
        for chunk in chunks:
            if(chunk_id==chunk["id"]):
                context_chunks.append(chunk["text"])
                break

    context="\n\n".join(context_chunks)
    return context

def generate_final_answer(query,context,llm):
        prompt = f"""
You are a technical assistant.

Answer the user's question using only the information provided in the context.

If the answer cannot be found in the context, say that the information is not available in the provided knowledge base.

Context:
{context}

User Question:
{query}

Answer:
"""
        response=llm.invoke(prompt)
        return response
    


#main function to load docs and create chunks
if __name__=="__main__":
    documents=load_documents()
    chunks=create_chunks(documents)

    print("The total number of chunks is:",len(chunks))

    for chunk in chunks[:3]:
        print("The first 3 chunks of the source:",chunk)

    model,embeddings=create_embeddings(chunks)
    print("The number of embeddings is:",len(embeddings))
    print("The dimension of the embedding is:",len(embeddings[0]))

    store_embeddings(chunks,embeddings)

    index=create_faiss_index(embeddings)
    print("The total number of Faiss indexes is:",index.ntotal)

    bm25=create_bm25(chunks)

    query="What is BM25?"
    llm=create_query_generator()
    queries=create_multiple_versionsofquery(query,llm)

    final_faiss_results=FinalResult_for_allversions_usingFaiss(queries,model,index,chunks,k=3)
    print("\nFaiss Results:")

    for rank,result in enumerate(final_faiss_results,start=1):

        print("\nRank:",rank)
        print("Chunk ID:",result["chunk"]["id"])
        print("FAISS Score:",result["score"])
        print("Text:",result["chunk"]["text"])

    final_bm25_results=FinalResult_for_allversions_usingBM25(queries,bm25,chunks,k=3)
    print("\nBM25 Results:")

    for rank,result in enumerate(final_bm25_results,start=1):

        print("\nRank:",rank)
        print("Chunk ID:",result["chunk"]["id"])
        print("BM25 Score:",result["score"])
        print("Text:",result["chunk"]["text"])
    
    final_efficient_chunks=reciprocal_rank_fusion(final_bm25_results,final_faiss_results,k=60)
    print("\nRRF results:")

    for rank,(chunk_id,score) in enumerate(
        final_efficient_chunks,
        start=1
    ):

        print("\nRank:",rank)
        print("Chunk ID:",chunk_id)
        print("RRF Score:",score)

    context=create_retrieved_text(final_efficient_chunks,chunks,top_k=3)
    print("\nContext:")
    print(context)

    response=generate_final_answer(query,context,llm)
    print("\nFinal Answer:")
    print(response)

    


    

    



    




                