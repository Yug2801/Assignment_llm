import faiss
import numpy as np
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os
import pickle
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
# FAISS index for vector search
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
dimension = model.get_sentence_embedding_dimension()  # Ensure this matches your model's output dimension

index = faiss.IndexFlatL2(dimension)


def build_index(doc_chunks, embeddings):
    for embedding in embeddings:
        if embedding.shape[0] != dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} does not match expected dimension {dimension}")
        index.add(np.array([embedding]))  # Add each document's embedding to the index
    return index

def search_similar(index, query_embedding, k=5):
    query_embedding = np.array(query_embedding).reshape(1, -1)
    D, I = index.search(query_embedding, k)  
    return I[0]  

def get_embeddings(doc_chunks):
    embeddings = []
    for chunk in doc_chunks:
        embedding = compute_embedding(chunk)  # Replace with your embedding function
        embeddings.append(embedding)
    return embeddings

def compute_embedding(text):
    """Compute embeddings using the SentenceTransformer model."""
    embeddings = model.encode(text)
    return embeddings

def handle_query(query, doc_chunks, embeddings, index):
    if index is None:
        raise ValueError("The FAISS index has not been created.")
    
    query_embedding = compute_embedding(query)  # Compute embedding for the query
    similar_indices = search_similar(index, query_embedding)
    similar_chunks = [doc_chunks[i] for i in similar_indices if i < len(doc_chunks)]
    return similar_chunks




def save_index(index, file_path):
    faiss.write_index(index, file_path)

def load_index(file_path):
    return faiss.read_index(file_path)

def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def load_processed_data(data_file_path):
    with open(data_file_path, 'rb') as f:
        processed_data = pickle.load(f)
    return processed_data