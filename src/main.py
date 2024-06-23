from langchain_community.document_loaders import DirectoryLoader
import os
import pickle
from tqdm import tqdm
from faiss_module import build_index, compute_embedding, handle_query, save_index, load_index, save_embeddings, load_embeddings
from process_text import process_text_and_query
from langchain.text_splitter import RecursiveCharacterTextSplitter


def list_files_in_directory(directory):
    """List all files in the given directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))

def load_documents(directory_paths):
    all_documents = []
    for path in directory_paths:
        loader = DirectoryLoader(path)
        documents = loader.load()
        
        for doc in tqdm(documents, desc=f"Loading documents from {path}", unit="document"):
            all_documents.append(doc.page_content)
    
    return all_documents

def preprocess_documents(documents, chunk_size=512, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))
    return chunks

def save_data_to_file(documents, doc_chunks, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({'documents': documents, 'doc_chunks': doc_chunks}, f)

def load_data_from_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['documents'], data['doc_chunks']

def main():
    print("Welcome to the Question Answering System!")
    directories = [
        'E:/Assignment2/data/milestone_papers_text',
        'E:/Assignment2/data/lecture_notes'
    ]
    
    for directory in directories:
        print(f"Listing files in directory: {directory}")
        list_files_in_directory(directory)

    data_file_path = 'processed_data.pkl'
    embeddings_file_path = 'embeddings.pkl'
    index_file_path = 'faiss_index.index'

    if os.path.exists(data_file_path):
        print("Loading data from file...")
        documents, doc_chunks = load_data_from_file(data_file_path)
    else:
        print("Loading and preprocessing documents...")
        documents = load_documents(directories)
        print("Documents loaded successfully")
        doc_chunks = preprocess_documents(documents)
        print("Document chunks generated successfully")
        save_data_to_file(documents, doc_chunks, data_file_path)
        print(f"Data saved to {data_file_path}")
    if os.path.exists(embeddings_file_path) and os.path.exists(index_file_path):
        print("Loading embeddings and index from files...")
        embeddings = load_embeddings(embeddings_file_path)
        index = load_index(index_file_path)
    else:
        print("Generating embeddings and building index...")
        embeddings = [compute_embedding(chunk) for chunk in doc_chunks]
        print("Embeddings generated successfully")
        index = build_index(doc_chunks, embeddings)
        save_embeddings(embeddings, embeddings_file_path)
        save_index(index, index_file_path)
        print(f"Embeddings saved to {embeddings_file_path}")
        print(f"Index saved to {index_file_path}")

    if index is not None:
        print("Index built successfully")
    else:
        print("Index creation failed")
        return


    while True:
        query = input("Please enter your question (type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting the system...")
            break

        similar_chunks = handle_query(query, doc_chunks, embeddings, index)
        print("Query handled successfully")
        
        print("Similar chunks:")
        for chunk in similar_chunks:
            print(chunk)

        output_folder = r"E:\Assignment2\notebooks"
        os.makedirs(output_folder, exist_ok=True)

        filename = os.path.join(output_folder, "a.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            for chunk in similar_chunks:
                f.write(chunk + '\n')

        response = process_text_and_query(output_folder, similar_chunks, query)
        response_text = response["key_info"]

        output_folder = r"E:\Assignment2\notebooks"
        filename = os.path.join(output_folder, "a.txt")

        with open(filename, 'r', encoding='utf-8') as f:
            text_content = f.read()

        text_content_position = response_text.find(text_content)
        if text_content_position != -1:
            response_text = response_text[text_content_position + len(text_content):]

        print(response_text)

if __name__ == "__main__":
    main()
