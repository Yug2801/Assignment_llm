import streamlit as st
import os
import pickle
from faiss_module import build_index, compute_embedding, handle_query, save_index, load_index, save_embeddings, load_embeddings, load_processed_data
from process_text import process_text_and_query
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Function to list files in a directory
def list_files_in_directory(directory):
    files_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            files_list.append(os.path.join(root, file))
    return files_list

# Function to load documents from directories
def load_documents(directory_paths):
    all_documents = []
    for path in directory_paths:
        loader = DirectoryLoader(path)
        documents = loader.load()
        
        for doc in tqdm(documents, desc=f"Loading documents from {path}", unit="document"):
            all_documents.append(doc.page_content)
    
    return all_documents

# Function to preprocess documents into chunks
def preprocess_documents(documents, chunk_size=512, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))
    return chunks

# Function to save embeddings and index
def save_data_and_index(documents, doc_chunks):
    embeddings = [compute_embedding(chunk) for chunk in doc_chunks]
    index = build_index(doc_chunks, embeddings)

    embeddings_file_path = 'embeddings.pkl'
    index_file_path = 'faiss_index.index'

    save_embeddings(embeddings, embeddings_file_path)
    save_index(index, index_file_path)

    return embeddings, index

# Main Streamlit app
def main():
    st.title("Question Answering System")

    directories = [
        'data/milestone_papers_text',
        'data/lecture_notes'
    ]

    # Sidebar - Display directory files
    st.sidebar.header('List of Files in Directories')
    for directory in directories:
        st.sidebar.subheader(f"Directory: {directory}")
        files_list = list_files_in_directory(directory)
        for file in files_list:
            st.sidebar.write(file)

    # Load or preprocess documents
   
    query = st.text_input('Enter your question:')
    if st.button('Submit'):
        try:
            try:
                processed_data = load_processed_data('processed_data.pkl')
                documents, doc_chunks = processed_data['documents'], processed_data['doc_chunks']
                st.success("Processed data loaded successfully.")
            except FileNotFoundError:
                st.error("Processed data file ('processed_data.pkl') not found.")
            embeddings, index = load_embeddings('embeddings.pkl'), load_index('faiss_index.index')
            similar_chunks = handle_query(query, doc_chunks, embeddings, index)
            st.success("query handled successfully.")

            st.subheader("Similar Chunks:")
            for chunk in similar_chunks:
                st.write(chunk)

            # Save results to a file
            
            output_folder = r"E:\Assignment2\notebooks"
            os.makedirs(output_folder, exist_ok=True)

            filename = os.path.join(output_folder, "a.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                for chunk in similar_chunks:
                    f.write(chunk + '\n')
            response = process_text_and_query(output_folder, similar_chunks, query)
            response_text = response["key_info"]

            with open(filename, 'r', encoding='utf-8') as f:
                text_content = f.read()
            st.success("A.txt data loaded successfully.")

            text_content_position = response_text.find(text_content)

            if text_content_position != -1:
                response_text = response_text[text_content_position + len(text_content):]

                st.info(response_text)

        except NameError:
            st.error("Load documents first to initialize doc_chunks.")

    # Load processed data from file
    
if __name__ == "__main__":
    main()
