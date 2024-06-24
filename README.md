# Automated Question Answering System

## Introduction

Welcome to the Automated Question Answering System project! This project leverages advanced natural language processing (NLP) and machine learning techniques to create a system capable of processing large textual datasets, analyzing them, and generating coherent responses to user queries.

## Features

- **Data Extraction**: Load and read text documents from specified directories.
- **Data Preprocessing**: Clean and chunk text data for processing.
- **Embedding Generation**: Generate embeddings using the `meta-llama/Meta-Llama-3-8B-Instruct` model.
- **Index Building**: Use FAISS to build an index for efficient similarity search.
- **Query Handling**: Retrieve relevant document chunks and generate answers using Hugging Face models.
- **Deployment**: Deploy the application using Streamlit for an interactive user experience.

## Requirements

- Python 3.8 or higher
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io)
- [TQDM](https://tqdm.github.io)
- [LangChain](https://github.com/langchain/langchain)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/automated-question-answering-system.git
    cd automated-question-answering-system
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up Hugging Face API Token:**
    Store your Hugging Face API token as an environment variable:
    ```bash
    export HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token  # On Windows use `set HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token`
    ```

## Usage

1. **Prepare Data:**
    - Place your text files in the `data/milestone_papers_text` and `data/lecture_notes` directories.

2. **Run the Streamlit Application:**
    ```bash
    streamlit run src/app.py
    ```

3. **Interact with the Application:**
    - Enter your query in the text box and get responses based on the provided data.

## File Structure

```
automated-question-answering-system/
│
├── data/
│   ├── milestone_papers_text/
│   └── lecture_notes/
│
├── src/
│   ├── app.py
│   ├── faiss_module.py
│   ├── process_text.py
│
├── requirements.txt
└── README.md
```

## Contributing

We welcome contributions to enhance the functionality and performance of this project. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your branch.
4. Open a pull request with a detailed description of your changes.

## Acknowledgements

- [Hugging Face](https://huggingface.co)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io)
- [LangChain](https://github.com/langchain/langchain)
