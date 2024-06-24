import os
from transformers import pipeline
from langchain.chains import SequentialChain, LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
def process_text_and_query(output_folder, similar_chunks, query):
    filename = os.path.join(output_folder, "a.txt")
    text_content=''
    with open(filename, 'r', encoding='utf-8') as f:
        text_content = f.read()

    llm1 = HuggingFaceHub(
        repo_id='meta-llama/Meta-Llama-3-8B-Instruct', 
        model_kwargs={'temperature': 0.7,'max-length':512}
    )

    preprocess_prompt = PromptTemplate.from_template("{query}. Write answer of the above question from the below text in one paragraph. {text}. ")
    analysis_prompt = PromptTemplate.from_template("Analysis the prompt and answer in detail in 100 words the question is: {query}. And answer the question from this give text: {text}")

    sequential_chain = SequentialChain(
        chains=[
            LLMChain(llm=llm1, prompt=preprocess_prompt, output_key="key_info"),
            LLMChain(llm=llm1, prompt=analysis_prompt, output_key="analysis")
        ],
        input_variables=["text","query"],
        output_variables=["analysis"]
    )

    response = sequential_chain({"text": text_content, "query": query})

    return response
