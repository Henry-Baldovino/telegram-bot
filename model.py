import os
from sentence_transformers import SentenceTransformer
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv


load_dotenv()

model = SentenceTransformer("distilroberta-base-paraphrase-v1")


def generation_response(question):
    pdf = open("China.pdf", "rb")
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )

    chunks = text_splitter.split_text(text=text)
    store_name = pdf.name[:-4]

    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

    print("Embeddings Computation Completed.")

    llm = OpenAI(tiktoken_model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    query = question
    docs = VectorStore.similarity_search(query=query, k=3)
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
    print(response)

    return response
