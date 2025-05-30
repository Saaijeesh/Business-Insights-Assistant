import os
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langsmith import traceable  

@traceable(name="create_faiss_indexes_from_folder")
def create_faiss_indexes_from_folder(folder_path="data/Property_details", chunk_size=1000, overlap=200):
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    faiss_indexes = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                text = ' '.join([page.extract_text() for page in reader.pages if page.extract_text()])

            chunks = splitter.split_text(text)
            documents = [Document(page_content=chunk) for chunk in chunks]
            faiss_index = FAISS.from_documents(documents, embeddings)

            index_name = filename.replace('.pdf', '')
            faiss_indexes[index_name] = faiss_index

    return faiss_indexes
