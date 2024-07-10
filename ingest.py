import os
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

pdf_folder_path = "C:\\Users\\DELL\\Desktop\\RAG\\Pdf"

def load_pdfs_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

documents = load_pdfs_from_folder(pdf_folder_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=model_name)

print('Embedding Model Loading...')
url = "http://localhost:6333"

qdrant = Qdrant.from_documents(
    documents=texts,
    embedding=embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="v_db"
)

print("Vector DB Successfully Created!")
