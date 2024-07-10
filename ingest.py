from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data.pdf")
documents = loader.load()

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
