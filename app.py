from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=model_name)

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

print(client)
print("##############")

db = Qdrant(client=client, embeddings=embeddings, collection_name="v_db")

print(db)
print("######")

query = "How has the COVID-19 pandemic impacted the academic performance, mental health, and social development of students across different educational levels?"

docs = db.similarity_search_with_score(query=query, k=5)
for doc, score in docs:
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})
