from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=model_name)

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

print(client)
print("##############")

db = Qdrant(client=client, embeddings=embeddings, collection_name="rag_db")

print(db)
print("######")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

def generate_answer(query, db, model, tokenizer, k=5):
    # Retrieve relevant documents
    docs = db.similarity_search_with_score(query=query, k=k)
    context = " ".join([doc.page_content for doc, _ in docs])
    
    # Prepare input for the generation model by concatenating context and query
    input_text = context + " " + query
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate answer
    outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

query = input("Enter your query: ")

answer = generate_answer(query, db, model, tokenizer)

print("Generated Answer:")
print(answer)
