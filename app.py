from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=model_name)

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

print(client)
print("##############")

db = Qdrant(client=client, embeddings=embeddings, collection_name="hr_db")

print(db)
print("######")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

def generate_answer(query):
    docs = db.similarity_search_with_score(query=query, k=5)
    context = " ".join([doc.page_content for doc, _ in docs])
    
    input_text = context + " " + query
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

iface = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs="text",
    title="PDF Query System",
    description="Enter a query to search the embedded PDF documents and get a context-based answer."
)

iface.launch()