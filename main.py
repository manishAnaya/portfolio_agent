from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("🧠 Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("💾 Loading Vector DB...")
if(os.path.exists("./my_db")):  
    vectorstore = Chroma(persist_directory="./my_db", embedding_function=embeddings)
    print("✅ Loaded existing DB!")
else:
    print("📄 Creating fresh DB...")
    loader = TextLoader("manish_data.txt", encoding="utf-8")
    documents = loader.load()
    splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitters.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./my_db")
    print("✅ DB Created!")

print("🤖 Loading LLM...")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

print("🚀 Server Ready!\n")

# Request model
class Question(BaseModel):
    question: str

# Health check endpoint
@app.get("/")
async def root():
    return {"status": "Manish Portfolio AI is running! 🚀"}

# Main AI endpoint
@app.post("/ask")
async def ask(body: Question):
    # Find relevant chunks
    relevant_chunks = vectorstore.similarity_search(body.question, k=5)

    # Build context
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

     # Send to LLM
    messages = [
        SystemMessage(content=f"""
            You are a professional portfolio assistant
            for Manish Kumar Agrahari.
            Answer questions ONLY based on context below.
            If not in context say 'I don't have that info.'
            Be professional and confident.

            CONTEXT:
            {context}
        """),
        HumanMessage(content=body.question)
    ]

    response = llm.invoke(messages)

    return {
        "status": "success",
        "answer": response.content
    }
