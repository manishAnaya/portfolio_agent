from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("🧠 Loading embeddings...")
embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY")
)

print("💾 Loading Vector DB...")
if os.path.exists("./my_db"):  
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
    session_id: str = "default"

# Health check endpoint
@app.get("/")
async def root():
    return {"status": "Manish Portfolio AI is running! 🚀"}

conversation_sessions = {}

# Main AI endpoint
@app.post("/ask")
async def ask(body: Question):

    # Get or create session history
    if body.session_id not in conversation_sessions:
        conversation_sessions[body.session_id] = []

    history = conversation_sessions[body.session_id]

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
    ]

    # Add past memory
    messages += history[-6:]

    messages.append(HumanMessage(content=body.question))

    response = llm.invoke(messages)

    # Save to history
    history.append(HumanMessage(content=body.question))
    history.append(response)

    # Keep only last 6 messages
    conversation_sessions[body.session_id] = history[-6:]
    print(conversation_sessions)
    return {
        "status": "success",
        "answer": response.content,
        "memory_size": len(conversation_sessions[body.session_id])
    }

@app.delete("/clear/{session_id}")
async def clear_session(session_id: str):
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    return {"status": "Session cleared!"}