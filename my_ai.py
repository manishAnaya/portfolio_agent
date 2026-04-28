import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_groq import ChatGroq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from question import Question

load_dotenv()

app = FastAPI()

app.add_middleware(CORSMiddleware,  allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

embedding = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))
print("Embedding model loaded")

if(os.path.exists("./ai_db")):
    vectorstore = Chroma(persist_directory="./ai_db", embedding_function=embedding)
    print("loaded existing db")
else:
    loader = TextLoader("manish_data.txt", encoding="utf-8")
    document = loader.load()
    print(f"document loaded with {len(document)} pages!")

    splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitters.split_documents(documents=document)
    print(f"loaded document splitted in {len(chunks)} chunks!")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./ai_db")

print("🤖 Loading LLM...")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

@app.get("/")
async def root():
    return {"status": "Manish Portfolio AI is running like rocket! 🚀"}

@app.post("/ask")
async def ask(body: Question):
    relevant_chunks = vectorstore.similarity_search(body.question, k=5)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    system_msg = SystemMessage(content=f"""
            You are a professional portfolio assistant
            for Manish Kumar Agrahari.
            Answer questions ONLY based on context below.
            If not in context say 'I don't have that info.'
            Be professional and confident.

            CONTEXT:
            {context}
        """)
    human_msg = HumanMessage(content=body.question)
    messages = [system_msg, human_msg]
    response = llm.invoke(messages)
    return {"status": "success", "response": response.content}