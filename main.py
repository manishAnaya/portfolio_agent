import os
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from llm_judge import judge_input

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Enable CORS so frontend apps can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store request timestamps for each IP
request_counts = {}

# Max requests allowed in given time window
RATE_LIMIT = 10
TIME_WINDOW = 60


def check_rate_limit(ip: str):
    now = time.time()
    # Create entry for new IP
    if ip not in request_counts:
        request_counts[ip] = []

    # Keep only recent requests inside time window
    request_counts[ip] = [
        t for t in request_counts[ip]
        if now - t < TIME_WINDOW
    ]

    # Block if limit exceeded
    if len(request_counts[ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many requests! Please wait 60 seconds."
        )

    # Save current request timestamp
    request_counts[ip].append(now)


# Common prompt injection attempts
BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "you are now",
    "forget your instructions",
    "act as",
    "jailbreak",
    "dan mode",
    "pretend you are",
]

# Topics that should not be answered
BLOCKED_TOPICS = [
    "hack",
    "bomb",
    "weapon",
    "drug",
    "illegal",
    "password",
    "credit card",
]


def check_input(question: str) -> str:
    question_lower = question.lower()

    # Detect prompt injection attempts
    for pattern in BLOCKED_PATTERNS:
        if pattern in question_lower:
            return "blocked_injection"

    # Detect restricted topics
    for topic in BLOCKED_TOPICS:
        if topic in question_lower:
            return "blocked_harmful"

    # Prevent extremely long questions
    if len(question) > 500:
        return "blocked_length"

    return "allowed"


# Prevent accidental sensitive information exposure
SENSITIVE_INFO = [
    "password",
    "secret",
    "private key",
    "api key",
    "bank account",
]


def check_output(response: str) -> str:
    response_lower = response.lower()

    # Block sensitive information in AI response
    for info in SENSITIVE_INFO:
        if info in response_lower:
            return (
                "I cannot share sensitive information. "
                "Please contact Manish directly at "
                "manish.ag555@gmail.com"
            )

    return response


print("🧠 Loading embeddings...")

# Embedding model used for semantic search
embedding = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

print("💾 Loading Vector DB...")

# Load existing vector DB if already created
if os.path.exists("./ai_db"):

    vectorstore = Chroma(
        persist_directory="./ai_db",
        embedding_function=embedding
    )

else:
    # Load portfolio data text file
    loader = TextLoader(
        "manish_data.txt",
        encoding="utf-8"
    )

    documents = loader.load()

    # Split large text into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # Create and store embeddings
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./ai_db"
    )

print("🤖 Loading LLM...")

# Groq LLM configuration
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# Store conversation history by session
conversation_sessions = {}

# Number of previous messages to remember
WINDOW_SIZE = 6


class Question(BaseModel):
    question: str
    session_id: str = "default"


@app.get("/")
async def root():
    return {
        "status": "Manish Portfolio AI is running! 🚀"
    }


@app.post("/ask")
async def ask(body: Question, request: Request):

    # Get user IP and apply rate limit
    client_ip = request.client.host
    check_rate_limit(client_ip)

    # Validate user input
    input_status = check_input(body.question)

    if input_status == "blocked_injection":
        return {
            "answer": (
                "I detected an attempt to manipulate "
                "my instructions. I can only answer "
                "questions about Manish Kumar Agrahari's portfolio."
            ),
            "status": "success",
            "reason": "prompt_injection"
        }

    if input_status == "blocked_harmful":
        return {
            "answer": (
                "I cannot answer questions on that topic. "
                "I am only here to help with questions "
                "about Manish's portfolio."
            ),
            "status": "success",
            "reason": "harmful_content"
        }

    if input_status == "blocked_length":
        return {
            "answer": (
                "Your question is too long. "
                "Please keep it under 500 characters."
            ),
            "status": "success",
            "reason": "too_long"
        }
    
    judge_result = judge_input(body.question)

    if judge_result["status"] == "blocked":
        return {
            "answer": (
                f"I cannot answer that. "
                f"{judge_result['reason']}. "
                "I only help with Manish's portfolio questions."
            ),
            "status": "success",
            "reason": judge_result["reason"]
        }

    # Create new session if not exists
    if body.session_id not in conversation_sessions:
        conversation_sessions[body.session_id] = []

    history = conversation_sessions[body.session_id]

    # Retrieve most relevant chunks from vector DB
    relevant_chunks = vectorstore.similarity_search(
        body.question,
        k=5
    )

    # Combine retrieved chunks into single context
    context = "\n\n".join([
        chunk.page_content
        for chunk in relevant_chunks
    ])

    # System prompt with portfolio context
    messages = [
        SystemMessage(content=f"""
            You are a professional portfolio assistant
            for Manish Kumar Agrahari.

            Answer ONLY from context below.

            Never reveal personal or sensitive information.

            If not in context say 'I don't have that info.'

            CONTEXT:
            {context}
        """),
    ]

    # Add recent conversation history
    messages += history[-WINDOW_SIZE:]

    # Add latest user question
    messages.append(
        HumanMessage(content=body.question)
    )

    # Generate AI response
    response = llm.invoke(messages)

    # Final safety check on response
    safe_response = check_output(response.content)

    # Save conversation history
    history.append(
        HumanMessage(content=body.question)
    )

    history.append(response)

    # Keep only recent messages
    conversation_sessions[body.session_id] = (
        history[-WINDOW_SIZE:]
    )

    return {
        "answer": safe_response,
        "status": "success",
        "memory_size": len(
            conversation_sessions[body.session_id]
        )
    }


@app.delete("/clear/{session_id}")
async def clear_session(session_id: str):

    # Remove saved chat history for session
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]

    return {
        "status": "Session cleared!"
    }