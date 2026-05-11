from dotenv import load_dotenv
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
import os
from typing import TypedDict

load_dotenv()

print("Loading Embeddings...")
embedding = CohereEmbeddings(model= "embed-english-light-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))
print("Embedding created...")

if os.path.exists("./ai_db"):
    vectorstore = Chroma(embedding_function=embedding, persist_directory="./ai_db")
    print("Loaded existing DB")
else:
    loader = TextLoader("manish_data.txt", encoding="utf-8")
    documents = loader.load()
    splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitters.split_documents(documents=documents)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./ai_db")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

class PortfolioState(TypedDict):
    question: str
    context: str
    answer: str

def searchNode(state: PortfolioState) -> PortfolioState:
    relevant_chunks = vectorstore.similarity_search(state["question"], k=5)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    return {"context": context}

def generate_node(state: PortfolioState) -> PortfolioState:
    messages = [
        SystemMessage(content=f"""
            You are a professional portfolio assistant
            for Manish Kumar Agrahari.
            Answer ONLY from context below.
            If not found say 'I dont have that info.'

            CONTEXT:
            {state['context']}
        """),
        HumanMessage(content=state['question'])
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(PortfolioState)

graph_builder.add_node("search", searchNode)
graph_builder.add_node("generate", generate_node)

graph_builder.add_edge(START, "search")
graph_builder.add_edge("search", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        print("Goodbye! 👋")
        break
    result = graph.invoke({
        "question": user_input,
        "context": "",
        "answer": ""
    })
    print(f"Assistant: {result['answer']}\n")