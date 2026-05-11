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
    chunks = splitters.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./ai_db")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

class PortfolioState(TypedDict):
    question: str
    context: str
    answer: str
    source: str

def searchNode(state: PortfolioState) -> PortfolioState:
    relevant_chunks = vectorstore.similarity_search(state["question"], k=5)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    return {"context": context}

def check_relevance(state: PortfolioState) -> str:
    messages = [
        SystemMessage(content="""
            You are a relevance checker.
            Check if the CONTEXT can answer the QUESTION.
            Reply with ONLY one word:
            'relevant' if context answers the question
            'irrelevant' if context cannot answer
        """),
        HumanMessage(content=f"""
            QUESTION: {state['question']}
            CONTEXT: {state['context']}
        """)
    ]
    response = llm.invoke(messages)
    result = response.content.strip().lower()
    print(f"✅ Relevance check: {result}")
    if "irrelevant" in result:
        return "irrelevant"
    else:
        return "relevant"

def generate_from_cv(state: PortfolioState) -> PortfolioState:
    messages = [
        SystemMessage(content=f"""
            You are a professional portfolio assistant
            for Manish Kumar Agrahari.
            Answer ONLY from context below.
            Be professional and confident.

            CONTEXT:
            {state['context']}
        """),
        HumanMessage(content=state['question'])
    ]
    response = llm.invoke(messages)
    print("✅ Answer generated from CV!")
    return {"answer": response.content, "source": "CV Data"}


def handle_unknown(state: PortfolioState) -> PortfolioState:
    messages = [
        SystemMessage(content="""
            You are a portfolio assistant for Manish Kumar Agrahari.
            The user asked something not in the CV data.
            Politely say you don't have that information
            and suggest they contact Manish directly.
            Always provide his email: manish.ag555@gmail.com
        """),
        HumanMessage(content=state['question'])
    ]
    response = llm.invoke(messages)
    print("✅ Handled unknown question!")
    return {"answer": response.content, "source": "Fallback"}

graph_builder = StateGraph(PortfolioState)

graph_builder.add_node("search", searchNode)
graph_builder.add_node("generate", generate_from_cv)
graph_builder.add_node("handle", handle_unknown)

graph_builder.add_edge(START, "search")

graph_builder.add_conditional_edges("search", check_relevance, {"relevant": "generate", "irrelevant": "handle"})

graph_builder.add_edge("generate", END)
graph_builder.add_edge("handle", END)

graph = graph_builder.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        print("Goodbye! 👋")
        break
    result = graph.invoke({
        "question": user_input,
        "context": "",
        "answer": "",
        "source": ""
    })
    print(f"Assistant: {result['answer']}")
    print(f"Source: {result['source']}\n")