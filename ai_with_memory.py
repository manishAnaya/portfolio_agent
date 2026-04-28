import os
from dotenv import load_dotenv
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

embeddings = CohereEmbeddings(model= "embed-english-light-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))
print("Embedding model loaded")

if(os.path.exists("./memory_ai")):
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./memory_ai")
    print("loaded existing db")
else:
    loader = TextLoader("pytha.txt", encoding="utf-8")
    documents = loader.load()
    print(f"document loaded with {len(documents)} pages!")

    splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitters.split_documents(documents=documents)
    print(f"loaded document splitted in {len(chunks)} chunks!")

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./memory_ai")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

conversation_history = []
WINDOW_SIZE = 6

while True:
    user_input = input("User: ")

    if(user_input.lower() == "quit"):
        print("bye...")
        break

    relevant_chunks = vectorstore.similarity_search(user_input, k=5)

    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    messages = [
        SystemMessage(content=f"""
            You are a professional Mathematician
            expert in Pythagorean Theorem.
            Answer questions ONLY based on context below.
            If not in context say 'I don't have that info.'
            Be professional and confident.

            CONTEXT:
            {context}
        """),
        ]
    # Add past memory
    messages.extend(conversation_history)
  
    # Add user input
    messages.append(HumanMessage(content=user_input))
    
    # 🤖 Get response
    response = llm.invoke(messages)
    print(f"Assistant: {response.content}")

     # 💾 Save memory
    conversation_history.append(HumanMessage(content=user_input))
    conversation_history.append(response)
     # 🧹 Trim memory
    conversation_history = conversation_history[-WINDOW_SIZE:]
    print(f"💾 Memory: {len(conversation_history)} messages stored\n")