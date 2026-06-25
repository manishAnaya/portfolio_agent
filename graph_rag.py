from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os

load_dotenv()

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def run_cypher(query: str) -> list:
    """Run Cypher query on Neo4j"""
    try:
        with driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]
    except Exception as e:
        return [{"error": str(e)}]

def generate_cypher(question: str) -> str:
    """Ask LLM to write Cypher query"""
    messages = [
        SystemMessage(content="""
            You are a Neo4j Cypher expert.
            Write Cypher queries for a portfolio graph database.

            Graph Schema:
            - (Person) nodes with properties: name, title, location, experience, email
            - (Company) nodes with properties: name, location
            - (Skill) nodes with properties: name
            - (Project) nodes with properties: name, type, status, rating

            Relationships:
            - (Person)-[:WORKS_AT]->(Company)
            - (Person)-[:WORKED_AT]->(Company)
            - (Person)-[:KNOWS]->(Skill)
            - (Person)-[:LEARNING]->(Skill)
            - (Person)-[:BUILT]->(Project)
            - (Project)-[:USES]->(Skill)

            Rules:
            - Return ONLY the Cypher query
            - No explanation, no markdown, no backticks
            - Always use RETURN to get results
            - Person name is 'Manish Kumar Agrahari'
        """),
        HumanMessage(content=f"Write Cypher query for: {question}")
    ]

    response = llm.invoke(messages)
    return response.content.strip()

def graph_rag(question: str) -> str:
    """
    Full Graph RAG pipeline!
    Question → Cypher → Neo4j → Context → Answer
    """
    print(f"\n❓ Question: {question}")

    # Step 1 - Generate Cypher query
    print("📝 Generating Cypher query...")
    cypher = generate_cypher(question)
    print(f"🔍 Cypher: {cypher}")

    # Step 2 - Run on Neo4j
    print("⚡ Running on Neo4j...")
    results = run_cypher(cypher)
    print(f"✅ Got {len(results)} results!")

    # Step 3 - Convert to context
    context = f"Graph query results: {results}"

    # Step 4 - Generate final answer
    messages = [
        SystemMessage(content=f"""
            You are a professional portfolio assistant
            for Manish Kumar Agrahari.
            Answer based on the graph data below.
            Be professional and friendly.

            GRAPH DATA:
            {context}
        """),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)
    return response.content

# Test it!
questions = [
    "What skills does Manish know?",
    "What projects has Manish built?",
    "Which projects use AI skills?",
    "Where has Manish worked?",
    "What is Manish currently learning?",
]

for q in questions:
    answer = graph_rag(q)
    print(f"\n🤖 Answer: {answer}")
    print("-" * 50)

driver.close()