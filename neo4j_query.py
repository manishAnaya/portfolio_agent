from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)

def query_graph(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]

# Query 1 - What skills does Manish know?
print("📚 Skills Manish knows:")
results = query_graph("""
    MATCH (m:Person {name: 'Manish Kumar Agrahari'})
          -[:KNOWS]->(s:Skill)
    RETURN s.name as skill, 
           s.level as level
""")
for r in results:
    print(f"  → {r}")

# Query 2 - What is Manish learning?
print("\n🎓 Skills Manish is learning:")
results = query_graph("""
    MATCH (m:Person {name: 'Manish Kumar Agrahari'})
          -[:LEARNING]->(s:Skill)
    RETURN s.name as skill
""")
for r in results:
    print(f"  → {r}")

# Query 3 - What projects has Manish built?
print("\n🚀 Projects Manish built:")
results = query_graph("""
    MATCH (m:Person)-[:BUILT]->(p:Project)
    RETURN p.name as project,
           p.type as type
""")
for r in results:
    print(f"  → {r}")

# Query 4 - What skills does AI Chatbot use?
print("\n🤖 AI Chatbot Platform uses:")
results = query_graph("""
    MATCH (p:Project {name: 'AI Chatbot Platform'})
          -[:USES]->(s:Skill)
    RETURN s.name as skill
""")
for r in results:
    print(f"  → {r}")

# Query 5 - Companies Manish worked at
print("\n🏢 Companies Manish worked at:")
results = query_graph("""
    MATCH (m:Person)-[r:WORKS_AT|WORKED_AT]->(c:Company)
    RETURN c.name as company,
           c.location as location,
           type(r) as relationship
""")
for r in results:
    print(f"  → {r}")

# Query 6 - Multi hop! Projects that use AI skills
print("\n🔗 Projects using AI skills:")
results = query_graph("""
    MATCH (m:Person)-[:BUILT]->(p:Project)
          -[:USES]->(s:Skill)
    WHERE s.name IN ['LangChain', 'RAG', 'LangGraph']
    RETURN p.name as project,
           s.name as ai_skill
""")
for r in results:
    print(f"  → {r}")

driver.close()
print("\n✅ All queries done!")