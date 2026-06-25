from dotenv import load_dotenv
from neo4j import GraphDatabase
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

# Test connection
driver.verify_connectivity()
print("✅ Connected to Neo4j!")

# Create some nodes and relationships about Manish
def create_portfolio_graph(driver):
    with driver.session() as session:

        # Clear existing data first
        session.run("MATCH (n) DETACH DELETE n")
        print("🗑️ Cleared existing data!")

        # Create Person node
        session.run("""
            CREATE (m:Person {
                name: 'Manish Kumar Agrahari',
                title: 'Flutter Developer',
                location: 'Bengaluru',
                experience: '5+ years',
                email: 'manish.ag555@gmail.com'
            })
        """)
        print("✅ Created Person node!")

        # Create Company nodes
        session.run("""
            CREATE (f:Company {name: 'Finva Tech Pvt Ltd', location: 'Bengaluru'})
            CREATE (a:Company {name: 'Act T Connect', location: 'Jhansi'})
            CREATE (i:Company {name: 'IBigDo Technologies', location: 'Noida'})
        """)
        print("✅ Created Company nodes!")

        # Create Skill nodes
        session.run("""
            CREATE (fl:Skill {name: 'Flutter'})
            CREATE (d:Skill {name: 'Dart'})
            CREATE (r:Skill {name: 'Riverpod'})
            CREATE (lc:Skill {name: 'LangChain'})
            CREATE (rag:Skill {name: 'RAG'})
            CREATE (lg:Skill {name: 'LangGraph'})
            CREATE (ca:Skill {name: 'Clean Architecture'})
            CREATE (fb:Skill {name: 'Firebase'})
        """)
        print("✅ Created Skill nodes!")

        # Create Project nodes
        session.run("""
            CREATE (p1:Project {
                name: 'AI Chatbot Platform',
                type: 'AI + Flutter',
                status: 'Production'
            })
            CREATE (p2:Project {
                name: 'Portfolio Risk Management',
                type: 'Fintech + Flutter',
                status: 'Production'
            })
            CREATE (p3:Project {
                name: 'Recruitment Platform',
                type: 'HR + Flutter',
                status: 'Production',
                rating: '4.8'
            })
        """)
        print("✅ Created Project nodes!")

        # Create RELATIONSHIPS!
        session.run("""
            MATCH (m:Person {name: 'Manish Kumar Agrahari'})
            MATCH (f:Company {name: 'Finva Tech Pvt Ltd'})
            MATCH (a:Company {name: 'Act T Connect'})
            MATCH (i:Company {name: 'IBigDo Technologies'})
            MATCH (fl:Skill {name: 'Flutter'})
            MATCH (d:Skill {name: 'Dart'})
            MATCH (r:Skill {name: 'Riverpod'})
            MATCH (lc:Skill {name: 'LangChain'})
            MATCH (rag:Skill {name: 'RAG'})
            MATCH (lg:Skill {name: 'LangGraph'})
            MATCH (ca:Skill {name: 'Clean Architecture'})
            MATCH (fb:Skill {name: 'Firebase'})
            MATCH (p1:Project {name: 'AI Chatbot Platform'})
            MATCH (p2:Project {name: 'Portfolio Risk Management'})
            MATCH (p3:Project {name: 'Recruitment Platform'})

            
            CREATE (m)-[:WORKS_AT {since: 'Mar 2024', current: true}]->(f)
            CREATE (m)-[:WORKED_AT {from: 'Jul 2022', to: 'Mar 2024'}]->(a)
            CREATE (m)-[:WORKED_AT {from: 'Aug 2020', to: 'Jun 2022'}]->(i)

            
            CREATE (m)-[:KNOWS {level: 'Expert'}]->(fl)
            CREATE (m)-[:KNOWS {level: 'Expert'}]->(d)
            CREATE (m)-[:KNOWS {level: 'Advanced'}]->(r)
            CREATE (m)-[:LEARNING]->(lc)
            CREATE (m)-[:LEARNING]->(rag)
            CREATE (m)-[:LEARNING]->(lg)
            CREATE (m)-[:KNOWS {level: 'Advanced'}]->(ca)
            CREATE (m)-[:KNOWS {level: 'Intermediate'}]->(fb)

            
            CREATE (m)-[:BUILT]->(p1)
            CREATE (m)-[:BUILT]->(p2)
            CREATE (m)-[:BUILT]->(p3)

            
            CREATE (p1)-[:USES]->(fl)
            CREATE (p1)-[:USES]->(lc)
            CREATE (p1)-[:USES]->(rag)
            CREATE (p2)-[:USES]->(fl)
            CREATE (p2)-[:USES]->(r)
            CREATE (p3)-[:USES]->(fl)
            CREATE (p3)-[:USES]->(fb)
        """)
        print("✅ Created all relationships!")

create_portfolio_graph(driver)
print("\n🎉 Portfolio Graph Created!")
print("Go to Neo4j Browser to see it visually!")
print(f"URL: {os.getenv('NEO4J_URI').replace('neo4j+s', 'https').split('.io')[0]}.io")

driver.close()