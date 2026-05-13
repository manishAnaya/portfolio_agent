import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

MANISH_DATA = open("manish_data.txt", "r", encoding="utf-8").read()

QUESTIONS = [
    "Who is Manish Kumar Agrahari?",
    "What is Manish's current job?",
    "How many years of experience does Manish have?",
    "What is Manish's tech stack?",
    "What projects has Manish built?",
    "Tell me about AI Chatbot Platform project",
    "What companies has Manish worked at?",
    "What is Manish's education background?",
    "Is Manish open to work?",
    "How can I contact Manish?",
    "What AI skills does Manish have?",
    "What state management does Manish know?",
    "What is Manish's LinkedIn?",
    "What certifications does Manish have?",
    "Where is Manish located?",
    "What is Manish's GitHub?",
    "Tell me about Portfolio Risk Management project",
    "What is Manish's portfolio website?",
    "What domains has Manish worked in?",
    "Is Manish good at Flutter?",

    # Personal Brand
    "Why should someone hire Manish?",
    "What makes Manish unique as a developer?",
    "What is Manish's biggest achievement?",
    "What is Manish's career goal?",
    "What type of work environment does Manish prefer?",
    "Can Manish work remotely?",
    "Is Manish a team player?",
    "What is Manish's notice period?",
    "What salary does Manish expect?",
    "What motivates Manish as a developer?",

    # Technical Deep Dive
    "What is Manish's experience with Clean Architecture?",
    "How does Manish handle state management in Flutter?",
    "What is Manish's experience with Firebase?",
    "Has Manish worked with REST APIs?",
    "What is Manish's experience with Riverpod?",
    "Does Manish know CI/CD?",
    "What is Manish's experience with Dart?",
    "Has Manish built fintech apps?",
    "What is Manish's experience with animations in Flutter?",
    "Does Manish know dark mode implementation?",

    # Projects Deep Dive
    "Tell me about Recruitment Career Platform project",
    "Tell me about HR Payroll Management project",
    "Tell me about EdTech Tutor Discovery project",
    "Tell me about Auto Parts Ecommerce project",
    "What Play Store apps has Manish published?",
    "How many apps has Manish shipped to production?",
    "What is the rating of Manish's recruitment app?",
    "Which project is Manish most proud of?",
    "What was the most challenging project Manish worked on?",
    "Has Manish built any AI powered mobile apps?",

    # Experience Deep Dive
    "What did Manish do at IBigDo Technologies?",
    "What did Manish do at Act T Connect?",
    "What is Manish doing at Finva Tech?",
    "How long did Manish work at Act T Connect?",
    "What kind of apps did Manish build at Act T Connect?",
    "Has Manish mentored junior developers?",
    "Has Manish contributed to architecture decisions?",
    "What compliance standards has Manish worked with?",
    "Has Manish worked with cross functional teams?",
    "What is Manish's leadership experience?",

    # AI & GenAI
    "Is Manish learning AI?",
    "What AI certifications does Manish have?",
    "What AI tools does Manish know?",
    "Is Manish working on GenAI projects?",
    "What is Manish's experience with LangChain?",
    "Does Manish know RAG?",
    "What AI courses has Manish completed?",
    "Is Manish combining Flutter with AI?",
    "What is Manish's vision for AI in mobile apps?",
    "Has Manish built any chatbot applications?",
]

dataset = []

print("🔥 Generating Fine Tuning Dataset...")
print(f"Total questions: {len(QUESTIONS)}\n")

for i, question in enumerate(QUESTIONS):
    print(f"Processing {i+1}/{len(QUESTIONS)}: {question}")

    messages = [
        SystemMessage(content=f"""
            You are a professional portfolio assistant
            for Manish Kumar Agrahari.
            Answer ONLY from the context below.
            Be professional, accurate and confident.
            Keep answers concise but complete.

            CONTEXT:
            {MANISH_DATA}
        """),
        HumanMessage(content=question)
    ]
 
    response = llm.invoke(messages)
    answer = response.content.strip()

    training_example = {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional AI assistant for Manish Kumar Agrahari's portfolio. Answer questions about his experience, skills, projects and background professionally and accurately."
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }

    dataset.append(training_example)
    print(f"✅ Done!\n")

with open("manish_finetune_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"\n🎉 Dataset created!")
print(f"Total examples: {len(dataset)}")
print(f"Saved to: manish_finetune_dataset.json")
print(f"\nSample entry:")
print(json.dumps(dataset[0], indent=2))