from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_community.tools.google_serper.tool import GoogleSerperRun
from langgraph.checkpoint.memory import MemorySaver



search = GoogleSerperRun()
llm = ChatGroq(model="openai/gpt-oss-20b")
agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt="You are an agent that can search the web using Google and answer questions accurately.",
    checkpointer=MemorySaver()
)

while True:
    query = input("user: ")
    if query.lower() == "quit":
        print("GoodBye")
        break
    response = agent.invoke({"messages": [{"role": "user", "content": query}]},
                            {"configurable":{"thread_id":"1"}}
                            )
    print("AI:", response["messages"][-1].content)
