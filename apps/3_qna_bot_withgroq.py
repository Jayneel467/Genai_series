from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.tools.google_serper.tool import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm=ChatGroq(model="openai/gpt-oss-20b")
search=GoogleSerperAPIWrapper()
if "memory" not in st.session_state:
    st.session_state.memory=MemorySaver()
    st.session_state.history=[]
agent=create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="",
    checkpointer=st.session_state.memory
)


st.subheader("QuickAnswer-Answer at speed of thought")

for message in st.session_state.history:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)


query=st.chat_input("Ask me anything....")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
    response=agent.stream_channels(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":"1"}},
        steam_mode="messages"
    )
    ai_container=st.chat_message("ai")
    with ai_container:
        space=st.empty()
        message=""
        for chunk in response:
            message=message+chunk[0].content
            space.write(message)
            answer=response["messages"][-1].content
        st.chat_message("assistant").markdown(answer)
        st.session_state.history.append({"role":"assistant","content":message})