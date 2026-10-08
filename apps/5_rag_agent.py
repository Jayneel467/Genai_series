from pathlib import Path

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []




def process_document(path):
    ### data loader
    loader = PyPDFDirectoryLoader(path)
    docs=loader.load()


    ### data splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_data = splitter.split_documents(docs)

    ### Data embeddings create and Storing in vector Db
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db = InMemoryVectorStore.from_documents(
        documents=split_data,
        embedding=embeddings
    )


    ### Agent creation llm,tool,prompt
    llm = ChatGroq(model="openai/gpt-oss-20b")

    ### tool creation
    @tool
    def context_generator(query: str) -> str:
        """
        Retrieve documents relevant to the query from the knowledge base.
        """
        relevant_docs = vector_db.similarity_search(query=query)
        return "\n\n".join(doc.page_content for doc in relevant_docs)

    ### System Prompt
    system_prompt = """
    You are a helpful assistant that answers questions using retrieved context.
    Always use the 'context_generator' tool for questions about the medical report.
    If the retrieved context does not contain the answer, say that you cannot answer
    from the report. Do not invent medical information.
    """

    ### Memory
    memory = InMemorySaver()

    ### Agent Creation
    agent = create_agent(
        model=llm,
        tools=[context_generator],
        system_prompt=system_prompt,
        checkpointer=memory,
    )

    st.session_state.agent=agent
    st.session_state.document_uploaded=True

### Upload UI
if not st.session_state.document_uploaded:
    uploaded=st.file_uploader(label="Select PDF Files",type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner(text="Processing..."):
            path = project_root / "doc_files"
            path.mkdir(parents=True, exist_ok=True)
            for file in uploaded:
                with open(path / file.name, "wb") as f:
                    f.write(file.getvalue())
            process_document(str(path))
            st.rerun()

### Chat UI

if st.session_state.document_uploaded and st.session_state.agent:
    query=st.chat_input("Ask Anything related to uploaded documents...")

    for message in st.session_state.messages:
        role=message.get("role")
        content=message.get("content")
        st.chat_message(role).markdown(content)


    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        response=st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer=response["messages"][-1].content
        st.chat_message("AI").markdown(answer)
        st.session_state.messages.append({"role":"AI","content":answer})
