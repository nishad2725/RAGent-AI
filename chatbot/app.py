# Enhanced chatbot/app.py with stylish RAG UI + LLM Toggle + Dark Mode + Icons + Colored Agent Output

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import HuggingFaceHub, Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain import hub

import streamlit as st
from streamlit_lottie import st_lottie
from dotenv import load_dotenv
import os
import requests
import tempfile
import speech_recognition as sr
from gtts import gTTS
import csv

# --- Load env vars
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# --- Branding and Layout
st.set_page_config(page_title="RAGent AI - Multi-Agent RAG Assistant", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .agent-output {
        background-color: #1a1a1a;
        color: #f4f4f4;
        padding: 1rem;
        border-radius: 8px;
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center;'>RAGent AI 🧠</h1>
    <h4 style='text-align: center; color: #BBBBBB;'>RAG + Multi-Agent Assistant Powered by LangChain</h4>
    <p style='text-align: center; font-size: 14px;'>Chat with your files, links, or research using OpenAI, HuggingFace, or Ollama — backed by memory, retrieval, and intelligent tool use.</p>
""", unsafe_allow_html=True)

@st.cache_data
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

with st.sidebar:
    lottie_ai = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_zrqthn6o.json")
    st_lottie(lottie_ai, height=140)
    st.markdown("---")
    st.markdown("### 🤖 Select LLM")
    llm_choice = st.radio("", ["🧠 OpenAI", "🌐 HuggingFace", "💻 Ollama"], help="Choose the model to power your RAG assistant")
    st.markdown("### 🔍 Search Mode")
    mode = st.radio("", ["Fast 🔎", "Smart 🧠", "Agent 🛠️"], help="Fast: Just vector search, Smart: RAG + memory, Agent: uses tools like Wikipedia/Arxiv")
    st.markdown("---")
    with st.expander("ℹ️ How does AskMe AI work?"):
        st.markdown("""
        1. Upload a PDF or enter a website URL
        2. We extract + embed the content with OpenAI Embeddings
        3. Select retrieval mode and model (OpenAI, HF, Ollama)
        4. Ask your question and get contextual responses 🔁
        """)

# --- Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# --- Choose model
llm_choice_clean = llm_choice.split(" ")[1]  # Extract model name without emoji
if llm_choice_clean == "OpenAI":
    llm = ChatOpenAI(model="gpt-3.5-turbo")
elif llm_choice_clean == "HuggingFace":
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", task="text2text-generation", model_kwargs={"temperature": 0.5, "max_new_tokens": 512})
elif llm_choice_clean == "Ollama":
    llm = Ollama(model="gemma3:1b")

# --- Voice Input/Output functions
def listen_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."

def speak_response(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    st.audio("response.mp3", format="audio/mp3")

# --- Feedback logging
def save_feedback(question, response, rating):
    with open("feedback_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([question, response, rating])

# --- TABS for inputs
tab1, tab2, tab3 = st.tabs(["💻 Website URL", "📄 Upload PDF", "✍️ Manual Text"])
user_query = None

# --- TAB 1: URL
with tab1:
    url = st.text_input("Enter a website URL")
    if url:
        st.success("Loading and indexing content from URL...")
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        st.session_state.vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        st.success("Ready! Ask questions in 'Manual Text' tab.")

# --- TAB 2: PDF Upload
with tab2:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        st.success("Processing PDF...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        st.session_state.vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        st.success("PDF indexed. Ask questions in 'Manual Text' tab.")

# --- TAB 3: Manual Questioning
with tab3:
    user_query = st.chat_input("Ask your question...")
    if st.button("🎤 Speak Your Question"):
        user_query = listen_from_mic()
        st.markdown(f"**You said:** {user_query}")

# --- Chat Display based on Mode
if user_query and st.session_state.vectorstore:
    with st.chat_message("user"):
        st.markdown(user_query)

    retriever = st.session_state.vectorstore.as_retriever()
    memory = st.session_state.memory

    if "Fast" in mode:
        docs = retriever.get_relevant_documents(user_query)
        response = docs[0].page_content if docs else "No relevant info found."
        with st.chat_message("assistant"):
            st.markdown(response)
            speak_response(response)

    elif "Smart" in mode:
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
        )
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = retrieval_chain.invoke({"question": user_query})
                st.markdown(result["answer"])
                speak_response(result["answer"])
                rating = st.radio("Rate this response:", ["👍", "👎"], horizontal=True, key=user_query)
                if rating:
                    save_feedback(user_query, result["answer"], rating)

    elif "Agent" in mode:
        wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
        arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1))
        retriever_tool = create_retriever_tool(retriever, "rag_search", "Search documents you've uploaded")
        tools = [wiki_tool, arxiv_tool, retriever_tool]

        prompt = hub.pull("hwchase17/openai-tools-agent")

        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        with st.chat_message("assistant"):
            with st.spinner("Running tools and generating answer..."):
                result = agent_executor.invoke({"input": user_query})
                st.markdown(f"<div class='agent-output'>{result['output']}</div>", unsafe_allow_html=True)
                speak_response(result["output"])

elif user_query:
    st.warning("Please upload a PDF or enter a URL first to provide context for the chatbot.")
