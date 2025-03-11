import streamlit as st
import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

# Set Page Title
st.set_page_config(page_title="🔍 LangChain - Smart Chatbot")

# Dark Mode Toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
if dark_mode:
    st.markdown(
        """
        <style>
            body {
                background-color: #121212;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# API Key Input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key", type='password')

# Search Options
search_option = st.sidebar.selectbox("Choose a search source", ["DuckDuckGo", "Arxiv", "Wikipedia"])

# Initialize Search Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

search = DuckDuckGoSearchRun(name="Search", include_links=True)

# Assign Tools Based on User Selection
if search_option == "DuckDuckGo":
    tools = [search]
elif search_option == "Arxiv":
    tools = [arxiv]
elif search_option == "Wikipedia":
    tools = [wiki]

# Initialize Memory for Chat Context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Chat History Setup
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a chatbot that can search the web. How can I help you?"}
    ]

# Display Chat History
st.sidebar.subheader("Chat History")
for msg in st.session_state.messages:
    role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Bot"
    st.sidebar.write(f"{role}: {msg['content']}")
    st.chat_message(msg["role"]).write(msg['content'])

# Voice Input Function
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except:
            st.write("Sorry, could not recognize your voice.")
            return ""

# Voice Input Button
st.sidebar.write("🎤 Voice Input")
if st.sidebar.button("Start Listening"):
    user_voice_text = voice_input()
    if user_voice_text:
        st.session_state.messages.append({"role": "user", "content": user_voice_text})
        st.chat_message("user").write(user_voice_text)

# Handle Text Input
if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Initialize LLM
    llm = ChatGroq(api_key=api_key, model='Llama3-8b-8192', streaming=True)
    
    # Create Search Agent
    search_agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True, memory=memory
    )
    
    # Generate Response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        
        # Append response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
        
        # Display Source Link if Available
        if hasattr(response, 'source'):
            st.write(f"**Source:** [{response.source}]({response.source})")
