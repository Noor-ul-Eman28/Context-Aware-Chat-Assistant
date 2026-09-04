import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: #000000;
        z-index: 999;
        padding: 1rem 2rem;
        border-bottom: 1px solid #333333;
    }
    .fixed-header h1 {
        margin: 0;
        color: #ffffff;
    }
    .block-container {
        padding-top: 6rem;
    }
    </style>
""", unsafe_allow_html=True)

st.divider()
st.markdown('<h1 style="color:#ffffff;">🤖 My ChatBot</h1>', unsafe_allow_html=True)
st.divider()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)

chat_prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        SystemMessage(content="You are a chatbot having conversation with a human."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

chain = LLMChain(
    llm=llm,
    prompt=chat_prompt,
    memory=memory,
    verbose=True
)

# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 My ChatBot")
    st.markdown("### 💬 Chat History")

    if st.button("➕ New Chat", use_container_width=True):
        # Save the current conversation before starting fresh (only if it has content)
        if st.session_state.messages:
            chat_id = st.session_state.current_chat_id or f"chat_{len(st.session_state.all_chats)}"
            st.session_state.all_chats[chat_id] = st.session_state.messages
        # Reset for a new conversation
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.rerun()

    st.divider()

    if st.session_state.all_chats:
        for chat_id, chat_messages in st.session_state.all_chats.items():
            # Use the first user message as a short title
            title = next((m["content"] for m in chat_messages if m["role"] == "user"), "New Chat")
            title = title[:30] + ("..." if len(title) > 30 else "")

            if st.button(title, key=f"load_{chat_id}", use_container_width=True):
                st.session_state.messages = chat_messages
                st.session_state.current_chat_id = chat_id
                st.rerun()
    else:
        st.caption("Your past conversations will appear here")

# --- Main chat display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Your Prompt:"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = chain.invoke({"content": user_input})

    with st.chat_message("assistant"):
        st.markdown(response["text"])
    st.session_state.messages.append({"role": "assistant", "content": response["text"]})

    # Auto-save current chat as it grows
    if st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = f"chat_{len(st.session_state.all_chats)}"
    st.session_state.all_chats[st.session_state.current_chat_id] = st.session_state.messages