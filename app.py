import os
import streamlit as st
import logging
from agent_router import handle_routed_query
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "Finance project")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit-ui")

# Streamlit setup
st.set_page_config(page_title="Business Insights Assistant", page_icon="📊", layout="centered")
st.title("📊 Business Insights Assistant")
st.markdown("""
Welcome! I'm your smart assistant for understanding everything about your property business.  
You can ask about:

- **Property details** (amenities, location features, nearby services)  
- **Sales and agents performance** (most sold cities, property pricing, date purchased)  
- **Customer and agent feedback** (what do clients say about properties or agents?)

---
""")

# Init session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Reset conversation
if st.button("New Chat"):
    st.session_state.chat_history = []
    st.success("Started a new conversation.")
    logger.info("[Session] Chat history reset by user.")

# Display existing chat history
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# Input
user_question = st.chat_input("Ask me anything about your business...")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Analyzing your data..."):
        try:
            # Convert history list of tuples to chat_memory_dict
            chat_memory_dict = {}
            for i, (q, a) in enumerate(st.session_state.chat_history, start=1):
                chat_memory_dict[f"q{i}"] = q
                chat_memory_dict[f"a{i}"] = a

            # Log question
            logger.info(f"[User Question] {user_question}")

            # Run routed query
            answer = handle_routed_query(user_question, chat_memory_dict=chat_memory_dict)

            # Save response
            st.session_state.chat_history.append((user_question, answer))

            with st.chat_message("assistant"):
                st.markdown(answer)

            logger.info(f"[Answer] {answer}")

        except Exception as e:
            logger.exception("[Error] Failed to process user query.")
            with st.chat_message("assistant"):
                st.error("Sorry, I couldn’t find the info.")

