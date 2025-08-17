import streamlit as st
from groq import Groq
from document_processor import load_and_process_documents
from chatbot import MedicalChatbot
from config import Config


def main():
    """Main function to run the Medical Chatbot Streamlit app"""
    st.set_page_config(page_title="Medical Chatbot", page_icon="🏥", layout="wide")

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.client = None
        st.session_state.vector_store = None
        st.session_state.messages = []

    # Initialize Groq client
    if not st.session_state.client:
        try:
            st.session_state.client = Groq(api_key=Config.GROQ_API_KEY)
            st.success("✅ Groq client initialized successfully")
        except Exception as e:
            st.error(f"❌ Error initializing Groq client: {e}")
            return

    # Load or create vector store
    if not st.session_state.vector_store:
        with st.spinner("Loading embeddings / documents..."):
            st.session_state.vector_store = load_and_process_documents()

        if st.session_state.vector_store is None:
            st.warning(
                "⚠️ No vector store (`vector_db/`) or PDF documents found. \n\n"
                "The chatbot will still work, but only with the LLM (no PDF knowledge)."
            )

    # Set up the UI
    st.title("🏥 Medical Chatbot")
    st.markdown(
        "Ask medical questions based on the provided PDF documents (if available).  \n"
        "⚠️ This is an AI assistant, **not a doctor**. Always consult a healthcare professional for medical advice."
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Type your medical question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chatbot = MedicalChatbot(
                    st.session_state.client,
                    st.session_state.vector_store  # can be None → chatbot should fallback
                )
                try:
                    response = chatbot.generate_response(prompt)
                except Exception as e:
                    response = f"❌ Error generating response: {e}"
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # Add a clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
