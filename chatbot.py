from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets
from groq import Groq
from config import Config

class MedicalChatbot:
    def __init__(self, client, vector_store=None):
        """
        MedicalChatbot using Groq LLM and optional vector store for retrieval.

        Args:
            client: Groq client instance
            vector_store: FAISS (or other) vector store, or None if unavailable
        """
        self.client = client
        self.vector_store = vector_store

    def generate_response(self, query: str) -> str:
        """
        Generate response using RAG if vector_store available,
        otherwise fallback to LLM-only.
        """
        try:
            context = ""
            if self.vector_store:
                # Retrieve top 3 docs from embeddings
                docs = self.vector_store.similarity_search(query, k=3)
                context = "\n\n".join([d.page_content for d in docs])

            # Build the prompt
            if context:
                prompt = (
                    "You are a helpful medical assistant. Use the following context "
                    "from medical PDFs to answer the question.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    "Answer as clearly and safely as possible."
                )
            else:
                prompt = (
                    "You are a helpful medical assistant. No reference documents are available, "
                    "so answer based only on your general medical knowledge.\n\n"
                    f"Question: {query}\n\n"
                    "Answer as clearly and safely as possible."
                )

            # Call Groq
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",  # or whichever Groq model you're using
                messages=[{"role": "user", "content": prompt}],
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"❌ Error in generating response: {e}"
