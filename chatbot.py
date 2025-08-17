from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets
from groq import Groq
from config import Config

class MedicalChatbot:
    def __init__(self, client, vector_store=None):
        self.client = client
        self.vector_store = vector_store
        self.system_prompt = """You are Dr. Bot, a medical AI assistant. Follow these rules:
1. Provide accurate, evidence-based information
2. Explain medical concepts in simple terms
3. Never diagnose or prescribe treatment
4. Always recommend consulting a doctor
5. Be empathetic and professional"""
        
    def setup_jupyter_ui(self):
        """Initialize the user interface components for Jupyter"""
        self.chat_history = widgets.Output()
        display(self.chat_history)
        
        self.user_input = widgets.Text(
            placeholder='Type your medical question here...',
            layout=widgets.Layout(width='80%')
        )
        
        self.submit_button = widgets.Button(
            description="Ask",
            button_style='primary',
            tooltip='Submit your question'
        )
        self.submit_button.on_click(self.process_input)
        
        self.clear_button = widgets.Button(
            description="Clear",
            button_style='warning'
        )
        self.clear_button.on_click(self.clear_chat)
        
        display(widgets.HBox([self.user_input, self.submit_button, self.clear_button]))
    
    def get_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context from documents"""
        if not self.vector_store:
            return ""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return "\n\n".join([f"Source {i+1}:\n{doc.page_content}" 
                               for i, doc in enumerate(docs)])
        except Exception as e:
            print(f"Context retrieval error: {e}")
            return ""
    
    def generate_response(self, query: str) -> str:
        """Generate response using Groq API"""
        context = self.get_context(query)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I'm having trouble responding. Error: {str(e)}"
    
    def process_input(self, _):
        """Handle user input for Jupyter"""
        query = self.user_input.value.strip()
        if not query:
            return
            
        with self.chat_history:
            display(Markdown(f"**You:** {query}"))
            
            response = self.generate_response(query)
            
            clear_output(wait=True)
            display(Markdown(f"**You:** {query}"))
            display(Markdown(f"**Dr. Bot:** {response}"))
        
        self.user_input.value = ""
    
    def clear_chat(self, _):
        """Clear the chat history"""
        self.chat_history.clear_output()