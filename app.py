import gradio as gr
from langgraph_agent import chatbot_query


def chat_fn(message, history):
    # history is ignored for now; we do stateless QA per query
    return chatbot_query(message)


demo = gr.ChatInterface(
    fn=chat_fn,
    title="🇮🇳 Government Scheme Assistant (LangGraph + RAG)",
    description="Ask about Indian government schemes: scholarships, business subsidies, farmer schemes, etc.",
)


if __name__ == "__main__":
    print("Starting Government Scheme Chatbot...")
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True, show_error=True)
