from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# --- load environment variables (.env) ---
load_dotenv()

# ====== STATE ======
class AgentState(BaseModel):
    question: str
    context: str = ""
    answer: str = ""

# ====== RETRIEVER (FAISS) ======
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectordb = FAISS.load_local(
    "vectordb/faiss_index",
    embedder,
    allow_dangerous_deserialization=True
)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# ====== LLM (OpenAI) ======
llm = ChatOpenAI(model="gpt-4o-mini")

# ====== QUERY REFORMULATION ======
rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Rewrite the user's question into a very clear search query for retrieving government schemes.
Focus on:
- Scheme type (student, woman, farmer, OBC, SC/ST, disability etc.)
- Education/Income class
- State name if mentioned
- Benefits they want (scholarship, pension, loan)

User question:
{question}

Search Query:
"""
)

def rewrite_query_node(state: AgentState) -> AgentState:
    rewritten_msg = llm.invoke(rewrite_prompt.format(question=state.question))
    state.question = rewritten_msg.content.strip()
    return state


# ====== MAIN ANSWER PROMPT ======
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a professional Government Scheme Assistant for India 🇮🇳.

Use ONLY the context provided below.
If missing information, say:
"Not available in my knowledge base."

Context:
{context}

User Question:
{question}

Answer with:
1️⃣ Scheme Name(s)
2️⃣ Eligibility
3️⃣ Benefits
4️⃣ Documents Required (if available)
5️⃣ How to Apply (if available)
6️⃣ Whether Central or State scheme
7️⃣ Notes / Tags (if available)
"""
)

# ====== RETRIEVAL NODE ======
def retrieve_node(state: AgentState) -> AgentState:
    docs = retriever.invoke(state.question)
    state.context = "\n\n".join(doc.page_content for doc in docs) if docs else ""
    return state


# ====== ANSWER NODE ======
def answer_node(state: AgentState) -> AgentState:
    if not state.context:
        state.answer = "I do not have this information in my knowledge base."
        return state

    prompt_text = prompt.format(
        context=state.context,
        question=state.question
    )

    response_msg = llm.invoke(prompt_text)
    state.answer = response_msg.content
    return state


# ====== BUILD LANGGRAPH WORKFLOW ======
workflow = StateGraph(AgentState)

workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("rewrite_query")
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", END)

graph_app = workflow.compile()


# ====== PUBLIC AGENT FUNCTION ======
def chatbot_query(question: str) -> str:
    result = graph_app.invoke({"question": question})
    return result["answer"]
