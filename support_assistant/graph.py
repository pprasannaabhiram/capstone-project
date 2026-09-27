
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import chromadb
from sentence_transformers import SentenceTransformer

MOCK_LLM = os.environ.get("MOCK_LLM", "1")

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_store")
_model = SentenceTransformer("all-MiniLM-L6-v2")
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_collection("zepto_policies")

POLICY_KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: List[str]
    confidence: float

def classify_intent(state: GraphState) -> GraphState:
    q_lower = state["query"].lower()
    state["intent"] = "policy_question" if any(kw in q_lower for kw in POLICY_KEYWORDS) else "general_question"
    return state

def retrieve_and_answer(state: GraphState) -> GraphState:
    query_embedding = _model.encode([state["query"]]).tolist()
    results = _collection.query(query_embeddings=query_embedding, n_results=3)
    retrieved_ids = results["ids"][0]
    retrieved_docs = results["documents"][0]
    top_chunk = retrieved_docs[0] if retrieved_docs else ""
    snippet = top_chunk[:200]

    state["answer"] = f"Based on the retrieved context: {snippet}"
    state["sources"] = retrieved_ids
    state["confidence"] = 1.0 if MOCK_LLM != "0" else 0.9
    return state

def direct_answer(state: GraphState) -> GraphState:
    state["answer"] = "I can only answer questions about Zepto policies right now."
    state["sources"] = []
    state["confidence"] = 1.0 if MOCK_LLM != "0" else 0.8
    return state

def route_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges("classify_intent", route_intent, {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    })
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)
    return workflow.compile()

graph = build_graph()

if __name__ == "__main__":
    for q in ["What is your delivery fee?", "Tell me a joke"]:
        result = graph.invoke({"query": q, "intent": "", "answer": "", "sources": [], "confidence": 0.0})
        print(q, "->", result)
