
from fastapi import FastAPI
from schema import AskRequest, AskResponse
from graph import graph

app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = graph.invoke({
        "query": request.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    })
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

@app.get("/")
def root():
    return {"status": "Zepto Support Assistant is running"}
