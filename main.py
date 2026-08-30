import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq

import pipeline
import ai_answer
import deep_multi_fetch

app = FastAPI()

histories: dict[str, list[list[str]]] = {}

class Question(BaseModel):
    question: str
    mode: str = "casual"
    session_id: str

def rephrase_if_followup(question: str, session_id: str) -> str:
    history = histories.get(session_id, [])
    if not history:
        return question  

    try:
        client6 = Groq(api_key=os.getenv("GROQ_KEY"))
        response = client6.chat.completions.create(
            model="openai/gpt-oss-20b", # Keeping your exact model
            messages=[
                {"role": "system", "content": "You are a follow_up rephraser. You will read the history, interpret the context, and rephrase the prompt. Your response will include nothing but the rephrased prompt. Understand pronouns wisely. If the user mentions any number, interpret by looking at the context/history. Your response should be such that it could be directly pasted into a search engine. No fluff, no introduction, no conclusion."},
                {"role": "user", "content": f"Here is the history:- {history[-4:]}, and here's the follow up:- {question}. Rephrase the prompt according to the context."}
            ],
            temperature=0.1,
            max_tokens=75,
            include_reasoning=False,
            reasoning_effort="low"
        )
        raw = response.choices[0].message.content
        return raw.strip() if raw else question
    except Exception as e:
        print(f"Groq Rephraser failed: {e}. Falling back to original query.")
        return question

@app.post("/ask")
def ask(q: Question):
    mode = q.mode.lower()
    rephrased = rephrase_if_followup(q.question, q.session_id)
    
    current_history = histories.get(q.session_id, [])

    if mode == "search":
        context = pipeline.basic_search(rephrased)
    else:
        context = ""
        mode = "casual"

    answer = ai_answer.ai_summary(rephrased, context, mode, current_history) or ""
    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer}

@app.post("/deep-research")
async def deep_research(q: Question):
    rephrased = rephrase_if_followup(q.question, q.session_id)
    current_history = histories.get(q.session_id, [])
    
    context = await deep_multi_fetch.combined_research(rephrased)
    
    answer = ai_answer.ai_summary(rephrased, context, "deep research", current_history) or ""
    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/health")
def health_check():
    return {"status": "alive"}