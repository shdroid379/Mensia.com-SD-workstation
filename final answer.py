import os
import asyncio
import ai_answer
import deep_multi_fetch
import pipeline
from pathlib import Path
from datetime import datetime
import json
from groq import Groq

def logging(question, answer):
    log_file = Path(__file__).resolve().parent/"chat_history.jsonl"
    log_entry = {
        "date & time:": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
    }
    with open(log_file, "a", encoding="utf-8") as o:
        o.write(json.dumps(log_entry) + "\n")

question = input("Hey, pls enter your question: ")
mode = input("Type (search/deep research) if you want web search/deep research respectively; if no, press enter: ")
search_loading = f"Searching the web and asking AI: {question}"
if mode.lower() == "search":
    print(search_loading)
    searched_content = pipeline.basic_search(question)
elif mode.lower() == "deep research":
    print(search_loading)
    searched_content = asyncio.run(deep_multi_fetch.combined_research(question))
else:
    print(f"Asking AI: {question}")
    searched_content = ""
    mode = "casual"

final_answer = ai_answer.ai_summary(question, searched_content, mode)
print(final_answer)
history = [[question, final_answer]]
logging(question, final_answer)

while True:
    follow_up = input("Follow up(if no, type 'exit'): ")
    if follow_up.lower() == 'exit':
        break
    try:
        client6 = Groq(api_key=os.getenv("GROQ_KEY"))
        question = follow_up
        response = client6.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a follow_up rephraser. You will read the history, interpret the context, and rephrase the prompt. Your response will include nothing but:- Follow up(dont forget these two words): (rephrased_prompt). Understand pronouns wisely. If the uer mentions any number, for example, 'What about the fourth one?', in that case you will interpret the meaning by looking at the context/history and find all numbered/non numbered pointers given and then rephrase the prompt. Your response should be such that i could directly paste it into a search engine. No fluff, no introduction, no conclusion."},
                {"role": "user", "content": f"Here is the history:- {history[-4:]}, and here's the follow up:- {question}. Rephrase the prompt according to the context."}
            ],
            temperature=0.1,
            max_tokens=75,
            include_reasoning=False,
            reasoning_effort="low"
        )
        raw = response.choices[0].message.content
        rephrased_prompt = raw.strip() if raw else follow_up
        mode = input("Enter 'search' if you want search mode(if not press enter)")
        if mode.lower() == "search":
            searched_content = pipeline.basic_search(rephrased_prompt)
        else:
            searched_content = ""
            mode = "casual"
        final_answer = ai_answer.ai_summary(rephrased_prompt, searched_content, mode)
        print(final_answer)
        history.append([question, final_answer])
        logging(question, final_answer)
    except Exception as e:
        print(f"Follow up is not available right now. {e} We are extremely sorry! 😓")
        print(f"Error Message: {repr(e)}")
        retry = input("Do you want to retry? (Retry/exit): ")
        if retry.lower() == "exit":
            break




