import os
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)

from ai_instructions import (
    casual_instructions,
    search_instructions,
    deep_research_instructions,
    casual_input,
    search_input,
    deep_input,
    mistral_instructions,
    mistral_prompt
)
from google import genai
from google.genai import types

def gemini_audit(prompt, context, outcome, mode):
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Search mode is no longer audited. This is strictly for Deep Research now.
        audit_model = "gemini-3.1-pro"
        
        response = client.models.generate_content(
            model=audit_model,
            contents=mistral_prompt.format(
                prompt=prompt,
                context=context,
                mode=mode,
                outcome=outcome
            ),
            config=types.GenerateContentConfig(
                system_instruction=mistral_instructions,
                temperature=0.0,
                max_output_tokens=3250
            )
        )

        raw_answer = response.text
        return raw_answer.strip() if raw_answer else outcome

    except Exception as e:
        print(f"Gemini Auditor Error ({audit_model}): {e}")
        return outcome

def ai_summary(prompt, context="", mode="casual", history=None):
    if history is None:
        history = []

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        if mode.lower() == "casual":
            history_text = ""
            if history:
                history_text = "PREVIOUS CHAT HISTORY:\n"
                for user_q, ai_a in history[-4:]:
                    history_text += f"User: {user_q}\nMensia: {ai_a}\n\n"

            final_prompt = f"{history_text}{casual_input.format(prompt=prompt)}"

            outcome = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=casual_instructions,
                    temperature=1.0,
                    max_output_tokens=1000
                )
            ).text
            return outcome

        elif mode.lower() == "search":
            # NO AUDITING FOR STANDARD SEARCH - Latency optimized single-pass
            outcome = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=search_input.format(prompt=prompt, context=context),
                config=types.GenerateContentConfig(
                    system_instruction=search_instructions,
                    temperature=0.1,
                    max_output_tokens=1500
                )
            ).text
            
            return outcome

        elif mode.lower() == "deep research":
            # DEEP RESEARCH - Actively audited by Gemini 3.1 Pro
            outcome = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=deep_input.format(prompt=prompt, context=context),
                config=types.GenerateContentConfig(
                    system_instruction=deep_research_instructions,
                    temperature=0.0,
                    max_output_tokens=3250
                )
            ).text

            audited_answer = gemini_audit(prompt, context, outcome, mode) or outcome

            if audited_answer.upper() == "CORRECT":
                print("Gemini answered. Gemini 3.1 Pro Auditor approved.")
                return outcome
            else:
                print("Gemini 3.1 Pro Auditor revised Gemini's answer.")
                return audited_answer

        else:
            return "Please select a valid mode (casual, search, or deep research)."

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "System Error: The core synthesis engine is currently unavailable. Please try again."