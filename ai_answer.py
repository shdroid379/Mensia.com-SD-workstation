import os
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
import pipeline
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

# Import the OpenAI client to use with Hyperbolic
from openai import OpenAI

def hyperbolic_audit(prompt, context, outcome, mode):
    try:
        # Initialize OpenAI client but point it to Hyperbolic's servers
        client = OpenAI(
            api_key=os.getenv("HYPERBOLIC_API_KEY"),
            base_url="https://api.hyperbolic.xyz/v1"
        )
        
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3", 
            messages=[
                {"role": "system", "content": mistral_instructions},
                {"role": "user", "content": mistral_prompt.format(prompt=prompt, context=context, mode=mode, outcome=outcome)}
            ],
            temperature=0.0
        )
        
        raw_answer = response.choices[0].message.content  
        return raw_answer.strip() if raw_answer else outcome
        
    except Exception as e:
        print(f"Hyperbolic Auditor Error: {e}")
        return outcome # Bypasses auditor and returns Gemini's draft directly if it fails

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
            outcome = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=search_input.format(prompt=prompt, context=context),
                    config=types.GenerateContentConfig(
                        system_instruction=search_instructions,
                        temperature=0.1,
                        max_output_tokens=1500
                    )
                ).text
            
            # Send to Hyperbolic for auditing
            audited_answer = hyperbolic_audit(prompt, context, outcome, mode) or outcome
            
            if audited_answer.upper() == "CORRECT":
                print("Gemini answered, and Hyperbolic (Qwen 72B) approved it.")
                return outcome
            else:
                print("Hyperbolic revised Gemini's answer.")
                return audited_answer

        elif mode.lower() == "deep research":
            outcome = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=deep_input.format(prompt=prompt, context=context),
                    config=types.GenerateContentConfig(
                        system_instruction=deep_research_instructions,
                        temperature=0.0,
                        max_output_tokens=3250
                    )
                ).text
            
            # Send to Hyperbolic for auditing
            audited_answer = hyperbolic_audit(prompt, context, outcome, mode) or outcome
            
            if audited_answer.upper() == "CORRECT":
                print("Gemini answered, and Hyperbolic (Qwen 72B) approved it.")
                return outcome
            else:
                print("Hyperbolic revised Gemini's answer.")
                return audited_answer
        else:
            return "Pls select a valid mode."

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "System Error: The core synthesis engine is currently unavailable. Please try again."