from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
# Get your API key from https://www.cometapi.com/console/token, and paste it here
API_KEY = os.getenv("MORPH_API_KEY") 
BASE_URL = "https://api.morphllm.com/v1"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0)

completion = client.chat.completions.create(
    model="morph-glm53flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with one short sentence confirming that you are ready."},
    ],
)

print(completion.choices[0].message.content)
