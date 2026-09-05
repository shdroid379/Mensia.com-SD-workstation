from openai import OpenAI
import os

# Get your API key from https://www.cometapi.com/console/token, and paste it here
API_KEY = os.environ.get("COMETAPI_KEY") or ""
BASE_URL = "https://api.cometapi.com/v1"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0)

completion = client.chat.completions.create(
    model="glm-5.3-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with one short sentence confirming that you are ready."},
    ],
)

print(completion.choices[0].message.content)
