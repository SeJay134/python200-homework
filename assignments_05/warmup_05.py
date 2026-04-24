from dotenv import load_dotenv
from openai import OpenAI

print('API Question 1')

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

# print(response)
print('message content:', response.choices[0].message.content)
print('model:', response.model)
print('total tokens:', response.usage.total_tokens)