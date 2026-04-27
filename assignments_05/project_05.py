# Part 2: Mini-Project — Job Application Helper

from dotenv import load_dotenv
from openai import OpenAI

if load_dotenv():
    print("Successfully loaded api key")

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


