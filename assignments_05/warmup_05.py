from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def control_mode(messages, t):
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=t
    )
        return response

print('API Question 1\n')
messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
response = control_mode(messages, 1.0)
print('message content:', response.choices[0].message.content)
print('model:', response.model)
print('total tokens:', response.usage.total_tokens)


print('API Question 2\n')
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

messages=[{"role": "user", "content": prompt}]
response = control_mode(messages, temperatures[0])
print('temp 0:\n', response.choices[0].message.content)
print()
messages=[{"role": "user", "content": prompt}]
response = control_mode(messages, temperatures[1])
print('temp 0.7:\n', response.choices[0].message.content)
print()
messages=[{"role": "user", "content": prompt}]
response = control_mode(messages, temperatures[2])
print('temp 1.5:\n', response.choices[0].message.content)
print()

# At temperature 0, the output is deterministic and consistent, usually giving the same answer every time.
# At temperature 0.7, the output is more creative but still relatively stable.
# At temperature 1.5, the output becomes highly creative and varied, but less predictable and sometimes inconsistent.
#
# If I needed a consistent, reproducible output, I would use temperature = 0.