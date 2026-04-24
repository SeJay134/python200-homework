from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def control_mode(messages, t, n, max_tokens=None):
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=t,
        n=n,
        max_tokens=max_tokens
    )
        return response

print('API Question 1\n')
messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
response = control_mode(messages, 1.0, n=1)
print('message content:', response.choices[0].message.content)
print('model:', response.model)
print('total tokens:', response.usage.total_tokens)


print('API Question 2\n')
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

messages=[{"role": "user", "content": prompt}]
response = control_mode(messages, temperatures[0], n=1)
print('temp 0:\n', response.choices[0].message.content)
print()
messages=[{"role": "user", "content": prompt}]
response = control_mode(messages, temperatures[1], n=1)
print('temp 0.7:\n', response.choices[0].message.content)
print()
messages=[{"role": "user", "content": prompt}]
response = control_mode(messages, temperatures[2], n=1)
print('temp 1.5:\n', response.choices[0].message.content)
print()

# At temperature 0, the output is deterministic and consistent, usually giving the same answer every time.
# At temperature 0.7, the output is more creative but still relatively stable.
# At temperature 1.5, the output becomes highly creative and varied, but less predictable and sometimes inconsistent.
#
# If I needed a consistent, reproducible output, I would use temperature = 0.

print('API Question 3')
prompt = "Give me a one-sentence fun fact about pandas (the animal, not the library)."
messages=[{"role": "user", "content": prompt}]
response = control_mode(messages, 1.0, 3)
for i, choice in enumerate(response.choices):
    print(f'{i+1}: {choice.message.content}')

print('API Question 4')
prompt = 'Explain how neural networks work.'
messages=[{"role": "user", "content": prompt}]
response = control_mode(messages, 1.5, 1, 15)
print('max token 15', response.choices[0].message.content)

# The response was cut off before the full explanation because max_tokens=15 limits the number of tokens
# the model is allowed to generate, so the answer cannot be completed.
#
# In real applications, max_tokens is useful to control response length, reduce cost, ensure faster responses,
# and prevent overly long or unstructured outputs (e.g., in chatbots or APIs with strict formatting limits).

# System Messages and Personas
print('System Question 1')
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = control_mode(messages, 0.7, 1)
print('content 1:', response.choices[0].message.content)
print()
messages = [
    {"role": "system", "content": "You are a dancer, encouraging Dances tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = control_mode(messages, 0.7, 1)
print('content 2:', response.choices[0].message.content)

# The system message changed the personality of the model.
# In the first case, it acted like a patient Python tutor with structured, detailed explanations.
# In the second case, it became more playful and energetic (dance-themed), but the technical explanation
# stayed similar.
# This shows that the system prompt mainly affects tone and style, not the correctness of the content.


print('System Question 2')
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = control_mode(messages, 0.7, 1)
print('content:', response.choices[0].message.content)

# The model knows Jordan's name because the full conversation history is included in the messages list.
# Even though the API is stateless (it does not remember previous requests),
# we manually provide all previous context in the current request, so the model can "see" the name.

