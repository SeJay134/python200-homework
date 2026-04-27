from dotenv import load_dotenv
from openai import OpenAI
import json
import re

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

print('API Question 3\n')
prompt = "Give me a one-sentence fun fact about pandas (the animal, not the library)."
messages=[{"role": "user", "content": prompt}]
response = control_mode(messages, 1.0, 3)
for i, choice in enumerate(response.choices):
    print(f'{i+1}: {choice.message.content}')
print()

print('API Question 4\n')
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
print('System Question 1\n')
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
print()
# The system message changed the personality of the model.
# In the first case, it acted like a patient Python tutor with structured, detailed explanations.
# In the second case, it became more playful and energetic (dance-themed), but the technical explanation
# stayed similar.
# This shows that the system prompt mainly affects tone and style, not the correctness of the content.


print('System Question 2\n')
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = control_mode(messages, 0.7, 1)
print('content:', response.choices[0].message.content)
print()
# The model knows Jordan's name because the full conversation history is included in the messages list.
# Even though the API is stateless (it does not remember previous requests),
# we manually provide all previous context in the current request, so the model can "see" the name.


# Prompt Engineering
print('Prompt Question 1 — Zero-Shot\n')
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

messages=[{'role': "user", "content": """
           Classify the sentiment of each review as positive, negative, or mixed. 
           Review 1 The onboarding process was smooth and the team was welcoming. 
           Review 2 The software crashes constantly and support never responds. 
           Review 3 Great price, but the documentation is nearly impossible to follow. 
           Respond in this format: Review 1: ..., Review 2: ..., Review 3: ..."""
           }]

response = control_mode(messages, 0.7, 1)
print('content:', response.choices[0].message.content)
print()

print('Prompt Question 2 — One-Shot\n')
messages=[{'role': "user", "content": """
           Classify the sentiment of each review as positive, negative, or mixed.

           Example:
           Review: Fast shipping but the item arrived damaged.
           Sentiment: mixed

           Review 1: The onboarding process was smooth and the team was welcoming. 
           Review 2: The software crashes constantly and support never responds. 
           Review 3: Great price, but the documentation is nearly impossible to follow. 
           Respond in this format:
           Review 1: ...
           Review 2: ...
           Review 3: ...
           """
           }]

response = control_mode(messages, 0.7, 1)
print('content:', response.choices[0].message.content)
print()
# Adding one example improved consistency.
# The model followed the required format more strictly and produced cleaner, more structured output
# compared to zero-shot, where formatting was slightly more variable.


print('Prompt Question 3 — Few-Shot\n')
messages=[{'role': "user", "content":"""
            Classify the sentiment of each review as positive, negative, or mixed.

            Example 1:
            Review: The app is easy to use and the customer support is very helpful.
            Sentiment: positive

            Example 2:
            Review: The product stopped working after one day and I couldn’t get a refund.
            Sentiment: negative

            Example 3:
            Review: The design looks great, but the battery life is too short.
            Sentiment: mixed

            Now classify the following reviews:

            Review 1: The onboarding process was smooth and the team was welcoming.
            Review 2: The software crashes constantly and support never responds.
            Review 3: Great price, but the documentation is nearly impossible to follow.

            Respond in this format:
            Review 1: ...
            Review 2: ...
            Review 3: ...
           """
           }]

response = control_mode(messages, 0.7, 1)
print('content:', response.choices[0].message.content)
print()

# Comparison of approaches:
# Zero-shot: no examples, fastest, but format may vary slightly.
# One-shot: one example helps the model understand format better and improves consistency.
# Few-shot: multiple examples make output most stable and accurate, especially for structured tasks.
#
# When to use:
# - Zero-shot: simple tasks or when you want fast prompts.
# - One-shot: when format consistency is important.
# - Few-shot: when you need high accuracy and strict output format.


print('Prompt Question 4 — Chain of Thought\n')
messages=[{'role': "user", "content":"""
            Solve the following problem. Explain your reasoning step by step, and then clearly state the final answer.
                
            A data engineer earns $85,000 per year.
            She gets a 12% raise, then 6 months later takes a new job that pays $7,500 more per year than her post-raise salary.

            What is her final annual salary?

            Final Answer: ...
           """
           }]

response = control_mode(messages, 0.7, 1)
print('content:', response.choices[0].message.content)
print()

# Asking the model to reason step by step improves accuracy because it breaks 
# a complex problem into smaller, structured steps.
# This reduces the chance of skipping calculations or making mental math errors, 
# and helps the model follow logical progression instead of guessing the final answer directly.


print('Prompt Question 5 — Structured Output\n')
messages=[{'role': "user", "content":"""
            Analyze the 'review' below and respond only with valid JSON with keys.
                    
            Return keys: sentiment, confidence (a float from 0 to 1), and reason (one sentence)
                    
            review = "I've been using this tool for three months. It handles large datasets well, \ but the UI is clunky and the export options are limited."

           """
           }]
response = control_mode(messages, 0.7, 1)

reviews_content = response.choices[0].message.content
print("Raw response:\n", reviews_content)
print()

text = reviews_content
match = re.search(r"\{.*\}", text, re.S) # r"\{[\s\S]*?\}"
print('match\n', match)

try:
    result = json.loads(match.group())
    print("Parsed sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")

# Structured output makes it easier to reliably parse model responses in code.
# However, models may still add extra text or formatting (like markdown),
# so we often need to extract JSON safely using regex and handle errors with try/except.


print('Prompt Question 6 — Delimiters\n')

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
        {"role": "user", "content": f"""
            You will be given text inside triple backticks.
            If it contains step-by-step instructions, rewrite them as a numbered list.
            If it does not contain instructions, respond with exactly: "No steps provided."

            ```{user_text}```
         
        """}
]
)
print(response.choices[0].message.content)
print()

non_instruction_text = "I really enjoy cooking on weekends and trying new recipes from different cultures."
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
        {"role": "user", "content": f"""
            You will be given text inside triple backticks.
            If it contains step-by-step instructions, rewrite them as a numbered list.
            If it does not contain instructions, respond with exactly: "No steps provided."
         
            ```{non_instruction_text}```
        """}
]
)
print(response.choices[0].message.content)

# Delimiters help prevent the model from confusing user input with instructions 
# and reduce the risk of prompt injection or misinterpretation of where the input starts and ends.




