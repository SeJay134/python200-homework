# Part 2: Mini-Project — Job Application Helper

from dotenv import load_dotenv
from openai import OpenAI
import json
import re
import textwrap

print('Task 1: Setup and System Prompt')

if load_dotenv():
    print("Successfully loaded api key")

if load_dotenv():
    print("Successfully loaded api key")
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

prompt = """
You are an AI job application coach helping users improve resumes, 
tailor experience for career transitions, draft cover letters, 
and prepare professional application materials.

Your purpose is to help users translate their skills and accomplishments into language 
that aligns with the roles they are applying for, especially when they are changing industries or entering a new field.

Guidelines:
- Stay focused only on job application support, including resume bullets, cover letters, 
follow-up questions, interview preparation, and related career materials.
- Ask clarifying questions when needed before drafting or revising materials so your help can be tailored to the user’s goals.
- Emphasize clear, professional, achievement-focused writing and suggest ways to quantify impact when possible.
- When rewriting content, preserve the truth of the user’s experience and do not invent qualifications or exaggerate achievements.
- Always remind the user to review and edit any generated material before submitting it to employers.
- Acknowledge that you may not know the specific norms, expectations, or conventions of every industry, 
company, or region, and remind the user to use their own judgment.
- Provide supportive, practical guidance and explain suggestions when helpful.
- If a request falls outside job application support, redirect the conversation back to career-related assistance.

Extra ruls:
- Never invent, assume, or exaggerate any qualifications, experience, metrics, or achievements that are not explicitly stated by the user.
- If information is missing or unclear, ask clarifying questions instead of filling in gaps.
- Maintain factual accuracy while improving clarity, structure, and impact of the user’s experience.
"""

# ---------------------------------------------------------------------------------------
# Project Task 1: System prompt	
# 10	
# Prompt is specific (role, audience, constraints); 
# comment explains at least one deliberate design choice

# Deliberate design choice:
# I added a rule prohibiting the model from inventing qualifications 
# or exaggerating achievements, # because language models tend to "fill gaps" plausibly, 
# which is dangerous in resume writing.
# ---------------------------------------------------------------------------------------------

print('Task 2: Bullet Point Rewriter')

# ------------------------------------------------------------------------------------------
# Project Task 2: Bullet rewriter	
# 15	
# Delimiters used; JSON parsed correctly; original and improved bullets printed side by side
# ------------------------------------------------------------------------------------------
def print_side_by_side(data, width=50):
    for i in data:

        left = textwrap.wrap(i["original"], width)
        right = textwrap.wrap(i["improved"], width)

        max_lines = max(len(left), len(right))

        print(f"{'Original':<50} | Improved")

        for j in range(max_lines):
            l = left[j] if j < len(left) else ""
            r = right[j] if j < len(right) else ""
            print(f"{l:<50} | {r}")

        print("-" * 110)

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
        You are a professional resume coach helping a career changer.
        Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
        Use strong action verbs. Do not invent facts that aren't implied by the original.

        Return ONLY a valid JSON list. Each item should have two keys:
        "original" (the original bullet) and "improved" (your rewritten version).

        Bullet points:
        ```
        {bullet_text}
        ```
        """

    messages = [{"role": "user", "content": prompt}]

    # Your code here: call get_completion(), parse the JSON, and return the result
    response = get_completion(messages)
    print('response\n', response)

    match = re.search(r"\[[\s\S]*\]", response, re.S)

    if not match:
        print("Error: no JSON found")
        return []

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")
        return []

    # for i in data:
    #     print('-'*50)
    #     print(f'Improved: {i['improved']} | Original: {i['original']}')
    #     #print('Original:', i['original'])

    return data

# Test it with these starter bullets:

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]


test_01 = rewrite_bullets(bullets)
#print(test_01)
side_by_side_data = print_side_by_side(test_01)
print(side_by_side_data)

# The original bullets were weak because they were vague, lacked measurable impact, and used generic verbs 
# like "helped" and "made." The model improved them by using stronger action verbs, adding specific details, 
# and including measurable outcomes such as percentages and time savings to make the experience more impactful and professional.

