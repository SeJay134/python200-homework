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

print('Task 3: Cover Letter Generator')
def generate_cover_letter(job_title: str, background: str) -> str:
# -------------------------------------------------------------------------------
# Project Task 3: Cover letter	
# 10	
# Two or more few-shot examples present in the prompt; 
# output is tailored to the input; comment explains example choices

    # Extra examples explanation:
    # I chose both examples to represent common career transition scenarios:
    # (1) healthcare → data analytics and (2) finance operations → software engineering.
    # Both cases show a clear shift from non-technical or domain-heavy roles into more technical fields.
    # This helps the model learn how to connect past domain experience to new target roles without sounding generic.
    # The examples also demonstrate a strong narrative structure: past experience → turning point → new skills → target role fit.
    # This encourages the model to produce opening paragraphs that are specific, story-driven, and tailored rather than template-like.
# -------------------------------------------------------------------------------

    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result

    result = get_completion(messages)
    print(result)

    return result

# Test it with:
job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."
generate_cover_letter(job_title, background)
print()

# I chose examples that show career transitions because the goal is to help the model learn how 
# to connect past experience to new roles in a narrative way. Few-shot prompting helps control the tone, 
# structure, and level of specificity in the output, making the response more natural and 
# less generic compared to zero-shot prompting.
