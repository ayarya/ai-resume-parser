import os
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("api error")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-20b"

job_description = """
Catallaxy is looking for a talented full-stack developer (node.js) able to quickly develop, deliver and support clean, scalable and high quality code for blockchain applications.

You will work closely with the head of product
You will have the opportunity to grow inside the organization
Job Location: Montreal, Quebec, Canada

Job type: Full time

Responsibilities:

Design, develop, deliver, and support web-based applications and APIs
Write maintainable code with extensive tests and version control
Ensure the scalability and quality of all code and products
Build applications that focus on great user experience
Create automated testing to quickly and confidently ship code
Assist with product documentation, unit testing and ensuring overall product quality
Participation in daily scrums, design discussions, code reviews and code analysis
Help choose and define the tech stack and tools
Requirements

3+ years experience in software development
2+ years with JavaScript technologies: Node.js and React/Angular or similar
Experience building scalable web applications using modern frameworks (React/ Angular, etc.) and tools (npm, webpack, etc.)
Good understanding of both front end (JavaScript, HTML, CSS) and server side (Node.js, express) components
Experience with API design and development (REST, JSON, RESTFull services).
CSS preprocessors like Sass or LESS
Front-end (CSS) frameworks like Foundation or Bootstrap
JS-based build tools like Grunt, Gulp, and Bower
Experience using test-driven development and writing accurate tests
Experience with databases SQL and NoSQL (MongoDB, Postgres, etc...)
Ability to translate technical requirements into elegant code solutions
Focus on continuous improvement
Experience with cloud providers (AWS/OVH/Google, etc.) and service deployment
Embrace Agile/Scrum methodologies and development best practices
Experience with Git
Keeping in mind Security concerns throughout the process
Excellent problem solving and debugging skills to rapidly identify and fix bugs and problems
Ability to identify, implement and manage code libraries that minimize repetitive code and improve application design
Use code optimization techniques to improve robustness and performance of software
Human skills:

Proven and strong communication skills (in English and/or French)
Team player, but also fully able to efficiently accomplish tasks independently
Strong analytical and problem solving skills
You get things done
Willingness to wear multiple hats and be flexible in a fast-moving environment
Nice to have

Bachelor or master degree in Computer Science, a related field or equivalent experience
Knowledge/experience with other programing languages (C++, Go, .Net, Python)
Blockchain, bitcoin interest or willing to learn
Interest in smart contract (Ethereum)
Mobile dev (IOS, Android) or ability to hire and manage someone if needed
Experience with distributed systems and decentralized technologies
Experience with message or event driven architectures (such as RabbitMQ or Kafka)
Continuous Integration (CI) and Continuous Delivery (CD) DevOps experience.
Experience with Docker
Knowledge of Micro Services concepts and architecture
Please submit a resume, and a short text highlighting your experience with the mentioned programing languages and tools as well as your experience developing web applications and products.

About Catallaxy

With more than 15 Montreal-based experts and nearly 50 strategic partners worldwide, our service offering combines the technical expertise of the best core Blockchain talents and the strategic consulting power of Raymond Chabot Grant Thornton. Our objective is to prepare businesses for their blockchain transformation.

Catallaxy is also launching the Blockhouse. The Blockhouse is a unique innovation center bringing together the most influential players in the blockchain community. Strategically based in Montréal, the Blockhouse is a unique international showcase for decentralized and distributed technologies.
"""


class JobD(BaseModel):
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_experience: float | None
    education_requirements: list[str]
    responsibilities: list[str]


jobd_schema = JobD.model_json_schema()

system_prompt = f"""
You are an expert HR assistant.

Your job is to analyze job descriptions and extract structured information from them.

Return ONLY valid JSON matching this schema:

{jobd_schema}
IMPORTANT:
Do NOT return the schema itself.
Do NOT return fields like "properties", "title" or "type".
Fill the schema with actual information extracted from the job description.

If minimum experience is not mentioned, return null.
If information for a list is missing, return an empty list.
Do not invent information
"""

user_prompt = f"""
Analyze the following job description:
{job_description}
"""

message_system = {"role": "system", "content": system_prompt}
message_user = {"role": "user", "content": user_prompt}
response_format = {"type": "json_object"}

messages = [message_system, message_user]

response = client.chat.completions.create(
    model=model, messages=messages, response_format=response_format
)

answer = response.choices[0].message.content

raw_json = answer

import json

job_data = json.loads(raw_json)

job = JobD(**job_data)

print(job.minimum_experience)
print(job.education_requirements)


# parse real
class MatchResult(BaseModel):
    score: float
    details: dict


class Experience(BaseModel):
    company: str | None = None  # means optional
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []


resume_schema = Resume.model_json_schema()


def final_score(job, resume):
    match_schema = MatchResult.model_json_schema()
    prompt = f"""
    You are an HR recruiter.

    Compare the candidate's resume with the job description.

    JOB DESCRIPTION:
    {job.model_dump_json(indent=2)}

    CANDIDATE RESUME:
    {resume.model_dump_json(indent=2)}
    Return JSON matching this schema:
    {match_schema}

    Give me:

    1. Candidate name
    2. Matching skills
    3. Missing important skils
    4. Whether experience requirement is met
    5. Overall match percentage from 0 to 100
    6. A short final verdict

    keep the response concise and easy to read.
    """
    message_user = {"role": "user", "content": prompt}
    response_format = {"type": "json_object"}
    messages = [message_user]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        max_completion_tokens=8192,
    )
    raw_output = response.choices[0].message.content
    data = json.loads(raw_output)
    return MatchResult(**data)


def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in skills section, work experience, internships or projects.

    Return ONLY valid json matching this schema:
    {resume_schema}

    Important Rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. if a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """

    user_prompt = f"""
    Parse the following resume:
    {resume_text}
    """

    message_system = {"role": "system", "content": system_prompt}
    message_user = {"role": "user", "content": user_prompt}
    response_format = {"type": "json_object"}
    messages = [message_system, message_user]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        reasoning_effort="low",
        max_completion_tokens=8192,
        temperature=0.1,
    )
    raw_output = response.choices[0].message.content
    data = json.loads(raw_output)
    resume = Resume(**data)
    return resume


from pypdf import PdfReader
from docx import Document


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_docx(file_path):
    document = Document(file_path)
    text = ""
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text += cell.text + "\n"
    return text


def read_resume(file_path):
    if file_path.suffix.lower() == ".pdf":
        return read_pdf(file_path)
    elif file_path.suffix.lower() == ".docx":
        return read_docx(file_path)
    else:
        return None


from pathlib import Path
import time

resume_folder = Path("resumes")
all_results = []
for file_path in resume_folder.iterdir():
    if file_path.suffix.lower() not in [".pdf", ".docx"]:
        continue
    print("\nProcessing:", file_path.name)
    resume_text = read_resume(file_path)
    parsed_resume = parse_resume(resume_text)
    print(parsed_resume)
    time.sleep(5)
    result = final_score(job, parsed_resume)
    time.sleep(5)
    print("Score:", result.score)
    all_results.append(
        {"name": parsed_resume.name, "score": result.score, "details": result.details}
    )
all_results.sort(key=lambda candidate: candidate["score"], reverse=True)
print(all_results)
