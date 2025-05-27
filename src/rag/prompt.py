from langchain.prompts import PromptTemplate

def get_qa_prompt():
    prompt_template_str = """You are an AI assistant acting as a technical expert/domain specialist. Your task is to analyze the provided project information and extract details for a specific project and role. The user will provide a question specifying the project name and the role (position).

Based on the context, which includes the project name, summary, narrative description, and scope of work, provide a response in valid JSON format with the following fields:
- Name of the assignment or project: The project name from the context.
- Year: The start and end year of the project (e.g., "2023-N/A").
- Location: The project location from the context.
- Client: The client name from the context.
- Position held: The role specified in the question.
- Project description: The summary (short description) from the context.
- Activities performed: Infer specific activities performed by the person in the specified role based on the 'Scope of Work' and 'Narrative Description' in the context, combined with standard industry responsibilities for the role. If insufficient details are available or the project is not found, state: "Insufficient details provided to determine specific activities for this role."

Focus ONLY on information explicitly or implicitly described in the context for the specified project. Do NOT invent information not supported by the context. If the project is not found in the context, return a JSON response with empty fields except for the project name and position, and indicate insufficient details for activities.

Context (Project Information):
{context}

User's Question (Project Name and Role):
{question}

Return the response as a JSON object, enclosed in triple backticks. Do NOT include any narrative or explanation outside the JSON. Ensure the response is valid JSON:
```json
{{
  "Name_of_the_assignment_or_project": "",
  "Year": "",
  "Location": "",
  "Client": "",
  "Position": "",
  "Project_description": "",
  "Activities_performed": ""
}}
```"""
    return PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])

def get_role_qa_prompt():
    prompt_template_str = """You are an AI assistant acting as a technical expert. Your task is to analyze the provided role-based information and extract activities for a specific role. The user will provide a query specifying the role.

Based on the context, which includes the position and associated activities, provide a response in valid JSON format with the following fields:
- position: The role specified in the query.
- activities_performed: A list of activities performed by the role, extracted from the context. If the role is found but no activities are listed, generate reasonable activities based on standard industry responsibilities for the role. If the role is not found, include a message: "Role not found in context. Generic activities based on role provided." as the first activity, followed by generic role-based activities.

Focus ONLY on information in the context for the specified role. Do NOT invent information not supported by the context, but use industry knowledge to infer plausible activities when details are sparse. Ensure the response is valid JSON.

Example response format:
```json
{
  "position": "",
  "activities_performed": []
}
```

Context (Role Information):
{context}

User's Query (Role):
{question}
"""
    return PromptTemplate(template=prompt_template_str, input_variables=["context","question"])