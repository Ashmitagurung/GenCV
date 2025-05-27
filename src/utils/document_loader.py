import os
import json
from langchain_core.documents import Document
from src.utils.logger import setup_logger

def load_projects(projects_dir: str, json_file: str) -> list:
    """Loads project data from a specific JSON file in the specified directory."""
    logger = setup_logger()
    loaded_projects_data = []
    
    if not os.path.isdir(projects_dir):
        logger.error(f"Directory '{projects_dir}' not found. Please create it and add JSON files.")
        return []

    file_path = os.path.join(projects_dir, json_file)
    if not os.path.isfile(file_path):
        logger.error(f"JSON file '{file_path}' not found.")
        return []
    
    logger.info(f"Loading JSON file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = json.load(f)
            
            if isinstance(file_content, dict):
                loaded_projects_data.append(file_content)
            elif isinstance(file_content, list):
                for item_idx, item in enumerate(file_content):
                    if isinstance(item, dict):
                        loaded_projects_data.append(item)
                    else:
                        logger.warning(f"Item at index {item_idx} in '{file_path}' is not a JSON object. Skipping.")
            else:
                logger.warning(f"Content of '{file_path}' is not a JSON object or list. Skipping.")
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode JSON from '{file_path}': {e}. Skipping.")
        return []
    except Exception as e:
        logger.error(f"Error processing '{file_path}': {e}. Skipping.")
        return []

    if not loaded_projects_data:
        logger.warning(f"No valid project data loaded from '{file_path}'.")
        return []

    documents = []
    for proj_idx, proj in enumerate(loaded_projects_data):
        default_title = f'Project {proj_idx+1} (No Title Provided)'
        title = proj.get('Full Project Name in Contract', default_title)
        summary = proj.get('Summary (very short description of project for CV)', 'N/A')
        narrative_description = proj.get('Narrative Description of Project (Objective)', 'N/A')
        scope_of_work = proj.get('Description of Actual Services (Scope of work)', 'N/A')
        start_date = proj.get('Start Date', 'N/A')
        end_date = proj.get('End Date', 'N/A')
        year = f"{start_date}-{end_date}" if start_date != 'N/A' or end_date != 'N/A' else 'N/A'
        location = proj.get('Location Within Country', proj.get('Country', 'N/A'))
        client = proj.get('Name of the Client', 'N/A')
        
        tech_data = proj.get('Technology Used', "")
        technologies_list = []
        if isinstance(tech_data, str) and tech_data.strip():
            technologies_list = [t.strip() for t in tech_data.split(',') if t.strip()]
        elif isinstance(tech_data, list):
            technologies_list = [str(t).strip() for t in tech_data if str(t).strip()]

        content_parts = [
            f"Project Name: {title}",
            f"Summary: {summary}",
            f"Narrative Description: {narrative_description}",
            f"Scope of Work: {scope_of_work}",
            f"Year: {year}",
            f"Location: {location}",
            f"Client: {client}"
        ]
        if technologies_list:
            content_parts.append(f"Technologies Used: {', '.join(technologies_list)}")
        
        content = "\n".join(content_parts)
        metadata = {
            "title": title,
            "year": year,
            "location": location,
            "client": client,
            "technologies_list": technologies_list,
            "source_file": os.path.basename(file_path)
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    logger.info(f"Prepared {len(documents)} documents from '{file_path}'.")
    return documents

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load project documents from a specific JSON file.")
    parser.add_argument("projects_dir", help="Path to the directory containing JSON project files.")
    parser.add_argument("json_file", help="Name of the specific JSON file to load.")
    args = parser.parse_args()

    documents = load_projects(args.projects_dir, args.json_file)

    if documents:
        print("\n--- Preview of First Document ---")
        print(documents[0].page_content)
        print("\nMetadata:", documents[0].metadata)
    else:
        print("No documents were created.")