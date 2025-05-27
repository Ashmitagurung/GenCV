import os
from langchain_aws import ChatBedrock
from src.config import settings 

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

def get_llm():
    # Set environment variables programmatically
    os.environ['AWS_ACCESS_KEY_ID'] = settings.aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = settings.aws_secret_access_key
    os.environ['AWS_DEFAULT_REGION'] = settings.aws_default_region
    
    return ChatBedrock(
        region_name=settings.aws_default_region,
        model_id=model_id,
        model_kwargs=dict(temperature=0),
    )