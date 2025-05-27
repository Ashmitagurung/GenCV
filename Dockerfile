FROM python:3.12-slim

WORKDIR /code

# Copy requirements file (fix the path)
COPY requirements.txt /code

RUN apt-get update && apt-get install git -y && apt-get install curl -y

RUN pip install -r requirements.txt

# Copy the source code (this should match your directory structure)
COPY ./src ./src

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
