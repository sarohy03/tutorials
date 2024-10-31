from pymongo import MongoClient
from dotenv import load_dotenv
import os
import requests
import json
from openai import AzureOpenAI

load_dotenv()

# Environment variables
embedding_url = os.getenv("HUGGING_FACE_MODEL")
token = os.getenv("HUGGING_FACE_TOKEN")
key = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MONGO_LINK = os.getenv("MONGO_LINK")


def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": text}
    )
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

    return response.json()


def query_mongo(query: str) -> str:
    client = MongoClient(MONGO_LINK)
    db = client["property"]
    collection = db["data"]

    query_vector = generate_embedding(query)

    results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_vector,
                "path": "embedding",
                "numCandidates": 100,
                "limit": 1,
                "index": "PlotSemanticSearch"
            }
        }
    ])

    for document in results:
        return document["text"]  # Return the text of the first matching document

    return ""


def generate_completion(content: str) -> str:
    """Generates a completion using Azure OpenAI based on the provided content."""
    client = AzureOpenAI(
        api_version="2023-07-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=key
    )

    completion = client.chat.completions.create(
        model="asssessiq-gpt-4o",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    response = json.loads(completion.to_json())
    return response["choices"][0]["message"]["content"]


# Example usage
query = "AFTER-TAX INCOME"
value = query_mongo(query)
content = f"Query: {query}\nYou can have data from here: {value}"
response_content = generate_completion(content)

print(response_content)
