import requests
import json
import time

print("Sending POST request to FastAPI server...")
start = time.time()
response = requests.post(
    "http://127.0.0.1:8000/query",
    json={"question": "What is the attention mechanism?", "stream": False}
)
print(f"Response Time: {time.time() - start:.2f}s")
print("-" * 50)
if response.status_code == 200:
    data = response.json()
    print(f"ANSWER:\n{data['answer']}")
    print("-" * 50)
    print(f"Citations Valid: {data['citations_valid']}")
    print(f"Citations Invalid: {data['citations_invalid']}")
    print(f"Retrieved Docs: {data['retrieved_docs']}")
else:
    print(f"Error {response.status_code}: {response.text}")
