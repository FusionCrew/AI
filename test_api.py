import requests
import sys

url = "http://127.0.0.1:8000/api/sign-language/translate"
# Assume running from project root or adjust path
file_path = "signLanguage/test_data/test.mp4"

try:
    with open(file_path, 'rb') as f:
        print(f"Sending request to {url} with file {file_path}")
        response = requests.post(url, files={'video': f})
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response: {response.json()}")
        except:
            print(f"Response content: {response.text}")
except Exception as e:
    print(f"Error: {e}")
