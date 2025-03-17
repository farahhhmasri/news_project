import requests
import json

url = "http://127.0.0.1:8000/predict"
headers = {"Content-Type": "application/json"}
with open('test_data.json', 'r') as file:
    data = json.load(file)

response = requests.post(url, json=data, headers=headers)
print("Response:", response)  


### postman
# http://127.0.0.1:8000/predict
# [
#     {"id": 1, "news": "The stock market crashed today."},
#     {"id": 2, "news": "Scientists discover a new exoplanet."}
# ]