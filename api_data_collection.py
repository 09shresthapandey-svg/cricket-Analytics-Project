import requests
import zipfile
import io
import os

# Create folders
os.makedirs("data/raw", exist_ok=True)

# Cricsheet T20 API endpoint
url = "https://cricsheet.org/downloads/t20s_csv2.zip"

response = requests.get(url)

# Extract ZIP contents
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
zip_file.extractall("data/raw")

print("T20 cricket data downloaded and extracted successfully.")
