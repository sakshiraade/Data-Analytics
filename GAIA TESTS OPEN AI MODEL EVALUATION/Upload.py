import os
import requests
from bs4 import BeautifulSoup
import boto3

# S3 Configuration
s3_client = boto3.client('s3')
s3_bucket = 'gaiaproject'  # Replace with your S3 bucket name

# Hugging Face access token (replace with your actual token)
huggingface_access_token = "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"

# Local folder to store the files temporarily
local_folder = './gaia_files'
os.makedirs(local_folder, exist_ok=True)

# URL of the dataset page
dataset_url = 'https://huggingface.co/datasets/gaia-benchmark/GAIA/tree/main/2023/validation/'

def scrape_file_urls(dataset_url):
    """Scrape the dataset page for file URLs."""
    file_urls = []
    headers = {
        "Authorization": f"Bearer {huggingface_access_token}"
    }
    response = requests.get(dataset_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Look for all links to files
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and 'resolve/main' in href:
                file_url = 'https://huggingface.co' + href
                file_urls.append(file_url)
        print(f"Found {len(file_urls)} file URLs.")
    else:
        print(f"Failed to retrieve the dataset page. Status code: {response.status_code}")
    
    return file_urls

def download_file(url, local_path):
    """Download a file from a URL to a local path."""
    headers = {
        "Authorization": f"Bearer {huggingface_access_token}"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {url} to {local_path}")
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}")
    except PermissionError as e:
        print(f"Permission denied for {local_path}: {e}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def upload_to_s3(local_path, s3_key):
    """Upload a local file to the specified S3 bucket."""
    try:
        s3_client.upload_file(local_path, s3_bucket, s3_key)
        print(f"Uploaded {local_path} to S3: {s3_key}")
    except Exception as e:
        print(f"Failed to upload {local_path} to S3. Error: {e}")

# Scrape the dataset page for file URLs
file_urls = scrape_file_urls(dataset_url)

# Loop through the file URLs, download them, and upload them to S3
for file_url in file_urls:
    file_name = os.path.basename(file_url).split("?")[0]  # Remove query parameters like ?download=true
    local_file_path = os.path.join(local_folder, file_name)

    # Ensure directory exists and has write permission
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    
    # Download file
    download_file(file_url, local_file_path)

    # Upload file to S3
    s3_key = f'gaia/2023/validation/{file_name}'
    upload_to_s3(local_file_path, s3_key)
