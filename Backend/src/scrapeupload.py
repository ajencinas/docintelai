import os
import re
import requests
import asyncio
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient, TableEntity
import pandas as pd
from azure.core.exceptions import ResourceExistsError, HttpResponseError
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone

# Environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
load_dotenv(dotenv_path)
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = "ratecasefiles"
TARGET_URL = "https://apps.puc.state.or.us/edockets/docket.asp?DocketID=24011"

# MongoDB data
VCORE_URI = os.getenv("VCORE_URI")
DATABASE_NAME = "RateCaseDB"
COLLECTION_NAME = "FilesMetadata"

# Azure Clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

# Initialize MongoDB Client
mongo_client = AsyncIOMotorClient(VCORE_URI)
db = mongo_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Function to sanitize filenames
def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

# Function to resolve redirected links
def resolve_redirected_url(url):
    try:
        response = requests.head(url, allow_redirects=True)
        if response.status_code == 200:
            return response.url  # Return the final resolved URL
        else:
            print(f"Failed to resolve URL: {url}, Status Code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error resolving URL: {url}, Error: {e}")
        return None

def scrape_files(target_url, max_files):
    """
    Scrape file links from a target URL. Only includes valid .pdf files with "FileName=" in the URL.
    
    Parameters:
        target_url (str): The URL to scrape.
        max_files (int): The maximum number of files to process.

    Returns:
        list: A list of dictionaries containing file metadata (URL, filename, source URL).
    """
    response = requests.get(target_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    file_links = []

    for link_tag in soup.find_all("a", href=True):
        # Stop if we reach the maximum file limit
        if len(file_links) >= max_files:
            break

        # Extract the href attribute
        href = link_tag["href"]

        # Skip links that don't match the expected pattern
        if "edocs.asp" not in href or "FileName=" not in href:
            print(f"Skipping non-matching link: {href}")
            continue

        # Construct the full URL if necessary
        full_url = href if href.startswith("http") else f"https://apps.puc.state.or.us/edockets/{href}"

        # Resolve the redirected URL to get the final file link
        resolved_url = resolve_redirected_url(full_url)

        # Only include PDF files
        if resolved_url and resolved_url.endswith(".pdf"):
            filename = os.path.basename(resolved_url)
            file_links.append({
                "url": resolved_url,
                "filename": filename,
                "metadata": {
                    "source_url": full_url,  # Include the original source URL
                },
            })
            print(f"Found PDF file: {resolved_url}")
        else:
            print(f"Skipping non-PDF file: {resolved_url}")

    return file_links

# Function to upload file to Azure Blob Storage
async def upload_to_blob(file_url, filename, blob_directory, container_client, ratecase):
    unique_filename = os.path.join(blob_directory, f"{ratecase}_{sanitize_filename(filename)}")  # ✅ Append ratecase
    response = requests.get(file_url, stream=True)
    response.raise_for_status()
    blob_client = container_client.get_blob_client(blob=unique_filename)
    blob_client.upload_blob(response.content, overwrite=True)
    return blob_client.url, unique_filename  # ✅ Return the new filename

# Function to insert metadata into MongoDB
async def insert_into_mongo(filename, metadata, blob_url, ratecase, year, company):
    """Inserts metadata into Azure Cosmos DB for MongoDB."""
    # Remove the directory prefix if present
    clean_filename = os.path.basename(filename)  # Extracts only the filename, removing directories
    
    document = {
        "file_name": clean_filename,  # ✅ Store only the filename in MongoDB
        "source_url": metadata.get("source_url", "Unknown"),
        "blob_url": blob_url,
        "ratecase": ratecase,
        "year": year,
        "company": company,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    await collection.update_one(
        {"file_name": filename, "ratecase": ratecase},  # ✅ Prevents overwriting across ratecases
        {"$set": document},
        upsert=True
    )

# Scrape and upload
async def scrapeandupload(target_url, blob_directory, max_files=500, yearfiled=None, company=None):
    '''Scrape files from a target URL and upload them to Azure Blob Storage.'''

    # Ensure Azure resources exist
    if not container_client.exists():
        container_client.create_container()
    
    file_links = scrape_files(target_url, max_files)

    for file in file_links:
        try:
            if not file["url"].startswith("http"):
                print(f"Invalid URL skipped: {file['url']}")
                continue

            # Upload to blob
            blob_url, unique_filename = await upload_to_blob(file["url"], file["filename"], blob_directory, container_client, blob_directory)       
            
            # Insert metadata into MongoDB
            await insert_into_mongo(unique_filename, file["metadata"], blob_url, blob_directory, yearfiled, company)
            print(f"Uploaded and logged: {file['filename']}")

        except requests.exceptions.RequestException as req_err:
            print(f"Failed to download {file['filename']}: {req_err}")
        except Exception as e:
            print(f"Failed to process {file['filename']}: {e}")


# Query and display the data
async def query_mongo():
    """Retrieves and displays metadata from MongoDB."""
    documents = await collection.find({}, {"_id": 0}).to_list(length=None)
    df = pd.DataFrame(documents)

    # Save to Excel
    df.to_excel("ratecasefiles.xlsx", index=False)
    print("Data exported to ratecasefiles.xlsx")

async def main():
    await collection.create_index("ratecase")  # ✅ Indexing for fast retrieval
    await scrapeandupload(TARGET_URL, "PGE2024", max_files=5, yearfiled="2024", company="Portland General Electric(PGE, POR)")
    await query_mongo()

if __name__ == "__main__":
    asyncio.run(main())