import json
import asyncio
import warnings
import faiss
import requests
import pickle
import numpy as np
import io
from motor.motor_asyncio import AsyncIOMotorClient
import openai   
import redis.asyncio as aioredis
from redis.asyncio.cluster import RedisCluster
import os
from pathlib import Path
from dotenv import load_dotenv
from pymongo import ASCENDING
from langchain_openai import OpenAIEmbeddings

# Fetch paths from environment variables or use defaults
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
load_dotenv(dotenv_path)

# MongoDB Connection
MONGO_URI = os.getenv("VCORE_URI")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
SAS_TOKEN = os.getenv("SAS_TOKEN")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATABASE_NAME = "RateCaseDB"
COLLECTION_NAME = "FilesMetadata"

print("🔍 Redis Connection Details:")
print(f"Host: {REDIS_HOST}")
print(f"Port: {REDIS_PORT}")
print(f"Password Set: {'Yes' if REDIS_PASSWORD else 'No'}")

# Suppress CosmosDB warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Redis Cluster Client
redis_client = RedisCluster(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,  # Ensures responses are strings
    ssl=True  # Enforce SSL for secure connections
)

# Initialize Async MongoDB Client
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Initialize OpenAI Embedding Model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Function to log debug messages
async def debug(*args):
    """Logs debug messages to a file."""
    message = " ".join(map(str, args))
    with open("debug_log.txt", "a") as debug_file:  # Open the file in append mode
        debug_file.write(message + "\n")

#Gets all the possible metadata options
async def get_metadata_options():
    """Fetch metadata options from Redis cache or MongoDB if not cached."""
    cache_key = "metadata_options"

    # Try getting from Redis
    cached_data = await redis_client.get(cache_key)
    if cached_data:
        print("✅ Cache hit: Returning metadata from Redis")
        return json.loads(cached_data)

    print("🚀 Cache miss: Fetching metadata from MongoDB")
    
    # Fetch from MongoDB if cache miss
    companies_cursor = collection.distinct("company")
    years_cursor = collection.distinct("year")
    companies, years = await asyncio.gather(companies_cursor, years_cursor)

    metadata = {"company": list(companies), "year": list(years)}

    # Store in Redis (with expiration of 1 hour)
    await redis_client.setex(cache_key, 3600, json.dumps(metadata))

    return metadata

#Infers the needed filters for a given query
async def infer_needed_filters(query):
    """Infers metadata filters needed for a given query using OpenAI's GPT-4 model.
    Args:
        query (str): The user's query.
        Returns:
        dict: A dictionary containing the inferred metadata filters.
    """
    
    # Fetch metadata options from MongoDB
    valid_values = await get_metadata_options()

    # Format allowed values
    allowed_values_prompt = (
        "Allowed values for filtering:\n"
        f"- company: {'/ '.join(valid_values['company'])}\n"
        f"- year: {'/ '.join(valid_values['year'])}\n\n"
    )

    # Build the LLM prompt
    system_prompt = (
        "You are an AI assistant that determines relevant metadata filters "
        "for a user's query. Based on the query, identify which metadata fields should be applied "
        "and return the field names and values as JSON. "
        "Allowed values are separated by forward slashes and they may include parenthesis - you need to return exactly the name as written. Only use the following allowed values for filtering:\n\n"
        + allowed_values_prompt
        + "Return only valid filters, or an empty JSON if no filtering is needed."
    )

    user_prompt = f"Query: {query}\n\nWhich metadata filters should apply? Return the JSON response only."
 
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=100,
        temperature=0,
    )

    # Debug output
    # ✅ Fix: Corrected response parsing
    inferred_filters = response.choices[0].message.content.strip()
    #print("Filters inferred for query:", inferred_filters)

    # Parse JSON response and return filters
    return json.loads(inferred_filters)

#Finds the right files to look into to answer the query
async def findrightfiles(query):
    """Filters files in MongoDB based on inferred metadata from a query."""
    
    # Infer filters using the existing inference function
    inferred_filters = await infer_needed_filters(query)
    print("📌 Inferred Filters:", inferred_filters)

   # Build MongoDB query dynamically based on inferred filters
    mongo_query = {key: value for key, value in inferred_filters.items()}
    
    print("🔍 MongoDB Query:", mongo_query)

    # Retrieve matching files from MongoDB
    matching_files = await collection.find(mongo_query).to_list(None)

    print(f"✅ Found {len(matching_files)} matching files.")

    return matching_files

# Perform vector search within filtered files
async def vector_search_within_files(query_text, files, top_k=5):
    """Performs a vector search among a filtered set of files."""
    
    query_vector = embeddings.embed_query(query_text)  # Generate query embedding
    file_ids = [file["_id"] for file in files]
    
    mongo_pipeline = [
        {
            "$search": {
                "cosmosSearch": {
                    "vector": query_vector,
                    "path": "summary.summary_vector",
                    "k": top_k,
                    "efSearch": 40
                }
            }
        },
        {
            "$match": {
                "_id": {"$in": file_ids}  # ✅ Correct: Separate $match stage
            }
        }
    ]
    
    print("🔍 Performing Vector Search within filtered files...")
    matching_files = await collection.aggregate(mongo_pipeline).to_list(top_k)
    print(f"✅ Found {len(matching_files)} most relevant files.")

    sample_vector = await collection.find_one({}, {"summary.summary_vector": 1})
    
    if sample_vector and "summary" in sample_vector and "summary_vector" in sample_vector["summary"]:
        if len(sample_vector["summary"]["summary_vector"]) != len(query_vector):
            print(f"⚠️ Dimension mismatch! DB vectors have {len(sample_vector['summary']['summary_vector'])}, but query has {len(query_vector)}.")
    else:
        print("⚠️ No sample vector found in the database.")
    
    for file in matching_files:
        if "summary" not in file or "summary_vector" not in file["summary"]:
            print(f"⚠️ File {file['_id']} is missing 'summary_vector'")

    return matching_files

#Look into the FAISS file associated to a file and find right vectors. Return content and metadata
async def get_faiss_vectors(matching_files, query_vector, top_k=5):
    """
    Retrieve the most relevant K vectors and their associated text from FAISS indexes stored in Azure Blob.

    Args:
        matching_files (list): List of file metadata dicts from the database.
        query_vector (list): The embedding vector of the query.
        top_k (int): Number of top vectors to retrieve.

    Returns:
        list: The top_k most relevant vectors, along with their text and metadata.
    """
    relevant_data = []

    for file_metadata in matching_files:
        faiss_urls = file_metadata.get("vector_db_url", [])
        faiss_file_url = next((url for url in faiss_urls if url.endswith(".faiss")), None)
        pkl_file_url = next((url for url in faiss_urls if url.endswith(".pkl")), None)

        if not faiss_file_url or not pkl_file_url:
            print(f"⚠️ No FAISS file found for {file_metadata['file_name']}")
            continue

        # ✅ Append SAS token to URLs
        faiss_file_url += f"&{SAS_TOKEN}" if "?" in faiss_file_url else f"?{SAS_TOKEN}"
        pkl_file_url += f"&{SAS_TOKEN}" if "?" in pkl_file_url else f"?{SAS_TOKEN}"

        print(f"📥 Downloading FAISS index from {faiss_file_url}...")

        # Download FAISS index file into memory (byte stream)
        response_faiss = requests.get(faiss_file_url)
        response_pkl = requests.get(pkl_file_url)

        if response_faiss.status_code != 200 or response_pkl.status_code != 200:
            print(f"⚠️ Failed to download FAISS or metadata for {file_metadata['file_name']}")
            continue
        
        try:
            # ✅ Load FAISS index
            faiss_bytes = response_faiss.content
            index = faiss.deserialize_index(np.frombuffer(faiss_bytes, dtype=np.uint8))

            # ✅ Load FAISS metadata 
            docstore, id_map = pickle.loads(response_pkl.content)
            
            # ✅ Ensure metadata contains text mappings
            if not isinstance(docstore, dict) and hasattr(docstore, "search"):
                print(f"✅ FAISS metadata loaded correctly for {file_metadata['file_name']}")

            # ✅ Perform FAISS search
            query_vector_np = np.array([query_vector], dtype="float32")
            distances, indices = index.search(query_vector_np, top_k)

            for idx, distance in zip(indices[0], distances[0]):
                doc_id = id_map.get(idx)  # Get document ID

                if doc_id and doc_id in docstore._dict:
                    doc = docstore._dict[doc_id]  # Retrieve Document object

                    relevant_data.append({
                        "text": doc.page_content,
                        "file_name": file_metadata["file_name"],
                        "page_number": doc.metadata.get("page_number", "Unknown"),
                        "blob_url": doc.metadata.get("blob_url", "Unknown"),
                        "distance": float(distance)  # Convert NumPy float to regular float
                    })

        except Exception as e:
            print(f"⚠️ Error processing FAISS index for {file_metadata['file_name']}: {e}")

    return relevant_data


async def get_text_from_faiss_pkl(matching_files, retrieved_vectors, sas_token):
    """
    Retrieve the actual text associated with the retrieved FAISS vectors from the .pkl metadata.

    Args:
        matching_files (list): List of file metadata dicts.
        retrieved_vectors (list): List of (file_name, vector_index, distance).
        sas_token (str): The Azure SAS token for accessing blob storage.

    Returns:
        list: List of relevant text snippets associated with the vectors.
    """
    relevant_texts = []

    for file_metadata in matching_files:
        file_name = file_metadata["file_name"]

        # Find the corresponding .pkl metadata file URL
        pkl_urls = file_metadata.get("vector_db_url", [])
        pkl_file_url = next((url for url in pkl_urls if url.endswith(".pkl")), None)

        if not pkl_file_url:
            print(f"⚠️ No FAISS .pkl metadata found for {file_name}")
            continue

        # Append SAS token to the URL
        if "?" in pkl_file_url:
            pkl_file_url += f"&{sas_token}"
        else:
            pkl_file_url += f"?{sas_token}"

        print(f"📥 Downloading FAISS metadata from {pkl_file_url}...")

        # Download the FAISS .pkl file
        response = requests.get(pkl_file_url)
        if response.status_code != 200:
            print(f"⚠️ Failed to download FAISS metadata for {file_name}")
            continue

        try:
            # Load the FAISS metadata
            faiss_metadata = pickle.loads(response.content)

            # Check if metadata contains text mappings
            if "texts" not in faiss_metadata:
                print(f"⚠️ No text mappings found in FAISS metadata for {file_name}")
                continue
            
            # ✅ Extract text and metadata
            text_chunks = faiss_metadata["texts"]
            metadata_chunks = faiss_metadata["metadata"]
                                             
            # ✅ Find matching vectors for this file
            file_vectors = [v for v in retrieved_vectors if v[0] == file_name]

            for _, vector_index, _ in file_vectors:
                if 0 <= vector_index < len(text_chunks):  # Ensure index is valid
                    relevant_texts.append({
                        "text": text_chunks[vector_index],
                        "file_name": file_name,
                        "page_number": metadata_chunks[vector_index].get("page_number", "Unknown"),
                        "blob_url": metadata_chunks[vector_index].get("blob_url", "Unknown")
                    })

        except Exception as e:
            print(f"⚠️ Error loading FAISS metadata: {e}")

    return relevant_texts


# Example Usage
async def main():
    query = "Find documents related to Portland General Electric in 2024"
    
    # ✅ Step 1: Find relevant files
    filtered_files = await findrightfiles(query)
    if not filtered_files:
        print("⚠️ No matching files found.")
        return
    
    # ✅ Step 2: Perform vector search within the relevant files
    top_files = await vector_search_within_files(query, filtered_files)
    if not top_files:
        print("⚠️ No vectors found within the filtered files.")
        return

    # ✅ Step 3: Retrieve text & metadata directly from FAISS (NO NEED for get_text_from_faiss_pkl)
    embeddings = OpenAIEmbeddings()
    retrieved_data = await get_faiss_vectors(top_files, embeddings.embed_query(query))

    if not retrieved_data:
        print("⚠️ No relevant text found in FAISS.")
        return

    # ✅ Print retrieved results in a structured format
    print("\n📌 Retrieved Results:")
    await debug(json.dumps(retrieved_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
