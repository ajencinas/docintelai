import os
import openai
import json
from pathlib import Path
import asyncio
import pickle
import webbrowser
from dotenv import load_dotenv
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient
from PyPDF2 import PdfReader
from motor.motor_asyncio import AsyncIOMotorClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from datetime import datetime, timezone
from scrapeupload import query_mongo
import shutil

# Fetch paths from environment variables or use defaults
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_FILE_PATH = PROJECT_ROOT / "tmp/vectorstore"

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
load_dotenv(dotenv_path)

# Azure connection setup
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = "ratecasefiles"
MONGO_URI = os.getenv("VCORE_URI")
DATABASE_NAME = "RateCaseDB"
COLLECTION_NAME = "FilesMetadata"
SAS_TOKEN = os.getenv("SAS_TOKEN")

# Azure clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

# Initialize MongoDB Client
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to extract text from a PDF
async def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    return " ".join(page.extract_text() for page in reader.pages if page.extract_text())

# Function to summarize text using OpenAI
async def generate_summary(text):
    print("Text lenght: ",len(text))
    openai.api_key = OPENAI_API_KEY
    
    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)  # Split plain text into chunks
    print("Number of chunks: ",len(chunks))

    # Initialize the OpenAI Chat Model for summarization
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=1000)

    # Use a simple summarization prompt
    prompt_template = """
    Summarize the following text:

    {text}

    Summary:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm | StrOutputParser()

    # Summarize each chunk and combine the results
    summaries = [chain.invoke({"text": chunk}) for chunk in chunks]

    # Combine all summaries into a single final summary
    final_summary = " ".join(summaries)
    final_summary = chain.invoke({"text": final_summary})

    # Create vector for search
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    summary_vector = embeddings.embed_query(final_summary)

    # Step 2: Infer Author Type from First 5 Chunks
    author_inference_prompt = """
    Based on the following text, infer who the author is. The author could be one of the following:
    - Utility - this is a filing by the utility itself - companies like Portland General, Oncor, etc.
    - Interveners - these are non-utility parties that file comments or other documents in a rate case
    - The commission - this is a filing by the public utility commission itself
    - The staff - this is a filing by the staff of the commission

    Text:
    {text}

    Who is the author that filed the file? Choose from the following - Utility, Interveners, Commission, Staff. Jusr return the author name ready to use as a tag - nothing more.
    """
    # Take the first 5 chunks (or fewer if there are less than 5)
    first_2_chunks = " ".join(chunks[:min(len(chunks), 2)])
    inference_prompt = PromptTemplate(template=author_inference_prompt, input_variables=["text"])
    inference_chain = inference_prompt | llm | StrOutputParser()
    author_inference = inference_chain.invoke({"text": first_2_chunks})

    return {
        "summary": final_summary,
        "summary_vector": summary_vector,
        "author_inference": author_inference
    }

def chunkpdf(file_path, blob_url):
    """Processes a PDF file and returns chunks and basic metadata - chunk number, file with path, page number."""
    try:
        #Read pdf
        pdf_reader = PdfReader(file_path)
                
        #Page index
        page_texts = []

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            page_texts.append(page_text)

        # Combine all text for chunking
        full_text = "".join(page_texts)
        print(f"🔍 Debug: Total extracted text length for {file_path}: {len(full_text)}")
                          
        if not full_text.strip():
            raise ValueError("No text found in the PDF file.")
    
        # Debugging: Print the length of the extracted text
        print(f"Total text extracted (length): {len(full_text)}")
        
        #Split the text for processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200, separators=[".","\n"])
        chunks = text_splitter.split_text(full_text)

        # Create merged page pairs for boundary checks
        merged_pages = [
            page_texts[i] + page_texts[i + 1] if i + 1 < len(page_texts) else page_texts[i]
            for i in range(len(page_texts))
        ]

       # Create chunks with metadata
        chunks_with_meta = []
        for chunk in chunks:
            # Find which page(s) the chunk belongs to
            page_number = next(
                (i + 1 for i, page_text in enumerate(page_texts) if chunk in page_text),
                None  # Default to None if not found (shouldn't happen if text is consistent)
            )

            if page_number is None:
                # Check in merged pages if not found in single pages
                page_number = next(
                    (i + 1 for i, merged_text in enumerate(merged_pages) if chunk in merged_text),
                    None  # Default to None if not found
                )

            # Append chunk with metadata
            chunks_with_meta.append({
                "chunk": chunk,
                "file": blob_url,
                "page_number": page_number
            })

        #Debugging: Print how many chunks were created
        print(f"Total chunks created: {len(chunks)}")
        #if chunks:
        #    print(f"Example chunk: {chunks_with_meta[1]['chunk'][:200]}... (Page {chunks_with_meta[1]['page_number']})")
        #   # Use Firefox
        #    url=f"file://{file_path}#page={chunks_with_meta[1]['page_number']}"
        #    print(url)
        #    browser = webbrowser.get("firefox")
        #    browser.open(url)

        return chunks_with_meta

    except Exception as e:
        print(f"An error occurred while chunking the document: {e}")
        return None

# Function to create FAISS vector database
async def create_faiss_vector_db(chunks_with_meta, file_name, vector_store_dir):
    """
    Creates a FAISS vector database from pre-chunked text and saves it as a single file.

    Parameters:
        chunks_with_meta (list): List of dictionaries containing chunked text and metadata.
        file_name (str): The name of the file being processed.
        vector_store_dir (str): The directory to save the vector database.

    Returns:
        str: Path to the saved FAISS vector database file.
    """
    try:
        
         # ✅ Convert chunked data into Document objects with metadata
        documents = [
            Document(
                page_content=chunk["chunk"],
                metadata={
                    "file_name": file_name,
                    "page_number": chunk["page_number"],
                    "blob_url": chunk["file"]  # ✅ Store blob URL for retrieval
                }
            )
            for chunk in chunks_with_meta
        ]

        # ✅ Create embeddings and FAISS vector store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_documents(documents, embeddings)

        # ✅ Define save path
        faiss_file_path = os.path.join(vector_store_dir, f"{file_name}_vector_store")

        # ✅ Save FAISS vector store locally
        vector_store.save_local(faiss_file_path)

        print(f"✅ FAISS vector store created for {file_name}, stored at {faiss_file_path}")
        return faiss_file_path

    except Exception as e:
        print(f"⚠️ Error in creating FAISS vector DB for {file_name}: {e}")
        return None

# Function to upload a file to Azure Blob Storage
async def upload_to_blob(directory_path, blob_prefix):
    """
    Uploads all files in a directory to Azure Blob Storage.

    Parameters:
        directory_path (str): The local directory path containing files to upload.
        blob_prefix (str): The prefix to use for blobs in Azure Blob Storage.

    Returns:
        list: A list of URLs for the uploaded files.
    """
    uploaded_urls = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            blob_name = f"{blob_prefix}/{file}"
            with open(local_file_path, "rb") as data:
                blob_client = container_client.get_blob_client(blob=blob_name)
                blob_client.upload_blob(data, overwrite=True)
            uploaded_urls.append(
                f"https://{blob_service_client.account_name}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{blob_name}"
            )
    return uploaded_urls

# Function to update metadata in MongoDB
async def update_existing_metadata(file_name, ratecase, vector_db_url, summary):
    """Finds an existing document and updates only vector_db_url & summary, keeping all other fields unchanged."""
    existing_doc = await collection.find_one({"file_name": file_name, "ratecase": ratecase})
    
    if not existing_doc:
        print(f"❌ No existing metadata found for {file_name}, skipping update.")
        return

    # Update only the required fields
    update_data = {
        "vector_db_url": vector_db_url,
        "summary": summary,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

    await collection.update_one(
        {"file_name": file_name, "ratecase": ratecase},
        {"$set": update_data}
    )
    print(f"✅ Updated metadata for {file_name}")


# Main function to process files
async def process_files():
    # Directory to store vector database files
    vector_store_dir = TMP_FILE_PATH
    os.makedirs(vector_store_dir, exist_ok=True)

    # List all blobs in the container
    blobs = container_client.list_blobs(name_starts_with="PGE2024/")

    for blob in blobs:
        file_name = os.path.basename(blob.name)
        local_file_path = TMP_FILE_PATH / file_name
        print("Processing file:", file_name)
        vector_db_path = None  # Initialize to ensure it exists in all cases

        # Download the file locally
        with open(local_file_path, "wb") as local_file:
            blob_data = container_client.get_blob_client(blob.name).download_blob()
            local_file.write(blob_data.readall())

        try:
            # Extract text and generate a summary
            text = await extract_text_from_pdf(local_file_path)
            print(f"✅ Text length: {len(text)}, Preview: '{text[:20]}'")
            summary = await generate_summary(text)

            # ✅ Ensure summary is not None
            print(f"✅ Summary length: {len(summary['summary'])}, Preview: '{summary['summary'][:20]}'")

            # Chunk file with metadata
            blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{blob.name}"
            url_with_sas = f"{blob_url}?{SAS_TOKEN}"

            # ✅ Step 1: Chunk the document (with metadata)
            chunks_with_meta = chunkpdf(local_file_path,blob_url)

            if not chunks_with_meta:
                print(f"⚠️ No chunks extracted for {file_name}. Skipping FAISS processing.")
                continue

            # ✅ Step 2: Store chunked data in FAISS
            vector_db_path = await create_faiss_vector_db(chunks_with_meta, file_name, vector_store_dir)

            if not vector_db_path:
                print(f"⚠️ Failed to create FAISS vector store for {file_name}")
                continue

            # ✅ Step 3: Upload FAISS vector store to Azure Blob Storage
            unique_filename = f"vectors/{file_name}_vector_store"
            vector_db_blob_url = await upload_to_blob(vector_db_path, unique_filename)

            # ✅ Step 4: Update MongoDB with vector DB & Blob URL
            await update_existing_metadata(file_name, "PGE2024", vector_db_blob_url, summary)

            print(f"Processed and updated metadata for file: {file_name}")

        except Exception as e:
            print(f"Failed to process file {file_name}: {e}")

        finally:
            # Clean up temporary files
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            if vector_db_path and os.path.exists(vector_db_path):
                shutil.rmtree(vector_db_path)  # Use rmtree to delete a directory


async def create_vector_index():
    """Creates a vector index on the summary_vector field using IVF in MongoDB."""
    index_name = "vectorSearchIndex"
    
    index_spec = {
        "createIndexes": COLLECTION_NAME,
        "indexes": [
            {
                "name": index_name,
                "key": {
                    "summary.summary_vector": "cosmosSearch"
                },
                "cosmosSearchOptions": {
                    "kind": "vector-ivf",  # Use IVF indexing method
                    "numLists": 100,       # Number of partitions (adjust based on dataset size)
                    "similarity": "COS",   # Cosine similarity
                    "dimensions": 1536     # OpenAI embedding vector size
                }
            }
        ]
    }

    # Execute index creation
    try:
        result = await db.command(index_spec)
        print(f"✅ Vector index '{index_name}' created successfully in '{COLLECTION_NAME}' using IVF.")
    except Exception as e:
        print(f"⚠️ Error creating vector index in '{COLLECTION_NAME}': {e}")


async def main():

    await collection.create_index("ratecase")
    await process_files()
    
    #Create vector index
    await create_vector_index()

    #Print results to table
    await query_mongo()

if __name__ == "__main__":
    asyncio.run(main())
