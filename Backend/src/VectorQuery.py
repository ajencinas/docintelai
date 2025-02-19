import os
import openai
import faiss
import json
import webbrowser
import asyncio
from datetime import datetime
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever


# Fetch paths from environment variables or use defaults
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_FILE_PATH = PROJECT_ROOT / "data/vector"

# Fetch the OpenAI API key from the system environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("The OPENAI_API_KEY environment variable is not set in the system.")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

#Metadata information for filtering
metadata_field_info = [
    AttributeInfo(
        name="description",
        description="The company that filed the document",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the document was filed",
        type="integer",
    ),
    AttributeInfo(
        name="type",
        description="The type of filing, e.g., 'public filing', 'response to filing'",
        type="string",
    ),
]

# Function to log debug messages
def debug(*args):
    """Logs debug messages to a file."""
    message = " ".join(map(str, args))
    with open("debug_log.txt", "a") as debug_file:  # Open the file in append mode
        debug_file.write(message + "\n")

def main():
    print("Vectorization module is running.")
   
    #Open vector store
    vector_store = returnvectorstore(VECTOR_STORE_FILE_PATH)
    if not vector_store:
        print("Failed to load vector store. Exiting.")
        return

    #Initialize chat history
    chathistory = []

    print("\nEnter your questions below. Type 'exit' to terminate.")
    
    # Open a text file in write mode to clear its contents at the start
    with open("debug_log.txt", "w") as debug_file:
        debug_file.write("Starting debug log...\n")  # Optional: Write a starting message

    while True:
        # Get user input
        query = input("\nYour question: ").strip()
        if query.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        # Improve the query with context
        debug("Original query:", query)
        debug("Chat history",chathistory)
        query = improve_query_with_context(chathistory, query)
        debug("Improved query",query)

        #Decide if a filter is needed
        filters=infer_needed_filters(query, metadata_field_info)
        debug("\nInferred Filters:")
        debug(json.dumps(filters, indent=4))  # Pretty-print the filters for better readability

        #Print vector metadata
        print_vector_store_metadata(vector_store)
        
        #Pre-filter the vector store
        prefiltered_vector_store=prefilter_vector_store(vector_store, query, metadata_field_info)

        # Query the document
        resultoutput = query_document(prefiltered_vector_store, query)
        debug("Result outputs\n",resultoutput)
        if not resultoutput:
            print("No relevant results found.")
            continue
        
        # Process the response
        response = process_response(resultoutput, query, chathistory, window_size=5)
        print("\nResponse:")
        print(response)

        # Identify relevant queries
        # relevant_queries = get_relevant_queries(resultoutput, response)

        # Display relevant queries
        # print("\nRelevant Queries Used:")
        # if relevant_queries:
        #    for idx, item in enumerate(relevant_queries, start=1):
        #        print(f"Result {idx}:")
        #        print(f"Source File: {item['source']}")
        #        print(f"Page Number: {item['page_number']}")
        #        print()
        #else:
        #    print("No relevant queries were identified as contributing to the response.")
        
        # Update chat history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chathistory.insert(0, {
            "query": query,
            "response": response,
            "timestamp": timestamp
        })

        # Limit the size of chat history to avoid performance issues
        if len(chathistory) > 5:
            chathistory.pop()

    
    #chunks=chunkpdf("/home/alfonso-encinas/Python2024/DocIntelAI/data/files/Oncor-2024-1.pdf")
    #vector_store=vectorizechunks(chunks,"Oncor","2024","Public Fiing")
    #storevectorlocally(vector_store,VECTOR_STORE_FILE_PATH)
    #resultoutput=query_document(vector_store,"What are the Oncor resiliency metrics?")
    #filters=infer_needed_filters("What are the Oncor resiliency metrics?", metadata_field_info)
    #print(filters)
    #for result in resultoutput:
    #    source = result.metadata.get("file", "Unknown")
    #    page_number = result.metadata.get("page_number", "Unknown")
    #    print(source, " Page number:", page_number)
    #process_response(resultoutput,"What are the Oncor resiliency metrics?")


def processnewfile(file_path, file_name, metadata_descr, metadata_year, metadata_type):
    """Process a new file"""
    try:
        print(f"Adding document '{file_name}' to the database and vector store.")
        addfiletodb(file_name, metadata_descr, metadata_year, metadata_type)
        chunks=chunkpdf(file_path)
        vector_store=vectorizechunks(chunks,metadata_descr,metadata_year, metadata_type)
        storevectorlocally(vector_store,VECTOR_STORE_FILE_PATH)
    
    except Exception as e:
        print(f"An error occurred while processing the new file: {e}")


def addfiletodb(file_name, metadata_descr, metadata_year, metadata_type):
    """Adds a file to the database"""
    try:
        # Define the relative path to the JSON database
        db_file_path = Path(__file__).parent.parent / "data/filedb/filedb.json"
        
        # Ensure the parent directory for the JSON database exists
        db_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing database or initialize an empty list
        if db_file_path.exists():
            with db_file_path.open("r") as db_file:
                database = json.load(db_file)
        else:
            database = []

        # Check for duplicates
        if any(entry["file_name"] == file_name for entry in database):
            print(f"File '{file_name}' already exists in the database. Skipping addition.")
            return
        
        # Add the new entry to the database
        new_entry = {
            "file_name": file_name,
            "metadata_descr": metadata_descr,
            "metadata_year": metadata_year,
            "metadata_type": metadata_type
        }
        database.append(new_entry)

        # Save the updated database
        with db_file_path.open("w") as db_file:
            json.dump(database, db_file, indent=4)
        
        print(f"File '{file_name}' successfully added to the database.")

    except Exception as e:
        print(f"An error occurred while storing vector in database: {e}")
        return None
    

def removefilefromdb(file_name):
    """Removes a file from the db"""
    
    try:
        # Define the relative path to the JSON database
        db_file_path = Path(__file__).parent.parent / "data/filedb/filedb.json"
        
        # Check if the database file exists
        if not db_file_path.exists():
            print(f"The database file '{db_file_path}' does not exist.")
            return
        
        # Load the existing database
        with db_file_path.open("r") as db_file:
            database = json.load(db_file)

        # Find the entry to remove
        initial_count = len(database)
        database = [entry for entry in database if entry["file_name"] != file_name]
        
        # Check if any entries were removed
        if len(database) == initial_count:
            print(f"File '{file_name}' was not found in the database.")
        else:
            # Save the updated database
            with db_file_path.open("w") as db_file:
                json.dump(database, db_file, indent=4)
            print(f"File '{file_name}' successfully removed from the database.")

    except Exception as e:
        print(f"An error occurred while removing the file from the database: {e}")


def returnvectorstore(vector_directory):
    """Returns the vector store"""
    try:

        #Create embeddings 
        print("Creating embeddings and vector store")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        #Load existing vector store if it exists
        if os.path.exists(vector_directory):
            vector_store = FAISS.load_local(vector_directory, embeddings, allow_dangerous_deserialization=True)
            print("Vector store found. Number vectors: ",vector_store.index.ntotal)
            return vector_store
        else:
            print("No existing vector store found.")

    except Exception as e:
        print(f"An error occurred while retrieving vector store: {e}")
        return None


def savevectorstore(vector_store,vector_directory):
    """Stores the vector store locally in a file"""
    try:
        #Create embeddings 
        print("Creating embeddings and vector store")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        #Save vector store locally
        vector_store.save_local(vector_directory)
        
        #Debugging - Print vectors
        #num_vectors = vector_store.index.ntotal
        #vectors = [vector_store.index.reconstruct(i) for i in max(range(num_vectors),5)]
        #print(vectors)

    except Exception as e:
        print(f"An error occurred while storing vector store: {e}")
        return None

def storevectorlocally(vector_store,vector_directory):
    """Adds vector store to the local vector database"""
    try:
        #Create embeddings 
        print("Creating embeddings and vector store")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        #Load existing vector store if it exists
        if os.path.exists(vector_directory):
            existing_vector_store = FAISS.load_local(vector_directory, embeddings, allow_dangerous_deserialization=True)
            vector_store.merge_from(existing_vector_store)
            print("Vector store found. Number vectors: ",vector_store.index.ntotal)
        else:
            print("No existing vector store found. Storing a new one.")

        #Save vector store locally
        vector_store.save_local(vector_directory)
        
        #Debugging - Print vectors
        #num_vectors = vector_store.index.ntotal
        #vectors = [vector_store.index.reconstruct(i) for i in range(num_vectors)]
        #print(vectors)

    except Exception as e:
        print(f"An error occurred while adding to vector store: {e}")
        return None
    
def remove_documents_by_metadata(vector_store, metadata_descr):
    """Removes documents from the vector store based on a given metadata description."""

    try:
        
        # Access the private _dict attribute of the docstore
        docstore_dict = vector_store.docstore._dict
        
        # List of document IDs to remove
        ids_to_remove = []

        # Iterate through the docstore to find matching documents
        for doc_id, document in docstore_dict.items():
            if document.metadata.get("description") == metadata_descr:
                ids_to_remove.append(doc_id)

        if not ids_to_remove:
            print(f"No documents found with metadata description: '{metadata_descr}'")
            return

        # Remove documents from the vector store
        vector_store.delete(ids_to_remove)

        print(f"Removed {len(ids_to_remove)} documents with metadata description: '{metadata_descr}'")

    except Exception as e:
        print(f"An error occurred while removing documents: {e}")

def print_vector_store_metadata(vector_store):
    """
    Prints the metadata of all documents in the vector store.
    """
    try:
        docstore_dict = vector_store.docstore._dict

        debug(f"Total documents in vector store: {len(docstore_dict)}")
        if not docstore_dict:
            debug("The vector store is empty.")
            return

        for idx, (doc_id, document) in enumerate(docstore_dict.items(), start=1):
            debug(f"Document {idx}:")
            debug(f"  ID: {doc_id}")
            debug(f"  Metadata: {document.metadata}")
            debug(f"  Content (first 100 chars): {document.page_content[:100]}...")
            debug("-" * 50)

    except Exception as e:
        print(f"An error occurred while printing metadata: {e}")

def vectorizechunks(chunks_with_meta, metadata_descr, metadata_year, metadata_type):
    """Vectorize chunks and metadata and returns a vector store"""
    try:
        #Add metadata
        base_metadata={"description":metadata_descr,"year":metadata_year, "type":metadata_type}

        # Create Document objects for each chunk, combining chunk metadata with general metadata. Metadata is also added to content for better filtering
        documents = [
            Document(
                page_content=chunk_data["chunk"]+"Metadata company: "+metadata_descr+" Metadata year: "+metadata_year+" Metadata type: "+metadata_type,
                metadata={**base_metadata, "file": chunk_data["file"], "page_number": chunk_data["page_number"]}
            )
            for chunk_data in chunks_with_meta
        ]

        #Debugging: Print length of Documents and example
        print(f"Total documents created: {len(documents)}")
        print(f"Example document content: {documents[100].page_content[:200]}")
        print(f"Example document metadata: {documents[100].metadata}")

        #Create embeddings 
        print("Creating embeddings and vector store")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("dummy")))

       #Vectorize and add documents and metadata
        vector_store = FAISS(embedding_function=OpenAIEmbeddings(),index=faiss_index,docstore= InMemoryDocstore(),index_to_docstore_id={})
        vector_store.add_documents(documents)

        #Debugging: Print vector
        doc_id =  vector_store.index_to_docstore_id[100]
        document = vector_store.docstore.search(doc_id)
        print("Content of the first vector:")
        print(document.page_content[:100])
        print("\nMetadata of the first vector:")
        print(document.metadata)
        
        print(f"Document vectorized successfully!")
     
        return vector_store

    except Exception as e:
        print(f"An error occurred while vectorizing the document: {e}")
        return None


def chunkpdf(file_path):
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
                          
        if not full_text.strip():
            raise ValueError("No text found in the PDF file.")
    
        # Debugging: Print the length of the extracted text
        print(f"Total text extracted (length): {len(full_text)}")
        
        #Split the text for processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, separators=[".","\n"])
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
                "file": file_path,
                "page_number": page_number
            })

        # Debugging: Print how many chunks were created
        #print(f"Total chunks created: {len(chunks)}")
        #if chunks:
        #    print(f"Example chunk: {chunks_with_meta[100]['chunk'][:200]}... (Page {chunks_with_meta[100]['page_number']})")
            # Use Firefox
        #    url=f"file://{file_path}#page={chunks_with_meta[100]['page_number']}"
        #    print(url)
        #    browser = webbrowser.get("firefox")
        #    browser.open(url)

        return chunks_with_meta

    except Exception as e:
        print(f"An error occurred while chunking the document: {e}")
        return None
    

def infer_needed_filters(query, metadata_field_info):
    """
    Use an LLM to infer which metadata filters are needed based on the query.
    Restrict the choices to the values present in the JSON database.
    Returns a dictionary of filter names and values if applicable.
    """

    # Define the relative path to the JSON database
    db_file_path = Path(__file__).parent.parent / "data/filedb/filedb.json"
        
    # Ensure the parent directory for the JSON database exists
    db_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing database or initialize an empty list
    if db_file_path.exists():
        with db_file_path.open("r") as db_file:
            database = json.load(db_file)
    else:
        database = []

    # Extract unique values for each metadata field
    valid_values = {
        "description": {entry["metadata_descr"] for entry in database},
        "year": {entry["metadata_year"] for entry in database},
        "type": {entry["metadata_type"] for entry in database},
    }

    # Format the valid values for inclusion in the prompt
    allowed_values_prompt = (
        "Allowed values for filtering:\n"
        f"- description: {', '.join(valid_values['description'])}\n"
        f"- year: {', '.join(valid_values['year'])}\n"
        f"- type: {', '.join(valid_values['type'])}\n\n"
    )

    # Build the prompt to guide the LLM
    system_prompt = (
        "You are an assistant that determines which metadata fields are relevant "
        "to a user's query. Based on the query, identify if filtering by any metadata fields "
        "is required and return the field names and values as JSON. "
        "Only use the following allowed values for filtering:\n\n"
        + allowed_values_prompt
        + "Metadata fields are defined as follows:\n\n"
        + "\n".join(
            f"- {field.name}: {field.description} (type: {field.type})"
            for field in metadata_field_info
        )
        + "\nIf no metadata filtering is needed, return an empty JSON object."
    )
    
    user_prompt = f"Query: {query}\n\nWhich metadata filters should apply? Just return the JSON ready to use in the query."

    response = openai.chat.completions.create(
        model="gpt-4",  # or "gpt-4" if available "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1000,
        temperature=0)
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    debug("We will apply these filters in the vector database:"+response.choices[0].message.content.strip())

    # Parse the response and return the filter dictionary
    filters = json.loads(response.choices[0].message.content.strip())
    return filters

def improve_query_with_context(prev_interactions, current_query):
    """
    Evaluates if additional context is needed for the current query and updates it if necessary.
    
    Parameters:
        prev_interactions (list): A list of dictionaries, each containing 'query' and 'response'.
                                  Example: [{'query': '...', 'response': '...'}, ...]
        current_query (str): The current query being processed.
        
    Returns:
        str: The updated query, integrating additional context if needed.
    """
    try:
        # Construct the system prompt
        system_prompt = (
            "You are an assistant that evaluates if additional context from previous questions and answers "
            "should be added to a query to improve its clarity or completeness. Please rewrite the query "
            "to include the relevant context. If no additional context is necessary, return the query unchanged."
            "Keep your output to a better query that can be input to the vector search. Do not include any other text."
             )
        
        # Build the user prompt dynamically
        interaction_details = "\n\n".join(
            f"Interaction {i + 1}:\nQuestion: {interaction['query']}\nAnswer: {interaction['response']}"
            for i, interaction in enumerate(prev_interactions)
        )
        user_prompt = (
            f"{interaction_details}\n\n"
            f"Current Query:\n{current_query}\n\n"
            "Return the updated query that includes additional context if necessary, or the original query if no changes are needed."
        )

        # Call the OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0,
        )

        # Extract and return the updated query
        updated_query = response.choices[0].message.content.strip()
        return updated_query

    except Exception as e:
        print(f"An error occurred while evaluating and updating the query: {e}")
        return current_query

def prefilter_vector_store(vector_store, query, metadata_field_info=metadata_field_info):
    """
    Prefilters the vector store based on the query using the relevant filters 
    inferred from infer_needed_filters and returns the prefiltered vector store.
    """
    try:
        # Infer the relevant filters from the query
        filters = infer_needed_filters(query, metadata_field_info)
        if not filters:
            print("No filters inferred. Returning the original vector store.")
            return vector_store
        
        # Debug: Print the inferred filters
        debug(f"Inferred Filters: {filters}")

        # Access the private _dict attribute of the docstore
        docstore_dict = vector_store.docstore._dict

        # Collect document IDs to remove
        ids_to_remove = []
        for doc_id, document in docstore_dict.items():
            # Check if the document matches the filters
            if not all(document.metadata.get(key) == value for key, value in filters.items()):
                ids_to_remove.append(doc_id)

        # Remove non-matching documents
        if ids_to_remove:
            vector_store.delete(ids_to_remove)
            print(f"Removed {len(ids_to_remove)} documents that did not match the filters.")

        return vector_store

    except Exception as e:
        debug(f"An error occurred while prefiltering the vector store: {e}")
        return vector_store

def query_document(vector_store, query, k=10, similarity_threshold=0.2):
    """Queries the vector store for a given question."""
    try:
        #Create retriever
        llm = ChatOpenAI(temperature=0)
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"score_threshold": similarity_threshold, "k":k}
            )   
        results = retriever.invoke(query)

        if not results:
            print("No results in database.")
            return 

        #Debug - print results
        print("\nNumber of Query Results:", len(results))
        debug("\nQuery Results:")
        
        for idx, result in enumerate(results, start=1):
            debug(f"Result {idx}: {result.page_content[0:10000]}")
            debug(f"Source: {result.metadata.get('file')}, Page: {result.metadata.get('page_number')}")
    
        return results

    except Exception as e:
        print(f"An error occurred while querying the document: {e}")

def get_relevant_queries(query_return, process_response):
    """
    Use OpenAI to identify which query results are relevant to the process_response.

    Parameters:
        query_return (list): List of results from the query_document function.
        process_response (str): The final response generated by the process_response function.

    Returns:
        list: Relevant query results with metadata for the source file and page.
    """
    relevant_queries = []

    for result in query_return:
        # Prepare the prompt for OpenAI
        
        prompt = (
        "You are an AI assistant evaluating the relevance of a retrieved document excerpt to a final response. "
        "A document excerpt is relevant only if it directly contributes specific details, explanations, or critical context to the response. "
        "If the excerpt is not clearly aligned with the response, mark it as 'Not Relevant'. Provide an explanation for your evaluation.\n\n"
        f"Document Excerpt:\n{result.page_content}\n\n"
        f"Generated Response:\n{process_response}\n\n"
        "Is this excerpt relevant to the response? Reply with 'Relevant' or 'Not Relevant' and provide a brief explanation."
        )

        try:
            # Call OpenAI API to evaluate relevance
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0
            )
            debug(response)

            # Parse the response for relevance
            relevance = response.choices[0].message.content.strip()
            if "Not Relevant" not in relevance:
                relevant_queries.append({
                    "source": result.metadata.get("file", "Unknown"),
                    "page_number": result.metadata.get("page_number", "Unknown"),
                    "content": result.page_content.strip(),
                    "explanation": relevance
                })
                debug("Selected as relevant")

        except Exception as e:
            print(f"An error occurred while evaluating relevance: {e}")
            continue

    return relevant_queries


def process_response(query_return,query,chat_history=None, window_size=5):
    """Process a response through ChatGPT to provide full answer"""
    try:
        results=query_return
        
        #Get all query returns
        retrieved_texts = []
        for idx, result in enumerate(results, start=1):
            retrieved_texts.append(result.page_content)
        
        # Combine retrieved texts
        combined_context = "\n\n".join(retrieved_texts)
        
        # Filter chat history to include only user queries, maintaining their respective responses
        if chat_history:
            user_entries = [
                f"User: {entry['query']}\nBot: {entry['response']}"
                for entry in chat_history
                if 'query' in entry
            ]
            # Limit by window_size, counting only user queries
            history_context = "\n".join(user_entries[-window_size:])
        else:
            history_context = ""

        #Build message for chatgpt
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the following context and history to answer. Consider all possible options"},
            {"role": "user", "content": f"Past questions that you were asked and answers that were given:\n{history_context}\n\nContext:\n{combined_context}\n\nCurrent question that you need to answer:\n{query}\n\nAnswer:"}
        ]
        
        debug("\nChatGPT Message: ")
        debug(messages)

        response = openai.chat.completions.create(
            model="gpt-4o",  # or "gpt-4" if available "gpt-3.5-turbo"
            messages=messages,
            max_tokens=1000,
            temperature=0.7)
        
        #Debug - Print the drafted answer
        #print("\nDrafted Answer from ChatGPT:")
        #print(response.choices[0].message.content.strip())

        return(response.choices[0].message.content.strip())

    except Exception as e:
        print(f"An error occurred while generating the response: {e}")

if __name__ == "__main__":
    main()