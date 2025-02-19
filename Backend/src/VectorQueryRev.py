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
from searchsupport import infer_needed_filters
from searchsupport import findrightfiles, get_faiss_vectors, embeddings


# Fetch paths from environment variables or use defaults
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fetch the OpenAI API key from the system environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("The OPENAI_API_KEY environment variable is not set in the system.")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Function to log debug messages
def debug(*args):
    """Logs debug messages to a file."""
    message = " ".join(map(str, args))
    with open("debug_log.txt", "a") as debug_file:  # Open the file in append mode
        debug_file.write(message + "\n")

async def improve_query_with_context(prev_interactions, current_query):
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

async def process_response(relevant_texts, query, chat_history=None, window_size=5):
    """Processes retrieved vectors and chat history to generate an answer using RAG."""
    try:
        debug("\n==== Processing Response ====")
        debug("Query:", query)
        debug(f"Retrieved {len(relevant_texts)} relevant text snippets.")
        
        # Combine retrieved texts into context
        combined_context = "\n\n".join(relevant_texts)
        debug("Combined Context:", combined_context[:500] + "...")  # Limit log siz

        # Prepare relevant chat history
        if chat_history:
            user_entries = [
                f"User: {entry['query']}\nBot: {entry['response']}"
                for entry in chat_history
                if 'query' in entry
            ]
            history_context = "\n".join(user_entries[-window_size:])
        else:
            history_context = ""

        # Build prompt for GPT-4
        messages = [
            {"role": "system", "content": "You are a helpful AI that answers questions based on retrieved context and prior chat history."},
            {"role": "user", "content": f"Past interactions:\n{history_context}\n\nRelevant Context:\n{combined_context}\n\nCurrent Question:\n{query}\n\nAnswer:"}
        ]

        debug("Constructed GPT-4 Prompt:", json.dumps(messages, indent=2)[:1000] + "...")  # Partial log to avoid large output

        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error while generating response: {e}")
        return "I'm unable to generate a response at the moment."

async def main():
    print("Query module started.")

    #Initialize chat history
    chathistory = []

    # Open a text file in write mode to clear its contents at the start
    with open("debug_log.txt", "w") as debug_file:
        debug_file.write("Starting debug log...\n")  # Optional: Write a starting message

    print("\nEnter your questions below. Type 'exit' to terminate.")

    while True:
        # Get user input
        query = input("\nYour question: ").strip()
        if query.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        # Improve the query with context
        debug("Original query:", query)
        debug("Chat history",chathistory)
        query = await improve_query_with_context(chathistory, query)
        print("The improved query with the context in your chat history is: ",query)

        # 🔥 Find relevant files
        matching_files = await findrightfiles(query)
        print(f"✅ Found {len(matching_files)} relevant files.")
        
        if not matching_files:
            print("⚠️ No relevant files found. Try refining your query.")
            continue

        # 🔍 Convert query into vector
        query_vector = embeddings.embed_query(query)

        # 🔥 Retrieve top vectors from FAISS indexes
        relevant_vectors = await get_faiss_vectors(matching_files, query_vector, top_k=5)
        print(f"✅ Retrieved {len(relevant_vectors)} vectors.")

        if not relevant_vectors:
            print("⚠️ No relevant vectors found in the files.")
            continue

        # ✅ Extract texts from vector
        relevant_texts = [vector["text"] for vector in relevant_vectors if "text" in vector]
        print(f"✅ Retrieved {len(relevant_texts)} relevant text snippets.")

        # 📌 Generate final response using RAG
        final_response = await process_response(relevant_texts, query, chathistory)

        # Display the response
        print("\n🤖 AI Response:\n", final_response)

        # Update chat history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chathistory.insert(0, {
            "query": query,
            "response": final_response,
            "timestamp": timestamp
        })

        # Limit the size of chat history to avoid performance issues
        if len(chathistory) > 5:
            chathistory.pop()

if __name__ == "__main__":
    asyncio.run(main())  