import streamlit as st
import random
import threading
import http.server
import socketserver
import socket
from PIL import Image
from datetime import datetime
import os
import json 
from pathlib import Path
import pandas as pd
from VectorQuery import removefilefromdb, returnvectorstore, remove_documents_by_metadata, processnewfile, query_document, process_response, savevectorstore, storevectorlocally, improve_query_with_context, debug, prefilter_vector_store

# Fetch paths from environment variables or use defaults
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_FILE_PATH = PROJECT_ROOT / "data/vector"
CHAT_STORE_FILE_PATH = PROJECT_ROOT / "data/chats"
FILE_STORE_FILE_PATH = PROJECT_ROOT / "data/files"
IMAGE_FILE_PATH = PROJECT_ROOT / "data/images"
PDF_DIRECTORY = FILE_STORE_FILE_PATH

# Get the port from the environment variable or default to 8501
port = int(os.getenv("PORT", 8501))

# Fetch the OpenAI API key from the system environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("The OPENAI_API_KEY environment variable is not set in the system.")

def main():
      
    #Project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Session state to manage navigation
    if "screen" not in st.session_state:
        st.session_state.screen = "main"

    # Open a text file in write mode to clear its contents at the start
    with open("debug_log.txt", "w") as debug_file:
        debug_file.write("Starting debug log...\n")  # Optional: Write a starting message

    debug(f"IMAGE_FILE_PATH type: {type(IMAGE_FILE_PATH)}, value: {IMAGE_FILE_PATH}")

    # Render screens based on session state
    if st.session_state.screen == "main":
        main_screen()
    elif st.session_state.screen == "add_documents":
        add_documents_screen()
    elif st.session_state.screen == "query_documents":
        query_documents_screen()
    elif st.session_state.screen == "check_quality":
        check_quality_screen()

def start_pdf_server(directory, port=None):
    if port is None:
        port = find_free_port()
    handler = http.server.SimpleHTTPRequestHandler
    try:
        httpd = socketserver.TCPServer(("", port), handler)
        os.chdir(directory)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"Serving PDFs on port {port}")
    except OSError as e:
        st.error(f"Error starting PDF server on port {port}: {e}")
        raise

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def main_screen():
    
    image_path = IMAGE_FILE_PATH / "Header.png"    
    debug(f"Main screen loaded. Image path set to: {image_path}")
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning(f"Header image not found at: {image_path}")

    # Title of the tool
    st.title("Document Intelligence App")

    # Overview of functionality
    st.markdown(
        """
        Welcome to the Document Intelligence App. This tool allows you to:
        - **Add Documents**: Upload and process your documents.
        - **Query Documents**: Ask questions and retrieve information from your documents.
        - **Check Quality**: Evaluate the quality of document processing.
        """
    )

    # Main Menu Buttons with icons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📂 Manage Documents"):
            st.session_state.screen = "add_documents"
            st.rerun()
    with col2:
        if st.button("🔍 Query Documents"):
            st.session_state.screen = "query_documents"
            st.rerun()
    with col3:
        if st.button("✅ Check Quality"):
            st.session_state.screen = "check_quality"
            st.rerun()

def list_files_in_db():
    """Lists all files in the database."""
    db_file_path = Path(__file__).parent.parent / "data/filedb/filedb.json"
    if db_file_path.exists():
        with db_file_path.open("r") as db_file:
            return json.load(db_file)
    return []

def add_documents_screen():
    """Add and remove documents to the database"""
    st.title("Document manager")
    st.write("Use this screen to upload, remove and manage documents.")

    # List existing files
    st.subheader("Existing Files")
    files = list_files_in_db()

    # Search functionality
    search_query = st.text_input("Search files", placeholder="Search by name, description, or year")

    # Filter files based on search query
    if search_query:
        filtered_files = [
            f for f in files if
            search_query.lower() in f['file_name'].lower() or
            search_query.lower() in f['metadata_descr'].lower() or
            search_query.lower() in str(f['metadata_year']).lower() or
            search_query.lower() in str(f['metadata_type']).lower()
        ]
    else:
        filtered_files = files

    # Paginated display
    if filtered_files:
        page_size = 5
        total_pages = (len(filtered_files) - 1) // page_size + 1
        page_number = st.number_input("Page", min_value=1, max_value=total_pages, step=1)

        # Display the current page and total pages
        st.text(f"Page {page_number} out of {total_pages}")

        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size
        page_files = filtered_files[start_idx:end_idx]

        # Display files in a table
        for file in page_files:
            col1, col2, col3, col4, col5 = st.columns([3, 5, 2, 3, 1])  # Adjust column widths
            with col1:
                st.text(file['file_name'])
            with col2:
                st.text(file['metadata_descr'])
            with col3:
                st.text(file['metadata_year'])
            with col4:
                st.text(file['metadata_type'])
            with col5:
                if st.button("❌", key=f"remove_{file['file_name']}"):
                    removefilefromdb(file['file_name'])
                    vector_store = returnvectorstore(VECTOR_STORE_FILE_PATH)
                    remove_documents_by_metadata(vector_store, file['metadata_descr'])
                    savevectorstore(vector_store, VECTOR_STORE_FILE_PATH)
                    st.rerun()
    else:
        st.write("No files match the search criteria.")

    # Add new file
    st.subheader("Add New File")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    metadata_descr = st.text_input("Description")
    metadata_year = st.text_input("Year")
    metadata_type = st.text_input("Type")

    if st.button("Add File"):
        if uploaded_file and metadata_descr and metadata_year:
            # Save the file to directory
            save_dir = Path(FILE_STORE_FILE_PATH)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            processnewfile(file_path, uploaded_file.name, metadata_descr, metadata_year, metadata_type)
            st.success(f"File '{uploaded_file.name}' added successfully!")
            st.rerun()
        else:
            st.error("Please provide all the required information.")

    if st.button("Back to Main Menu"):
        st.session_state.screen = "main"
        st.rerun()

def save_chat_to_file(chat_name):
	"""Saves the given chat to a file."""
	chat_data = st.session_state.chat_history.get(chat_name, [])
	if chat_data:
		save_dir = Path(CHAT_STORE_FILE_PATH)
		save_dir.mkdir(parents=True, exist_ok=True)
		file_path = save_dir / f"{chat_name}.json"
		with open(file_path, "w") as f:
			json.dump(chat_data, f, indent=4)
		st.success(f"Chat '{chat_name}' saved to {file_path}")
	else:
		st.warning(f"No data to save for chat '{chat_name}'")


def query_documents_screen():
    """Query documents and manage chat history."""
    debug("Query documents screen loaded")
    st.title("Query Documents")
    st.write("Use this screen to ask questions about your documents.")

    # Load vector store
    vector_store = returnvectorstore(VECTOR_STORE_FILE_PATH)

    # Define chat storage path
    chat_storage_path = Path(CHAT_STORE_FILE_PATH)
    chat_storage_path.mkdir(parents=True, exist_ok=True)

    # Load existing chats into session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
        for chat_file in chat_storage_path.glob("*.json"):
            chat_name = chat_file.stem
            with chat_file.open("r") as f:
                st.session_state.chat_history[chat_name] = json.load(f)

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Default Chat"

    # Ensure current chat exists in chat history
    if st.session_state.current_chat not in st.session_state.chat_history:
        st.session_state.chat_history[st.session_state.current_chat] = []

    # Sidebar for chat management
    with st.sidebar:
        # Add Back to Main Menu button in the sidebar
        if st.button("🏠 Back to Main Menu"):
            st.session_state.screen = "main"
            st.rerun()

        # Display existing chats
        st.header("Chats")

        # Group chats by creation date
        grouped_chats = {}
        for chat_name, messages in st.session_state.chat_history.items():
            if messages:
                chat_date = messages[-1]["timestamp"].split(" ")[0]  # Extract date from last message
            else:
                chat_date = datetime.now().strftime("%Y-%m-%d")  # Use current date if no messages
            grouped_chats.setdefault(chat_date, []).append(chat_name)

        # Display chats grouped by date
        for chat_date, chat_names in sorted(grouped_chats.items()):
            st.subheader(chat_date)
            for chat_name in chat_names:
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    if st.button(chat_name, key=f"select_{chat_name}"):
                        st.session_state.current_chat = chat_name
                with col2:
                    if st.button("✏️", key=f"rename_{chat_name}"):
                        st.session_state.rename_popup_active = True
                        st.session_state.chat_to_rename = chat_name
                with col3:
                    if st.button("❌", key=f"delete_{chat_name}"):
                        st.session_state.chat_history.pop(chat_name)

                        # Remove the chat file
                        chat_file_path = Path(CHAT_STORE_FILE_PATH) / f"{chat_name}.json"
                        if chat_file_path.exists():
                            chat_file_path.unlink()
                            st.success(f"Deleted chat file for '{chat_name}'")
                        else:
                            st.warning(f"No file found for chat '{chat_name}'")

                        st.success(f"Deleted {chat_name}")
                        st.rerun()

        # Create new chat functionality
        if st.button("Create New Chat"):
            # Save the current chat to file
            current_chat_name = st.session_state.current_chat
            if current_chat_name in st.session_state.chat_history and st.session_state.chat_history[current_chat_name]:
                save_chat_to_file(current_chat_name)

            # Rename Default Chat if not renamed yet
            if current_chat_name == "Default Chat" and st.session_state.chat_history["Default Chat"]:
                first_query = st.session_state.chat_history["Default Chat"][0]["query"]
                auto_name = f"{first_query[:20]}..." if first_query else f"Chat {random.randint(1, 100)}"  # Truncate to first 20 characters
                print(auto_name)
                st.session_state.chat_history[auto_name] = st.session_state.chat_history.pop("Default Chat")
                st.success(f"Renamed 'Default Chat' to '{auto_name}'")
                save_chat_to_file(auto_name)

            # Open a new Default Chat
            st.session_state.chat_history["Default Chat"] = []
            st.session_state.current_chat = "Default Chat"

            # Save new chat to file
            st.success("New 'Default Chat' created")
            st.rerun()

    # Handle rename popup
    if st.session_state.get("rename_popup_active", False):
        st.write("### Rename Chat")
        with st.form(key="rename_form"):
            new_name = st.text_input("New chat name", value=st.session_state.chat_to_rename)
            submit = st.form_submit_button("Rename")
            if submit:
                if new_name in st.session_state.chat_history:
                    st.error("Chat name already exists!")
                else:
                    st.session_state.chat_history[new_name] = st.session_state.chat_history.pop(st.session_state.chat_to_rename)
                    save_chat_to_file(new_name)
                    st.session_state.rename_popup_active = False
                    st.session_state.chat_to_rename = None
                    st.success(f"Chat renamed to '{new_name}'")
                    st.rerun()

    # Chat input and processing
    def process_query():
        query = st.session_state.query_input
        if query:
            with st.spinner("Fetching response..."):
                #Context improvement
                debug("Query: "+query+"\n\n\n")
                updated_query = improve_query_with_context(st.session_state.chat_history.get(st.session_state.current_chat, []),query)
                debug("Updated Query: "+updated_query+"\n\n\n")
                
                # Query 10 results for drafting the response
                filtered_vector=prefilter_vector_store(vector_store, updated_query)
                query_results = query_document(filtered_vector, updated_query, k=10, similarity_threshold=0.1)
                debug(query_results)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if isinstance(query_results, str):
                    response = query_results
                    st.session_state.chat_history[st.session_state.current_chat].insert(0, {
                        "query": query,
                        "response": "No relevant documents matched your query.",
                        "timestamp": timestamp
                        })
                    return

                if not query_results:
                    st.write("No relevant documents matched your query.")
                    return
                
                response = process_response(query_results, updated_query, st.session_state.chat_history[st.session_state.current_chat])
                
                #Initialize PDF server if not active
                if 'PDF_SERVER_PORT' not in st.session_state:
                    free_port=find_free_port()
                    st.session_state['PDF_SERVER_PORT'] = free_port
                    start_pdf_server(PDF_DIRECTORY, free_port)
                
                # Show the top 5 sources - append to current chat
                sources=[]
                for result in query_results:
                    source = result.metadata.get("file","Unknown")
                    page_number = result.metadata.get("page_number", "Unknown")
                    link = f"[Open PDF](http://localhost:{st.session_state['PDF_SERVER_PORT']}/{os.path.basename(source)}#page={page_number})"
                    sources.append(f"**{source}, Page: {page_number}**: {link}")

                # Append sources to the response for the chat history
                response_display = response + "\n\n**Sources:**\n\n" + "\n".join(sources)

                # Update chat history
                st.session_state.chat_history[st.session_state.current_chat].insert(0, {
                    "query": query,
                    "response": response_display,
                    "timestamp": timestamp
                })

                # Save chat to file after update
                save_chat_to_file(st.session_state.current_chat)

            # Clear the input field
            st.session_state.query_input = ""

    # Query input
    st.subheader("Your Input")
    st.text_input("Type your question here...", key="query_input", on_change=process_query)

    # Display chat history for the current chat
    st.subheader("Chat History")
    chat_container = st.container()
    current_chat_history = st.session_state.chat_history.get(st.session_state.current_chat, [])

    with chat_container:
        for entry in current_chat_history:
            st.markdown(f"**You ({entry['timestamp']}):** {entry['query']}")
            st.markdown(f"**Bot ({entry['timestamp']}):** {entry['response']}")
            st.divider()

def check_quality_screen():
    st.title("Check Quality")
    st.write("Use this screen to evaluate the quality of document processing.")
    if st.button("Back to Main Menu"):
        st.session_state.screen = "main"
        st.rerun()

if __name__ == "__main__":
    # Start Streamlit with the dynamic port
    os.system(f"streamlit run AgentFrontend.py --server.port={port} --server.address=0.0.0.0")
    main()