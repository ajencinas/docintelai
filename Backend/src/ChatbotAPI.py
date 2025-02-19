from fastapi import FastAPI, HTTPException, Request, Depends
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import traceback
import asyncio
from ManageChats import save_chat_message, get_chat_history, ChatMessage, setup_database, get_user_chat_sessions, rename_chat_session, delete_chat_session
from ManageChats import create_chat_session  
from VectorQueryRev import improve_query_with_context, get_faiss_vectors, findrightfiles, process_response, embeddings
import jwt
import os

# Define lifespan event manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Starting up...")
    try:
        await setup_database()
        print("✅ Database setup complete.")
    except Exception as e:
        print(f"❌ Error during database setup: {e}")

    yield  # Application runs while this is active

    print("🛑 Shutting down...")  # Cleanup tasks can be added here

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify your frontend domain)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define request models
class QueryRequest(BaseModel):
    user_id: str
    session_id: str
    prev_interactions: list
    current_query: str

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    sender: str
    message: str
    chat_name: str 

class RenameChatRequest(BaseModel):
    user_id: str
    session_id: str
    new_chat_name: str

class DeleteChatRequest(BaseModel):
    user_id: str
    session_id: str

class ChatHistoryRequest(BaseModel):
    user_id: str
    session_id: str

class AnswerQuery(BaseModel):
    user_id: str
    session_id: str
    current_query: str

class UserChatsRequest(BaseModel):
    user_id: str


class CreateChatRequest(BaseModel):
    user_id: str
    session_id: str


# Load JWT secret key
JWT_SECRET = os.getenv("JWT_SECRET")

# ✅ JWT Verification Function
def verify_jwt(request: Request):
    """
    Extracts and validates the JWT token from Authorization header.
    """
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = auth_header.split(" ")[1]  # ✅ Extract token from "Bearer <token>"

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])  # ✅ Verify JWT
        print("✅ Decoded JWT Payload:", payload)  # ✅ Debugging
        return payload  # ✅ Return user data
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


HISTORY_LIMIT=5 # Limit for previous interactions

# API endpoint to produce a response
@app.post("/answer_query/")
async def answer_query(request: AnswerQuery, user_data: dict = Depends(verify_jwt)):
    print("Answering query")
    # print("Request user id:",request.user_id)
    # print("Request session id:",request.session_id)
    # ✅ Ensure the user making the request matches the JWT user
    if request.user_id != user_data.get("username"):
        raise HTTPException(status_code=403, detail="Unauthorized: User ID mismatch")
    
    try:
        # ✅ Fetch chat history for the user and session
        chat_history = await get_chat_history(request.user_id, request.session_id)
        # print("Retrieved messages:", messages)  # Debugging output
        
        # ✅ Extract last `HISTORY_LIMIT` messages (questions & answers)
        prev_interactions = []
        waiting_for_response = None  # Store last user message
        if chat_history and "messages" in chat_history:
            messages = chat_history["messages"][-HISTORY_LIMIT:]  # Get last `HISTORY_LIMIT` messages
            print("Retrieved messages:", messages)  # Debugging output

        # Skip initial bot messages at the start
        index = 0
        while index < len(messages) and messages[index]["sender"] == "bot":
            index += 1  # Move past any leading bot messages

        # Process the remaining messages
        for msg in messages[index:]:  
            if msg["sender"] == "user":
                waiting_for_response = msg["message"]  # Store user's message
            elif msg["sender"] == "bot" and waiting_for_response:
                prev_interactions.append({
                    "query": waiting_for_response,
                    "response": msg["message"]
                })
                waiting_for_response = None  # Reset for next pair
        
        # print("Previous interactions:", prev_interactions)  # Debugging

        # ✅ Call `improve_query_with_context` with filtered chat history
        improved_query = await improve_query_with_context(prev_interactions, request.current_query)
        print("Improved query:", improved_query)  # Debugging

        # ✅ Step 2: Find relevant files
        matching_files = await findrightfiles(improved_query)
        if not matching_files:
            print("⚠️ No relevant files found.")
            return {"response": "I couldn't find relevant documents for your query."}

        # ✅ Step 3: Retrieve relevant text from FAISS
        query_vector = embeddings.embed_query(improved_query)
        relevant_vectors = await get_faiss_vectors(matching_files, query_vector, top_k=5)

        if not relevant_vectors:
            print("⚠️ No relevant text found in FAISS.")
            return {"response": "I couldn't find relevant information in the documents."}

        # ✅ Step 4: Extract text from vectors
        relevant_texts = [vector["text"] for vector in relevant_vectors if "text" in vector]
        print(f"✅ Retrieved {len(relevant_texts)} text snippets.")

        # ✅ Step 5: Generate AI response using RAG
        final_response = await process_response(relevant_texts, improved_query, prev_interactions)

        return {"response": final_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#API endpoint to create a new chat session
@app.post("/create_chat/")
async def create_chat(request: CreateChatRequest, user_data: dict = Depends(verify_jwt)):
    """
    Creates a new chat session for the user with a given session_id.
    """
    print("Creating chat")
    if request.user_id != user_data.get("username"):
        raise HTTPException(status_code=403, detail="Unauthorized: User ID mismatch")

    try:
        print(f"Creating chat for user {request.user_id} with session {request.session_id}")
        chat_session = await create_chat_session(request.user_id, request.session_id)
        return {"status": "success", "chat": chat_session}
    except Exception as e:
        print("Error in create_chat_session:", str(e))  # ✅ Debugging step
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint to improve query
@app.post("/improve_query/")
async def improve_query(request: QueryRequest, user_data: dict = Depends(verify_jwt)):
    try:
        improved_query = await improve_query_with_context(request.prev_interactions, request.current_query)
        return {"improved_query": improved_query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to save a chat message
@app.post("/save_chat/")
async def save_chat(request: ChatRequest, user_data: dict = Depends(verify_jwt)):
    print(user_data.get("username"))
    # ✅ Ensure the user making the request matches the JWT user
    if request.user_id != user_data.get("username"):
        raise HTTPException(status_code=403, detail="Unauthorized: User ID mismatch")

    try:
        chat_message = ChatMessage(
            user_id=request.user_id,
            session_id=request.session_id,
            sender=request.sender,  
            message=request.message
        )
        print(request.user_id)
        await save_chat_message(chat_message, request.chat_name)
        return {"status": "success", "message": "Chat saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to retrieve chat history
@app.post("/get_chat/")
async def get_chat(request: ChatHistoryRequest, user_data: dict = Depends(verify_jwt)):
    try:
        print(user_data.get("username"))
        # ✅ Ensure the user making the request matches the JWT user
        if request.user_id != user_data.get("username"):
            raise HTTPException(status_code=403, detail="Unauthorized: User ID mismatch")

        print(f"Received user_id: '{request.user_id}' (Type: {type(request.user_id)})")
        print(f"Received session_id: '{request.session_id}' (Type: {type(request.session_id)})")

        chat_history = await get_chat_history(request.user_id, request.session_id)

        if chat_history:
            return {"status": "success", "chat_history": chat_history}
        else:
            raise HTTPException(status_code=404, detail="Chat history not found")
    
    except HTTPException as http_error:
        # If we already raised a 404, just propagate it
        raise http_error
    
    except Exception as e:
        print("Unexpected error:", traceback.format_exc())  # Logs full stack trace
        raise HTTPException(status_code=500, detail="Internal Server Error")

#Get all chats from a user
@app.post("/get_user_chats/")
async def get_user_chats(request: UserChatsRequest, user_data: dict = Depends(verify_jwt)):
    try:
        print(user_data.get("username"))
        # ✅ Ensure the user making the request matches the JWT user
        if request.user_id != user_data.get("username"):
            raise HTTPException(status_code=403, detail="Unauthorized: User ID mismatch")
        
        chat_sessions = await get_user_chat_sessions(request.user_id)

        if chat_sessions:
            return {"status": "success", "chats": chat_sessions}
        else:
            raise HTTPException(status_code=404, detail="No chats found for this user")
    
    except HTTPException as http_error:
        raise http_error
    
    except Exception as e:
        print("Unexpected error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/rename_chat/")
async def rename_chat(request: RenameChatRequest, user_data: dict = Depends(verify_jwt)):
    """
    Endpoint to rename a chat session.
    """
    print("Renaming chat",request.user_id,request.session_id,request.new_chat_name)
    if request.user_id != user_data.get("username"):
        raise HTTPException(status_code=403, detail="Unauthorized: User ID mismatch")

    try:
        response = await rename_chat_session(request.user_id, request.session_id, request.new_chat_name)
        
        if response["status"] == "error":
            raise HTTPException(status_code=404, detail=response["message"])
        
        return response

    except Exception as e:
        print("Unexpected error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post("/delete_chat/")
async def delete_chat(request: DeleteChatRequest, user_data: dict = Depends(verify_jwt)):
    """
    Endpoint to delete a chat session.
    """
    print("Delete request:",request.user_id,request.session_id)
    if request.user_id != user_data.get("username"):
        raise HTTPException(status_code=403, detail="Unauthorized: User ID mismatch")

    try:
        response = await delete_chat_session(request.user_id, request.session_id)

        if response["status"] == "error":
            raise HTTPException(status_code=404, detail=response["message"])

        return response

    except Exception as e:
        print("Unexpected error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run FastAPI server if this script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
