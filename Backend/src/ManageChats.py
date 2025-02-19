import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# MongoDB connection settings
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "chatdb"
COLLECTION_NAME = "user_chats"

# Initialize MongoDB client
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Pydantic model for chat messages
class ChatMessage(BaseModel):
    user_id: str
    session_id: str
    sender: str
    message: str
    timestamp: str = datetime.now(timezone.utc).isoformat()

# Ensure that the database and collection exist
async def setup_database():
    """Ensure the database and collection are properly configured."""
    existing_collections = await db.list_collection_names()
    if COLLECTION_NAME not in existing_collections:
        await db.create_collection(COLLECTION_NAME)
        await collection.create_index("user_id", unique=False)  # Shard key
        await collection.create_index("session_id")  # Efficient session queries

#Create a new chat session
async def create_chat_session(user_id: str, session_id: str):
    """Create a new empty chat session for a user with a given session_id."""
    chat_data = {
        "user_id": user_id,
        "session_id": session_id,
        "chat_name": "New Chat",
        "messages": [],
        "last_updated": datetime.now(timezone.utc)
    }
    result = await collection.insert_one(chat_data)
    # Convert the ObjectId to string so it can be serialized by FastAPI.
    chat_data["_id"] = str(result.inserted_id)
    return chat_data

# Save a chat message to a chat
async def save_chat_message(chat: ChatMessage, chat_name: str = "Untitled Chat"):
    """Save a chat message in a user-specific document."""
    update_query = {
        "user_id": chat.user_id,
        "session_id": chat.session_id
    }
    update_data = {
        "$push": {
            "messages": {
                "sender": chat.sender,
                "message": chat.message,
                "timestamp": chat.timestamp
            }
        },
        "$set": {
            "last_updated": datetime.now(timezone.utc),
            "chat_name": chat_name  # Ensures chat_name is always set
        }
    }
    await collection.update_one(update_query, update_data, upsert=True)

async def get_chat_history(user_id: str, session_id: str) -> Dict[str, Any]:
    """Retrieve chat history for a given user and session."""
    print("Getting chat history...")
    chat_data = await collection.find_one({"user_id": user_id, "session_id": session_id}, {"_id": 0, "messages": 1})

    if not chat_data or "messages" not in chat_data:
        return {"messages": []}  # Return empty list if no history found

    return chat_data

async def get_user_chat_sessions(user_id: str) -> List[Dict[str, Any]]:
    """Retrieve all chat sessions for a given user."""
    chats = await collection.find(
        {"user_id": user_id}, 
        {"_id": 0, "session_id": 1, "chat_name": 1, "last_updated": 1}
    ).to_list(length=None)
    return chats


async def rename_chat_session(user_id: str, session_id: str, new_name: str):
    """
    Rename a chat session in the database.
    """
    update_result = await collection.update_one(
        {"user_id": user_id, "session_id": session_id},
        {"$set": {"chat_name": new_name, "last_updated": datetime.now(timezone.utc)}}
    )

    if update_result.matched_count == 0:
        return {"status": "error", "message": "Chat session not found"}
    if update_result.modified_count == 0:
        return {"status": "error", "message": "Chat name unchanged"}

    return {"status": "success", "message": "Chat renamed successfully"}

async def delete_chat_session(user_id: str, session_id: str):
    """
    Delete a chat session from the database.
    """
    delete_result = await collection.delete_one({"user_id": user_id, "session_id": session_id})

    if delete_result.deleted_count == 0:
        return {"status": "error", "message": "Chat session not found"}

    return {"status": "success", "message": "Chat deleted successfully"}

async def main():
    """Main function to verify chat storage and retrieval."""
    await setup_database()

    test_message = ChatMessage(
        user_id="user_002",
        session_id="session_1234",
        sender = "user",
        message="Hello, this is a test message!"
    )

    await save_chat_message(test_message)
    chat_history = await get_chat_history("user_001", "session_1234")

      # ✅ Explicit validation instead of prints/asserts
    if not chat_history:
        return {"status": "error", "message": "Chat history not retrieved!"}
    if "messages" not in chat_history:
        return {"status": "error", "message": "No messages found in chat history!"}
    if chat_history["messages"][-1]["message"] != "Hello, this is a test message!":
        return {"status": "error", "message": "Message mismatch!"}

    return {"status": "success", "message": "Chat stored and retrieved successfully!", "chat": chat_history}

if __name__ == "__main__":
    result = asyncio.run(main())
    print(result)  # Output the final result in a structured way
