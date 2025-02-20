import pytest
import httpx
import jwt
import os
from dotenv import load_dotenv

# ✅ Load environment variables from the .env file in the parent directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(env_path)

# ✅ Read JWT_SECRET from .env
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise ValueError("❌ JWT_SECRET is not set in the .env file!")

# ✅ Function to create a valid JWT token for testing
def create_test_jwt(user_id="testuser"):
    payload = {"username": user_id}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

BASE_URL = "http://localhost:8000"

# ✅ Generate a test JWT token
JWT_TOKEN = create_test_jwt()

HEADERS = {
    "Authorization": f"Bearer {JWT_TOKEN}",
    "Content-Type": "application/json"
}

@pytest.mark.asyncio
async def test_answer_query():
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BASE_URL}/answer_query/",
            json={"user_id": "testuser", "session_id": "test_session", "current_query": "What is AI?"},
            headers=HEADERS
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_create_chat():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/create_chat/",
            json={"user_id": "testuser", "session_id": "test_session"},
            headers=HEADERS
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_improve_query():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/improve_query/",
            json={"user_id": "testuser", "session_id": "test_session", "prev_interactions": [], "current_query": "Hello"},
            headers=HEADERS
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_save_chat():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/save_chat/",
            json={"user_id": "testuser", "session_id": "test_session", "sender": "user", "message": "Hello", "chat_name": "Test Chat"},
            headers=HEADERS
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_get_chat():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/get_chat/",
            json={"user_id": "testuser", "session_id": "test_session"},
            headers=HEADERS
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_get_user_chats():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/get_user_chats/",
            json={"user_id": "testuser"},
            headers=HEADERS
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_rename_chat():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/rename_chat/",
            json={"user_id": "testuser", "session_id": "test_session", "new_chat_name": "New Chat Name"},
            headers=HEADERS
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_delete_chat():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/delete_chat/",
            json={"user_id": "testuser", "session_id": "test_session"},
            headers=HEADERS
        )
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main()
