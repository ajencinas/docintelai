"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Box,
  TextField,
  Button,
  Typography,
  List,
  ListItem,
  ListItemText,
  Paper,
  Drawer,
  IconButton,
  Divider,
  Tooltip,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import EditIcon from "@mui/icons-material/Edit";
import DeleteIcon from "@mui/icons-material/Delete";
import AddIcon from "@mui/icons-material/Add";

export default function Chatbot() {
  const [authToken, setAuthToken] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [chats, setChats] = useState([]);
  const [currentChat, setCurrentChat] = useState(null);
  const [user, setUser] = useState(null); // Store user info
  const [isSending, setIsSending] = useState(false);
  const router = useRouter();
  const API_BASE_URL =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

  useEffect(() => {
    const fetchUserDetails = async () => {
      try {
        const response = await fetch("/api/authenticate", {
          method: "GET",
          credentials: "include", // Include HTTP-only cookies
        });

        if (!response.ok) {
          throw new Error("Failed to fetch user details");
        }

        const data = await response.json();

        console.log("User Details:", data);
        console.log("🔍 API Base URL:", process.env.NEXT_PUBLIC_API_BASE_URL);

        if (data.user) {
          setAuthToken(data.token); // ✅ Store JWT in memory
          setUser(data.user); // Set user info from the backend response
        } else {
          router.push("/"); // Redirect to login if no user is found
        }
      } catch (error) {
        console.error("Error fetching user details:", error);
        router.push("/"); // Redirect to login if there's an error
      }
    };

    fetchUserDetails();
  }, [router]);

  // Call this in useEffect to load all chats when the user logs in
  useEffect(() => {
    if (user) {
      fetchUserChats(user.username);
    }
  }, [user]);

  useEffect(() => {
    const fetchChatHistory = async () => {
      if (user && currentChat) {
        try {
          console.log("user_id:", String(user?.username));
          console.log("session_id:", String(currentChat?.id));

          if (!authToken) {
            console.error("No auth token available");
            return;
          }

          const response = await fetch(`${API_BASE_URL}/get_chat/`, {
            method: "POST",
            headers: {
              Authorization: `Bearer ${authToken}`, // ✅ Attach JWT in header
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              user_id: String(user.username),
              session_id: String(currentChat.id),
            }),
          });
          console.log("This is the response from API :", response);
          if (response.ok) {
            const data = await response.json();
            console.log(
              "Retrieved Messages:",
              data.chat_history?.messages || []
            );
            const transformedMessages = await transformChatResponse(data);
            setMessages(transformedMessages);
          } else if (response.status === 404) {
            console.log("No chat history found.");
            setMessages([]);
          } else {
            console.error("Failed to fetch chat history");
          }
        } catch (error) {
          console.error("Error fetching chat history:", error);
        }
      }
    };
    fetchChatHistory();
  }, [currentChat]);

  // useEffect(() => {
  // console.log("Messages state updated:", messages);
  // }, [messages]);

  // useEffect(() => {
  //  if (Array.isArray(chats) && chats.length === 0) {
  //    handleAddNewChat(); // Ensures a new chat is created
  //  }
  // }, [chats]);

  async function transformChatResponse(response) {
    if (!response || response.status !== "success" || !response.chat_history) {
      console.error("Invalid chat response format");
      return [];
    }

    return response.chat_history?.messages?.map((msg) => ({
      sender: msg.sender === "me" || msg.sender === "user" ? "user" : "bot",
      text: msg.message,
    }));
  }

  async function fetchUserChats(user_id) {
    try {
      console.log("Fetch user chats");
      const response = await fetch(`${API_BASE_URL}/get_user_chats/`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`, // ✅ Attach JWT in header
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ user_id: String(user_id) }),
      });

      if (!response.ok) {
        console.warn("Failed to fetch user chats, keeping previous state.");
        createFirstChat(); // ✅ Create a new chat if none exist}
        return;
      }

      // ✅ Shows chat history in console - debug purposes
      const data = await response.json();
      console.log("User Chats:", data.chats);
      console.log("Number of chats:", data.chats.length);

      // ✅ Sets the chat history
      if (data.chats.length == 0) {
        handleAddNewChat(); // ✅ Create a new chat if none exist}
      } else if (data.chats.length > 0) {
        console.log("Setting chats");
        setChats(
          data.chats?.map((chat) => ({
            id: chat.session_id,
            name: chat.chat_name || "Untitled Chat",
          }))
        );
        // ✅ Select the last chat in the array
        const lastChat = data.chats[data.chats.length - 1];
        if (lastChat) {
          setCurrentChat({
            id: lastChat.session_id,
            name: lastChat.chat_name || "Untitled Chat",
          });
        }
      }
    } catch (error) {
      console.error("Error fetching user chats:", error);
      setChats([]);
    }
  }

  async function saveChatMessage(
    user_id,
    session_id,
    chat_name,
    sender,
    message
  ) {
    try {
      const response = await fetch(`${API_BASE_URL}/save_chat/`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`, // ✅ Attach JWT in header
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: String(user_id),
          session_id: String(session_id),
          chat_name: String(chat_name),
          sender: String(sender),
          message: String(message),
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to send message");
      }

      const data = await response.json();
      console.log("Response from API:", data);
    } catch (error) {
      console.error("Error handling chat:", error);
    }
  }

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: "user", text: input.trim() };

    // ✅ Instantly update UI with both user message and bot response
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    // ✅ Disable input and button to prevent multiple queries
    setInput("");
    setIsSending(true); // Disable button

    // ✅ Show "thinking..." message in UI
    setMessages((prevMessages) => [
      ...prevMessages,
      { sender: "bot", text: "🤖 Thinking..." },
    ]);

    try {
      // ✅ Fetch bot response from FastAPI
      const botResponseText = "Dummy trial";
      // wait botQuery(
      //  input.trim(),
      //  user.username,
      //  currentChat.id
      // );

      // ✅ Remove "thinking..." message and add actual response
      setMessages((prevMessages) =>
        prevMessages
          .filter((msg) => msg.text !== "🤖 Thinking...")
          .concat({
            sender: "bot",
            text: botResponseText || "⚠️ Error retrieving response.",
          })
      );

      // ✅ Save messages to history
      await saveChatMessage(
        user.username,
        currentChat.id,
        currentChat.name,
        "user",
        input.trim()
      );
      await saveChatMessage(
        user.username,
        currentChat.id,
        currentChat.name,
        "bot",
        botResponseText
      );
      await fetchUserChats(user.username);
    } catch (error) {
      console.error("Error retrieving bot response:", error);

      // ✅ Show error message in chat
      setMessages((prevMessages) =>
        prevMessages
          .filter((msg) => msg.text !== "🤖 Thinking...")
          .concat({
            sender: "bot",
            text: "⚠️ An error occurred. Please try again.",
          })
      );
    } finally {
      // ✅ Re-enable input field and send button after response
      setIsSending(false);
    }
  };

  const botQuery = async (query, username, userid) => {
    console.log("Querying bot with:", query, username, userid);
    try {
      const response = await fetch(`${API_BASE_URL}/answer_query/`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`, // ✅ Attach JWT in header
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: String(username),
          session_id: String(userid),
          current_query: String(query),
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch bot response");
      }

      const jsonData = await response.json(); // Convert to JSON
      const responsebot = jsonData.response; // Extract the string
      // console.log("Responsebot:", responsebot);  // Log the string
      return responsebot; // Extract the string
    } catch (error) {
      console.error("Error retrieving bot response:", error);
    }
  };

  const handleSelectChat = async (chat) => {
    if (!chat || !chat.id) {
      console.error("Chat selection error: No valid chat ID found.");
      return;
    }

    setCurrentChat(chat);
  };

  const handleDeleteChat = (chatId) => {
    console.log("Starting chat deletion...");

    console.log("Deleting from session state chat with ID:", chatId);
    const chatToDelete = chats.find((chat) => chat.id === chatId);
    console.log("Chat to delete:", chatToDelete.messages);
    if (
      !chatToDelete ||
      (chatToDelete.messages && chatToDelete.messages.length === 0)
    ) {
      console.log("Chat is empty, not deleting.");
      return; // Prevent deletion if the chat is empty
    }

    setChats((prev) => prev.filter((chat) => chat.id !== chatId));
    if (currentChat?.id === chatId) {
      setCurrentChat(null);
      setMessages([]);
    }

    deleteChat(chatId);
    console.log("Fetching existing chats...");
    fetchUserChats(user.username);
  };

  const handleRenameChat = async (chatId) => {
    const newName = prompt("Enter new name for the chat:");
    if (newName) {
      setChats((prev) =>
        prev?.map((chat) =>
          chat.id === chatId ? { ...chat, name: newName } : chat
        )
      );

      await renameChat(chatId, newName);
    }
  };

  const createFirstChat = async () => {
    console.log("Create first chat...");

    // Ensure updatedChats is correctly defined from current state
    let updatedChats = chats ? [...chats] : [];

    const newChat = {
      id: Date.now(),
      name: "New Chat",
      messages: [],
    };

    await initializeChat(newChat.id); // ✅ Initialize chat in DB
    await fetchUserChats(user.username); // ✅ Fetch updated chat list from backend
  };

  const handleAddNewChat = async () => {
    console.log("Adding new chat...");

    // Ensure updatedChats is correctly defined from current state
    let updatedChats = chats ? [...chats] : [];

    // Prevent creating a new chat if the current chat is empty
    if (messages.length === 0 && updatedChats.length > 0) {
      console.log("Current chat is empty, not creating a new chat.");
      return; // Stop execution
    }

    const newChat = {
      id: Date.now(),
      name: "New Chat",
      messages: [],
    };

    await initializeChat(newChat.id); // ✅ Initialize chat in DB

    // Rename the current chat only if its name is still "New Chat"
    if (currentChat && currentChat.name === "New Chat") {
      const newChatName = `Chat-${currentChat.id}`;

      try {
        await renameChat(currentChat.id, newChatName); // ✅ Update DB

        // Update chat state locally after renaming
        updatedChats = updatedChats
          ? updatedChats.map((chat) =>
              chat.id === currentChat.id ? { ...chat, name: newChatName } : chat
            )
          : [];

        console.log(`Chat renamed successfully in DB: ${newChatName}`);
      } catch (error) {
        console.error(
          "Error renaming chat:",
          data?.detail || "Unknown error from server"
        );
      }
    }

    // ✅ Update state correctly
    updatedChats.push(newChat); // Add new chat
    setChats(updatedChats); // ✅ Update the state

    // Set the new chat as the current chat
    setCurrentChat(newChat);
    setMessages([]);
  };

  const initializeChat = async (sessionId) => {
    // No chat exists, create a new one with the provided session ID
    try {
      console.log("Initializing Chat...", sessionId);

      const createResponse = await fetch(`${API_BASE_URL}/create_chat/`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`, // ✅ Authenticate via headers
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: String(user.username),
          session_id: String(sessionId),
        }),
      });

      if (!createResponse.ok) {
        const errorMessage = await createResponse.text();
        console.error("Chat creation failed:", errorMessage);
        return; // Stop execution if the request failed
      }

      const received_data = await createResponse.json();
      console.log("First chat created successfully:", received_data);
    } catch (error) {
      console.error("Error initializing chat:", error);
    }
  };

  const renameChat = async (sessionId, newChatName) => {
    try {
      console.log("Renaming chat...", newChatName, sessionId, newChatName);
      const response = await fetch(`${API_BASE_URL}/rename_chat/`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`, // ✅ Attach JWT in header
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: String(user.username),
          session_id: String(sessionId),
          new_chat_name: String(newChatName),
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        console.error("Error renaming chat:", data.detail);
      } else {
        console.log("Chat renamed successfully:", data.message);
      }
    } catch (error) {
      console.error("Network or unexpected error during renaming:", error);
    }
  };

  const deleteChat = async (sessionId) => {
    const response = await fetch(`${API_BASE_URL}/delete_chat/`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${authToken}`, // ✅ Attach JWT in header
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: String(user.username),
        session_id: String(sessionId),
      }),
    });

    if (!response.ok) {
      const errorMessage = await response.text();
      console.error("Error deleting chat:", errorMessage);
      return;
    } else {
      // ✅ Fetch updated chat list from backend to ensure UI reflects changes
      await fetchUserChats(user.username);

      // ✅ Remove chat from local state
      setChats((prevChats) => {
        const updatedChats = prevChats.filter((chat) => chat.id !== sessionId);
        console.log("Updated Chats:", updatedChats);
        // ✅ If all chats are deleted, create a new empty chat
        // if (updatedChats.length === 0) {
        //  console.log("All chats deleted, creating a new chat.");
        //  handleAddNewChat()
        //  }

        // ✅ If the deleted chat was the current one, select another chat
        if (currentChat?.id === sessionId) {
          const newCurrentChat =
            updatedChats.length > 0
              ? updatedChats[updatedChats.length - 1]
              : null;
          setCurrentChat(newCurrentChat);
          setMessages(newCurrentChat ? [] : []); // Clear messages if no chat remains
        }

        return updatedChats;
      });

      const data = await response.json();
      console.log("Chat deleted successfully:", data.message);
    }
    return;
  };

  const handleLogout = async () => {
    try {
      await fetch("/api/logout", {
        method: "POST",
        credentials: "include",
      });
      setUser(null);
      router.push("/"); // Redirect to login page
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  return (
    <Box sx={{ display: "flex", height: "100vh" }}>
      <Drawer
        variant="permanent"
        sx={{
          width: 240,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: 240, boxSizing: "border-box" },
        }}
      >
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: 2,
          }}
        >
          <Typography variant="h6">Chats</Typography>
          <Tooltip title="Add New Chat">
            <IconButton size="small" onClick={handleAddNewChat}>
              <AddIcon />
            </IconButton>
          </Tooltip>
        </Box>
        <Divider />
        <List>
          {(chats || []).map((chat) => (
            <ListItem
              key={chat.id}
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                cursor: "pointer",
              }}
              onClick={() => handleSelectChat(chat)}
            >
              <ListItemText primary={chat.name || "Unnamed Chat"} />
              <Box>
                <IconButton
                  size="small"
                  onClick={() => handleRenameChat(chat.id)}
                >
                  <EditIcon fontSize="small" />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={() => handleDeleteChat(chat.id)}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Box>
            </ListItem>
          ))}
        </List>
      </Drawer>

      <Box
        sx={{
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          padding: 3,
        }}
      >
        <Paper
          elevation={3}
          sx={{ width: "100%", maxWidth: 1000, padding: 3, borderRadius: 2 }}
        >
          <Typography variant="h5" gutterBottom>
            {currentChat ? currentChat.name : "Chatbot"}
          </Typography>

          <List sx={{ maxHeight: 400, overflow: "auto", marginBottom: 2 }}>
            {messages?.map((message, index) => (
              <ListItem
                key={index}
                sx={{
                  justifyContent:
                    message.sender === "user" ? "flex-end" : "flex-start",
                }}
              >
                <ListItemText
                  primary={message.text}
                  sx={{
                    backgroundColor:
                      message.sender === "user" ? "#e3f2fd" : "#f0f0f0",
                    padding: 1,
                    borderRadius: 2,
                  }}
                />
              </ListItem>
            ))}
          </List>

          <Box sx={{ display: "flex", gap: 1 }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSend();
              }}
            />
            <Button
              variant="contained"
              color="primary"
              onClick={handleSend}
              endIcon={<SendIcon />}
              disabled={isSending} // This ensures the button is disabled when sending
            >
              Send
            </Button>
          </Box>
        </Paper>
      </Box>

      <Drawer
        anchor="right"
        variant="permanent"
        sx={{
          width: 240,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: 240, boxSizing: "border-box" },
        }}
      >
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: 2,
          }}
        >
          <Button variant="contained" color="info" onClick={handleLogout}>
            Logout
          </Button>
        </Box>
        {/* Right sidebar content goes here */}
      </Drawer>
    </Box>
  );
}
