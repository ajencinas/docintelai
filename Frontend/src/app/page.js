"use client";

import { useRouter } from "next/navigation"; // Import useRouter from next/navigation
import { useState } from "react";
import {
  Box,
  Button,
  Checkbox,
  FormControlLabel,
  TextField,
  Typography,
  Divider,
  InputAdornment,
  Menu,
  MenuItem,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "@mui/material";
import { AiOutlineUser, AiOutlineLock } from "react-icons/ai";
import PasswordField from "@/components/PasswordField";

export default function Home() {
  const router = useRouter(); // Initialize router

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);
  const [openCreateUser, setOpenCreateUser] = useState(false);
  const [newUsername, setNewUsername] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [popupMessage, setPopupMessage] = useState("");

  const handleLogin = async () => {
    console.log("Password before API call:", password); // Debugging
    setMessage("");
    setIsLoading(true);
    try {
      const response = await fetch("/api/authenticate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: username.trim(),
          password: password.trim(), // Ensure password is included
        }),
      });

      const result = await response.json();

      if (response.ok) {
        setMessage("Login successful!");
        router.push("/chatbot"); // Navigate to /chatbot after successful login
      } else {
        setMessage(result.message || "Invalid username or password.");
      }
    } catch (error) {
      setMessage("An error occurred. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleMenuClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleCreateNewUserOpen = () => {
    setOpenCreateUser(true);
    handleMenuClose();
  };

  const handleCreateNewUserClose = () => {
    setOpenCreateUser(false);
  };

  const handleCreateUser = async () => {
    setPopupMessage(""); // Clear any previous popup messages

    if (!newUsername || !newPassword || !newEmail) {
      setPopupMessage("All fields are required.");
      return;
    }

    try {
      const response = await fetch("/api/createUser", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: newUsername,
          password: newPassword,
          email: newEmail,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        setPopupMessage("User created successfully!");
        setOpenCreateUser(false); // Close the dialog
        setNewUsername("");
        setNewPassword("");
        setPassword("");
        setNewEmail("");
      } else {
        setPopupMessage(result.message || "Failed to create user.");
      }
    } catch (error) {
      setPopupMessage("An error occurred. Please try again later.");
    }
  };

  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      minHeight="100vh"
      bgcolor="grey.100"
    >
      <Box
        width="100%"
        maxWidth={400}
        p={3}
        bgcolor="white"
        borderRadius={2}
        boxShadow={3}
      >
        <Box textAlign="center" mb={3}>
          <img
            src="/logo.png"
            alt="Logo"
            style={{ width: 48, height: 48, marginBottom: 16 }}
          />
          <Typography variant="h5" fontWeight="bold" gutterBottom>
            Welcome to Document AI
          </Typography>
        </Box>
        <Box mb={2}>
          <TextField
            fullWidth
            variant="outlined"
            label="Username"
            placeholder="Enter your username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            slotProps={{
              input: {
                startAdornment: (
                  <InputAdornment position="start">
                    <AiOutlineUser />
                  </InputAdornment>
                ),
              },
            }}
          />
        </Box>
        <Box mb={2}>
          <PasswordField password={password} setPassword={setPassword} />
        </Box>
        <Button
          fullWidth
          variant="contained"
          color="primary"
          onClick={handleLogin}
          disabled={isLoading}
          sx={{ mt: 2, mb: 2 }}
        >
          {isLoading ? "Logging in..." : "LOG IN"}
        </Button>
        {message && (
          <Typography
            variant="body2"
            color="error"
            align="center"
            sx={{ mt: 2 }}
          >
            {message}
          </Typography>
        )}
        <Divider sx={{ my: 2 }}>OR</Divider>
        <Box mt={2} textAlign="center">
          <Button
            variant="text"
            onClick={handleMenuClick}
            sx={{ textTransform: "none" }}
          >
            More Options
          </Button>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
          >
            <MenuItem onClick={handleCreateNewUserOpen}>
              Create New User
            </MenuItem>
          </Menu>
        </Box>
      </Box>
      <Dialog
        open={openCreateUser}
        onClose={handleCreateNewUserClose}
        fullWidth
        maxWidth="sm"
      >
        <DialogTitle>Create New User</DialogTitle>
        <DialogContent>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 3 }}>
            <TextField
              fullWidth
              variant="outlined"
              label="New Username"
              placeholder="Enter new username"
              value={newUsername}
              onChange={(e) => setNewUsername(e.target.value)}
            />
            <TextField
              fullWidth
              variant="outlined"
              type="email"
              label="Email"
              placeholder="Enter email address"
              value={newEmail}
              onChange={(e) => setNewEmail(e.target.value)}
            />
            <PasswordField
              password={newPassword}
              setPassword={setNewPassword}
            />
          </Box>
          {popupMessage && (
            <Typography variant="body2" color="error" sx={{ mt: 2 }}>
              {popupMessage}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCreateNewUserClose}>Cancel</Button>
          <Button
            onClick={handleCreateUser}
            variant="contained"
            color="primary"
          >
            Create User
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
