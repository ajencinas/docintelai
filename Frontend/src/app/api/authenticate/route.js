import bcrypt from "bcryptjs";
import yaml from "yaml";
import fs from "fs/promises";
import path from "path";
import jwt from "jsonwebtoken"; // Import jsonwebtoken
import dotenv from "dotenv";

dotenv.config();

const CONFIG_PATH = path.resolve("./data/users/config.yaml");
const JWT_SECRET = process.env.JWT_SECRET;
const TOKEN_EXPIRATION = "1h"; // Token validity duration

const loadConfig = async () => {
  try {
    const stats = await fs.stat(CONFIG_PATH);
    if (!stats.isFile()) {
      throw new Error("Config file not found.");
    }
    const file = await fs.readFile(CONFIG_PATH, "utf-8");
    return yaml.parse(file);
  } catch (error) {
    console.error("Error loading config:", error);
    throw error;
  }
};

export async function POST(req) {
  try {
    const { username, password } = await req.json();

    console.log("Received login payload:", { username, password });

    if (!username || !password) {
      return new Response(
        JSON.stringify({ message: "Username and password are required." }),
        { status: 400 }
      );
    }

    const config = await loadConfig();
    const user = config.credentials?.usernames?.[username];

    if (!user) {
      return new Response(JSON.stringify({ message: "User not found." }), {
        status: 404,
      });
    }

    const isValid = bcrypt.compareSync(password, user.password);
    if (isValid) {
      // Generate JWT token
      const token = jwt.sign(
        { username: username, name: user.name }, // Payload
        JWT_SECRET, // Secret key
        { expiresIn: TOKEN_EXPIRATION } // Options
      );

      // Set HttpOnly cookie
      const headers = new Headers();
      headers.append(
        "Set-Cookie",
        `token=${token}; HttpOnly; Secure; Path=/; Max-Age=${
          60 * 60
        }; SameSite=Strict`
      );
      console.log("Generated token:", token);

      return new Response(
        JSON.stringify({
          message: `Welcome ${user.name}!`,
          user: { username: username, name: user.name }, // Include user info for the client
        }),
        { status: 200, headers }
      );
    } else {
      return new Response(
        JSON.stringify({ message: "Incorrect username or password." }),
        { status: 401 }
      );
    }
  } catch (error) {
    console.error("Unexpected error:", error);
    return new Response(JSON.stringify({ message: "Internal Server Error" }), {
      status: 500,
    });
  }
}

export async function GET(req) {
  try {
    // Extract cookies from the request header
    const cookieHeader = req.headers.get("Cookie");
    if (!cookieHeader) {
      return new Response(JSON.stringify({ message: "No cookies found" }), {
        status: 401,
      });
    }

    // Parse cookies to find the token
    const cookies = Object.fromEntries(
      cookieHeader.split(";").map((cookie) => cookie.trim().split("="))
    );
    const token = cookies["token"];
    if (!token) {
      return new Response(
        JSON.stringify({ message: "Authentication token not found." }),
        { status: 401 }
      );
    }

    // Verify the token
    try {
      const decoded = jwt.verify(token, JWT_SECRET);
      return new Response(
        JSON.stringify({
          message: "User authenticated",
          user: { username: decoded.username, name: decoded.name },
          token: token,
        }),
        { status: 200 }
      );
    } catch (error) {
      return new Response(JSON.stringify({ message: "Invalid token" }), {
        status: 401,
      });
    }
  } catch (error) {
    console.error("Unexpected error:", error);
    return new Response(JSON.stringify({ message: "Internal Server Error" }), {
      status: 500,
    });
  }
}
