import bcrypt from "bcryptjs";
import yaml from "yaml";
import fs from "fs/promises";
import path from "path";

const CONFIG_PATH = path.resolve("./data/users/config.yaml");

// Helper function to load the YAML config file
const loadConfig = async () => {
  try {
    const stats = await fs.stat(CONFIG_PATH);
    if (!stats.isFile()) {
      throw new Error(`Config file not found at: ${CONFIG_PATH}`);
    }
    const file = await fs.readFile(CONFIG_PATH, "utf-8");
    return yaml.parse(file);
  } catch (error) {
    if (error.code === "ENOENT") {
      // Create a default config if the file doesn't exist
      const defaultConfig = { credentials: { usernames: {} } };
      await saveConfig(defaultConfig);
      return defaultConfig;
    }
    console.error(`Error loading config from ${CONFIG_PATH}:`, error);
    throw error;
  }
};

// Helper function to save the YAML config file
const saveConfig = async (config) => {
  try {
    const yamlData = yaml.stringify(config);
    await fs.writeFile(CONFIG_PATH, yamlData, "utf-8");
  } catch (error) {
    console.error("Error saving config:", error);
    throw error;
  }
};

export async function POST(req) {
  try {
    const { username, password, email } = await req.json();

    // Validate input
    if (!username || !password || !email) {
      return new Response(
        JSON.stringify({
          message: "Username, password, and email are required.",
        }),
        { status: 400 }
      );
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return new Response(
        JSON.stringify({ message: "Invalid email format." }),
        { status: 400 }
      );
    }

    // Load the current configuration
    const config = await loadConfig();

    // Check if the user already exists
    if (config.credentials?.usernames?.[username]) {
      return new Response(
        JSON.stringify({ message: "Username already exists." }),
        { status: 400 }
      );
    }

    // Check if the email already exists
    const emailExists = Object.values(config.credentials?.usernames || {}).some(
      (user) => user.email === email
    );
    if (emailExists) {
      return new Response(
        JSON.stringify({ message: "Email already exists." }),
        { status: 400 }
      );
    }

    // Hash the password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Add the new user to the configuration
    if (!config.credentials) {
      config.credentials = { usernames: {} };
    }
    config.credentials.usernames[username] = {
      password: hashedPassword,
      email,
    };

    // Save the updated configuration
    await saveConfig(config);

    return new Response(
      JSON.stringify({ message: "User created successfully." }),
      { status: 201 }
    );
  } catch (error) {
    console.error("Error creating user:", error);
    return new Response(
      JSON.stringify({ message: "An error occurred while creating the user." }),
      { status: 500 }
    );
  }
}
