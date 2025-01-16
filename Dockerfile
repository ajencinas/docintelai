# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Add environment variable
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on (Streamlit default port is 8501)
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "src/AgentFrontend.py", "--server.port=8501", "--server.address=0.0.0.0"]

