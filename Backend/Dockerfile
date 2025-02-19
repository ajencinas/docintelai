# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Set the Python module search path to include /app/src
ENV PYTHONPATH=/app/src

# Add environment variable
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the FastAPI app runs on
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "src.ChatbotAPI:app", "--host", "0.0.0.0", "--port", "8000"]
