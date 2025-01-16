# Use Python 3.9-slim as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt from the root of the project to the working directory
COPY requirements.txt ./  

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory inside the app folder
COPY app ./app

# Expose the port the app runs on
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "app.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
