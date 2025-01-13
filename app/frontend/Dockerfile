# Use Python 3.9-slim as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt from the root of the project to the working directory
COPY requirements.txt ./  

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend folder containing app.py into the container
COPY app/frontend/ ./frontend

# Expose the port the app runs on
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
