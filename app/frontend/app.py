import streamlit as st
import requests
import pandas as pd
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Fetch Gemini API key from environment variables
api_key= os.getenv("gemini_api")

# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Title for the Streamlit app
st.title('Patient Record Interaction App')

@st.cache_data
def load_data():
    """
    Load data from a FastAPI endpoint and cache it for reuse.

    Returns:
        pd.DataFrame: The loaded data as a Pandas DataFrame.
    """
    # Define the URL of the FastAPI endpoint
    api_url = "https://fastapi-backend-h0ki.onrender.com/load_data/"

    try:
        # Make a GET request to the FastAPI endpoint with a timeout
        response = requests.get(api_url, timeout=10)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json().get('data')

            if isinstance(data, list):  # Ensure data is a list
                # Convert the list of dictionaries to a DataFrame
                df = pd.DataFrame(data)
                return df
            else:
                st.error("Invalid data format: Expected a list of dictionaries.")
                return pd.DataFrame()  # Return an empty DataFrame
        else:
            st.error(f"Failed to fetch data. Status code: {response.status_code}")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()
    
df=load_data()

# Endpoint URL
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

@st.cache_resource
# Function to retrieve relevant data based on symptom embedding
def retrieve_relevant_data(symptom_embedding, df):
    similarities = cosine_similarity([symptom_embedding], df['symptom_embedding'].tolist())[0]
    top_indices = np.argsort(similarities)[-3:]  # Top 3 most similar entries
    return df.iloc[top_indices]

@st.cache_resource
# Function to interact with the API using the prompt template and embedding-based retrieval
def chat_with_embeddings(symptom_text, df):
    symptom_embedding = model.encode(symptom_text)
    retrieved_data = retrieve_relevant_data(symptom_embedding, df)

    formatted_prompt = f"Based on the following historical data, the similar patient has the simlar following problem: {retrieved_data['explanation'].tolist()} \
        and provide insights and treatment suggestions. \
        The full medical records and patterns are: {retrieved_data['user'].tolist()}."

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": formatted_prompt}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        response_json = response.json()
        candidates = response_json.get("candidates", [])
        if candidates:
            content_parts = candidates[0].get("content", {}).get("parts", [])
            if content_parts:
                result_texts = []
                for part in content_parts:
                    text = part.get("text", "")
                    formatted_text = text.replace("**", "")
                    result_texts.append(formatted_text)
                return "\n".join(result_texts)
            else:
                return "No content parts found."
        else:
            return "No candidates found in the response."
    else:
        return f"Error: {response.status_code}\n{response.text}"

# Streamlit app layout
st.write("### Patient Record Interaction App")
st.write("Input symptoms and get insights and treatment suggestions based on historical data.")

# Input section for symptoms
input_symptom = st.text_input("Enter a symptom:")

if input_symptom:
    result = chat_with_embeddings(input_symptom, df)
    st.write(result)
