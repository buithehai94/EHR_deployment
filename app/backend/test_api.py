from fastapi import FastAPI, HTTPException
import pandas as pd
import requests
import chromadb
from chromadb.config import Settings
from io import StringIO  # Import StringIO

# Initialize the FastAPI app
app = FastAPI()

# Define the live JSON URL
json_url = "https://raw.githubusercontent.com/buithehai1994/EHR/refs/heads/main/data/chunk_2.json"

# Initialize ChromaDB client
chroma_client = chromadb.Client(
    Settings(
        persist_directory="./chromadb",  # Directory to save the database
        chroma_db_impl="duckdb+parquet",  # Backend for persistence
    )
)

# Create or get a collection
collection = chroma_client.get_or_create_collection(name="vector_database")


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI App for Loading and Querying ChromaDB!"}


@app.get("/load_data/")
def load_all_data():
    """
    Fetch the JSON data from the live URL, load it into a pandas DataFrame,
    and add it to the ChromaDB vector database.
    """
    try:
        # Fetch the JSON data from the URL
        response = requests.get(json_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # Wrap the JSON string in StringIO
        json_text = response.text
        json_io = StringIO(json_text)

        # Load the JSON data into a pandas DataFrame
        data = pd.read_json(json_io)

        # Ensure the DataFrame is not empty
        if data.empty:
            raise HTTPException(status_code=400, detail="The JSON data is empty.")

        # Add data to ChromaDB
        for index, row in data.iterrows():
            collection.add(
                ids=[str(index)],  # Unique ID for each record
                embeddings=[row["explanation_embedding"]],
                metadatas=[{"user": row["user"], "explanation": row["explanation"]}],
            )

        return {"message": "Data successfully loaded into ChromaDB.", "records_loaded": len(data)}
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching the JSON data: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Error processing JSON data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/query/")
def query_vectors(query_vector: str, n_results: int = 3):
    """
    Query the ChromaDB collection for the most similar vectors to the provided query vector.
    """
    try:
        # Convert the query vector string to a list of floats
        query_vector = [float(x) for x in query_vector.split(",")]

        # Perform the query
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
        )

        # Return the results
        return {"query_results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during the query: {str(e)}")
