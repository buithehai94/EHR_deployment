from fastapi import FastAPI, HTTPException
import pandas as pd
import requests
from io import StringIO  # Import StringIO

# Initialize the FastAPI app
app = FastAPI()

# Define the live JSON URL
json_url = "https://raw.githubusercontent.com/buithehai1994/EHR/refs/heads/main/data/chunk_2.json"

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI App for Loading All JSON Data"}

@app.get("/load_data/")
def load_all_data():
    """
    Fetch the JSON data from the live URL, load it into a pandas DataFrame, and return all rows.
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

        # Convert DataFrame to a serializable format (JSON)
        result = data.to_dict(orient="records")
        
        return {"data": result}
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching the JSON data: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Error processing JSON data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
