"""
FastAPI application for CSV file processing.
Provides endpoints for uploading CSV files, cleaning data, generating statistics, and detecting anomalies.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io
from typing import Dict, Any
from data_processor import DataProcessor
import os

app = FastAPI(
    title="CSV Data Processing API",
    description="Upload CSV files, clean data, generate statistics, and detect anomalies",
    version="1.0.0"
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Serve the main UI page."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {
        "message": "CSV Data Processing API",
        "endpoints": {
            "upload_and_process": "/process-csv",
            "docs": "/docs"
        }
    }


@app.get("/process-csv")
async def process_csv_info():
    """GET endpoint providing information about how to use the POST endpoint."""
    return {
        "message": "This endpoint requires a POST request with a CSV file",
        "method": "POST",
        "usage": {
            "curl": 'curl -X POST "http://localhost:8000/process-csv?detect_anomalies=true&anomaly_contamination=0.1" -F "file=@your_file.csv"',
            "python": """
import requests
url = "http://localhost:8000/process-csv"
files = {"file": open("your_file.csv", "rb")}
params = {"detect_anomalies": True, "anomaly_contamination": 0.1}
response = requests.post(url, files=files, params=params)
print(response.json())
            """,
            "interactive_docs": "Visit http://localhost:8000/docs to use the interactive API interface"
        },
        "parameters": {
            "file": "CSV file to upload (required, multipart/form-data)",
            "detect_anomalies": "Whether to perform anomaly detection (optional, default: true)",
            "anomaly_contamination": "Expected proportion of outliers (optional, default: 0.1, range: 0-0.5)"
        }
    }


@app.post("/process-csv")
async def process_csv(
    file: UploadFile = File(..., description="CSV file to process"),
    detect_anomalies: bool = True,
    anomaly_contamination: float = 0.1
) -> JSONResponse:
    """
    Upload a CSV file and process it.
    
    Args:
        file: CSV file to upload
        detect_anomalies: Whether to perform anomaly detection (default: True)
        anomaly_contamination: Expected proportion of outliers for anomaly detection (default: 0.1)
    
    Returns:
        JSON response with:
        - cleaning_report: Information about data cleaning steps
        - statistics: Summary statistics of the cleaned data
        - anomalies: Anomaly detection results (if enabled)
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    try:
        # Read CSV file
        contents = await file.read()
        
        # Try to detect encoding
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading CSV file: {str(e)}"
                )
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Initialize data processor
        processor = DataProcessor(df)
        
        # Clean the data
        cleaned_df = processor.clean_data()
        
        # Generate statistics
        statistics = processor.generate_statistics()
        
        # Detect anomalies if requested
        anomalies = None
        if detect_anomalies:
            # Validate contamination parameter
            if not 0 < anomaly_contamination <= 0.5:
                raise HTTPException(
                    status_code=400,
                    detail="anomaly_contamination must be between 0 and 0.5"
                )
            anomalies = processor.detect_anomalies(contamination=anomaly_contamination)
        
        # Prepare response
        response = {
            "file_name": file.filename,
            "file_size_bytes": len(contents),
            "cleaning_report": processor.cleaning_report,
            "statistics": statistics,
        }
        
        if anomalies is not None:
            response["anomalies"] = anomalies
        
        return JSONResponse(content=response)
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error parsing CSV file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

