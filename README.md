# CSV Data Processing System

A FastAPI-based system for uploading CSV files, cleaning data, generating summary statistics, and detecting anomalies.

## Features

- **CSV Upload**: Upload CSV files via REST API
- **Data Cleaning**: 
  - Removes duplicate rows
  - Handles missing values (fills numeric columns with median, categorical with mode)
  - Removes leading/trailing whitespace
  - Converts columns to appropriate data types
- **Summary Statistics**: Comprehensive statistics including:
  - Dataset overview (rows, columns, memory usage)
  - Column-level statistics (mean, median, std, quartiles, etc. for numeric columns)
  - Categorical statistics (unique counts, most frequent values, etc.)
- **Anomaly Detection**: Uses Isolation Forest algorithm to detect outliers in numeric columns

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload
```

2. Access the application:
   - **Web UI**: Open `http://localhost:8000` in your browser
   - **Interactive API documentation**: `http://localhost:8000/docs`
   - **Alternative docs**: `http://localhost:8000/redoc`

### Using the Web UI

1. Open `http://localhost:8000` in your browser
2. Click the "Choose CSV File" button to select your CSV file
3. Optionally adjust the anomaly detection settings:
   - Toggle "Detect Anomalies" checkbox
   - Adjust the "Contamination" value (0.01 to 0.5)
4. Click "Process CSV" button
5. View the results including:
   - Summary cards with key metrics
   - Full JSON response with all statistics and anomaly details
   - Copy JSON button to copy the results

## API Endpoints

### POST `/process-csv`

Upload and process a CSV file.

**Parameters:**
- `file` (required): CSV file to upload
- `detect_anomalies` (optional, default: `true`): Whether to perform anomaly detection
- `anomaly_contamination` (optional, default: `0.1`): Expected proportion of outliers (0 to 0.5)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/process-csv?detect_anomalies=true&anomaly_contamination=0.1" \
  -F "file=@your_file.csv"
```

**Example using Python requests:**
```python
import requests

url = "http://localhost:8000/process-csv"
files = {"file": open("your_file.csv", "rb")}
params = {"detect_anomalies": True, "anomaly_contamination": 0.1}

response = requests.post(url, files=files, params=params)
print(response.json())
```

**Response Format:**
```json
{
  "file_name": "example.csv",
  "file_size_bytes": 12345,
  "cleaning_report": {
    "initial_rows": 1000,
    "final_rows": 950,
    "duplicates_removed": 50,
    "missing_values_by_column": {...},
    "cleaning_steps": [...]
  },
  "statistics": {
    "dataset_overview": {...},
    "column_statistics": {...}
  },
  "anomalies": {
    "anomalies_detected": 10,
    "anomaly_percentage": 1.05,
    "anomaly_indices": [...],
    "anomaly_details": [...]
  }
}
```

### GET `/`

Root endpoint with API information.

### GET `/health`

Health check endpoint.

## Example

1. Create a sample CSV file or use an existing one
2. Start the server: `python main.py`
3. Open `http://localhost:8000/docs` in your browser
4. Use the interactive interface to upload your CSV file
5. View the JSON response with cleaned data, statistics, and anomalies

