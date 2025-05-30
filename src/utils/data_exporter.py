import os
import json
import csv
import pandas as pd
import numpy as np
from src.utils.file_helpers import ensure_directory

def export_to_json(analysis_results, output_path):
    """Export analysis results to JSON file"""
    ensure_directory(output_path)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(analysis_results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return output_path

def export_to_csv(data, output_path, headers=None):
    """Export data to CSV file"""
    ensure_directory(output_path)
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(output_path, index=False)
    else:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            if headers:
                writer.writerow(headers)
            writer.writerows(data)
    
    return output_path
