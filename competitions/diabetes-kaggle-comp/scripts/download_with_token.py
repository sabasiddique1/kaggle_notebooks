#!/usr/bin/env python3
"""
Download Kaggle competition data using API token
"""
import os
import json
import requests
import zipfile
from pathlib import Path

# Your Kaggle API token
KAGGLE_KEY = "KGAT_316cdbe2d368b62e88853ac90d167c86"

# Competition name
COMPETITION = "playground-series-s5e12"

def download_competition_data():
    """Download competition data using API token"""
    
    # Create output directory
    output_dir = Path("playground-series-s5e12")
    output_dir.mkdir(exist_ok=True)
    
    # Kaggle API endpoint
    base_url = "https://www.kaggle.com/api/v1"
    
    # Set up headers with authentication
    headers = {
        "Authorization": f"Bearer {KAGGLE_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"📥 Downloading data for competition: {COMPETITION}")
    
    # Download all competition files
    download_url = f"{base_url}/competitions/data/download-all/{COMPETITION}"
    
    try:
        response = requests.get(download_url, headers=headers, stream=True)
        response.raise_for_status()
        
        zip_path = f"{COMPETITION}.zip"
        print(f"💾 Saving to {zip_path}...")
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Download complete!")
        print(f"📦 Extracting {zip_path}...")
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print(f"✅ Files extracted to {output_dir}/")
        
        # List downloaded files
        files = list(output_dir.glob("*.csv"))
        print(f"\n📄 Downloaded files:")
        for f in files:
            print(f"   - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Clean up zip file
        os.remove(zip_path)
        print(f"\n🗑️  Removed {zip_path}")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("❌ Authentication failed!")
            print("   Please check your API token or Kaggle username.")
            print("   You may need to accept the competition rules first:")
            print(f"   https://www.kaggle.com/competitions/{COMPETITION}")
        else:
            print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    download_competition_data()

