#!/bin/bash

# Download script for Kaggle competition data
# Make sure to set your Kaggle username before running

export PATH="$HOME/Library/Python/3.9/bin:$PATH"

# Set your Kaggle username here (replace with your actual Kaggle username)
# You can find it at: https://www.kaggle.com/account
export KAGGLE_USERNAME="your_kaggle_username"  # CHANGE THIS!
export KAGGLE_KEY="KGAT_316cdbe2d368b62e88853ac90d167c86"

# Create kaggle.json with correct username
mkdir -p ~/.kaggle
python3 << EOF
import json
import os
data = {
    'username': os.environ.get('KAGGLE_USERNAME'),
    'key': os.environ.get('KAGGLE_KEY')
}
with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    json.dump(data, f)
print(f"Created kaggle.json for user: {data['username']}")
EOF

chmod 600 ~/.kaggle/kaggle.json

# Download the competition data
cd /Users/saba/Desktop/diabetes-kaggle-comp
echo "Downloading competition data..."
kaggle competitions download -c playground-series-s5e12

# Extract if zip file was downloaded
if [ -f playground-series-s5e12.zip ]; then
    echo "Extracting data..."
    unzip -q playground-series-s5e12.zip -d playground-series-s5e12/
    echo "✅ Data extracted to ./playground-series-s5e12/"
    ls -lh playground-series-s5e12/
fi

