#!/usr/bin/env python3
"""
Run the comprehensive diabetes prediction notebook locally
Adapts Kaggle paths to local paths
"""
import json
import os
import sys
import subprocess
import tempfile

# Read the notebook
notebook_path = 'notebooks/comprehensive/comprehensive-diabetes-prediction.ipynb'

print("=" * 70)
print("Running Comprehensive Diabetes Prediction Notebook")
print("=" * 70)

# Read notebook
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Modify paths in the notebook
kaggle_paths = {
    '/kaggle/input/playground-series-s5e12/train.csv': './playground-series-s5e12/train.csv',
    '/kaggle/input/playground-series-s5e12/test.csv': './playground-series-s5e12/test.csv',
    '/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv': './diabetes-health-indicators-dataset/diabetes_binary_health_indicators_BRFSS2015.csv'
}

# Check if train.csv exists, if not, check alternative locations
if not os.path.exists('./playground-series-s5e12/train.csv'):
    alt_paths = [
        './data/train.csv',
        '../playground-series-s5e12/train.csv',
        './train.csv'
    ]
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            kaggle_paths['/kaggle/input/playground-series-s5e12/train.csv'] = alt_path
            kaggle_paths['/kaggle/input/playground-series-s5e12/test.csv'] = alt_path.replace('train.csv', 'test.csv')
            print(f"Found train.csv at: {alt_path}")
            break
    else:
        print("ERROR: train.csv not found!")
        print("Please download the competition data first.")
        print("Expected location: ./playground-series-s5e12/train.csv")
        sys.exit(1)

# Replace paths in notebook cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        for kaggle_path, local_path in kaggle_paths.items():
            if kaggle_path in source:
                cell['source'] = [line.replace(kaggle_path, local_path) for line in cell['source']]

# Save modified notebook to temp file
temp_nb = tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False)
json.dump(nb, temp_nb, indent=1)
temp_nb.close()

print(f"\nModified notebook saved to: {temp_nb.name}")
print("Executing notebook...\n")

# Execute notebook
try:
    result = subprocess.run(
        ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', temp_nb.name],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("Notebook executed successfully!")
        print("=" * 70)
        print(f"\nOutput notebook: {temp_nb.name}")
    else:
        print("Error executing notebook:")
        print(result.stderr)
        sys.exit(1)
        
except FileNotFoundError:
    print("ERROR: jupyter nbconvert not found!")
    print("Please install jupyter: pip install jupyter nbconvert")
    sys.exit(1)




