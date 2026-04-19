#!/usr/bin/env python3
"""
Generate submission.csv by running the diabetes prediction notebook
"""
import os
import sys
import nbformat
from nbclient import NotebookClient

# Set notebook path
notebook_path = 'notebooks/comprehensive/diabetes_prediction.ipynb'

if not os.path.exists(notebook_path):
    print(f"ERROR: Notebook not found at {notebook_path}")
    sys.exit(1)

print("=" * 70)
print("Generating submission.csv from notebook")
print("=" * 70)
print(f"Notebook: {notebook_path}")
print("This will take 5-10 minutes...\n")

# Read notebook
print("Reading notebook...")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Execute notebook using nbclient (more reliable)
try:
    print("Executing notebook...")
    client = NotebookClient(
        nb,
        timeout=900,  # 15 minute timeout
        kernel_name='python3',
        resources={'metadata': {'path': os.getcwd()}}
    )
    client.execute()
    print("\n✅ Notebook executed successfully!")
    
    # Check if submission.csv was created
    submission_paths = ['submission.csv', '/kaggle/working/submission.csv']
    submission_found = False
    
    for path in submission_paths:
        if os.path.exists(path):
            print(f"\n✅ Submission file found at: {path}")
            print(f"   File size: {os.path.getsize(path):,} bytes")
            submission_found = True
            
            # Copy to main directory if needed
            if path != 'submission.csv':
                import shutil
                shutil.copy(path, 'submission.csv')
                print(f"   ✅ Copied to: submission.csv")
            break
    
    if not submission_found:
        print("\n⚠️  WARNING: submission.csv not found!")
        print("   The notebook may not have completed successfully.")
        print("   Please check the notebook output for errors.")
    
except Exception as e:
    print(f"\n❌ Error executing notebook: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)

