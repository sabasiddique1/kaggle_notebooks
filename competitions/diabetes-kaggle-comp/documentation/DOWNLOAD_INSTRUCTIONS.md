# Download Competition Data

## Option 1: Using Kaggle CLI (Recommended)

1. **Install Kaggle CLI** (if not already installed):
   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle API credentials**:
   - Go to https://www.kaggle.com/account
   - Scroll down to "API" section
   - Click "Create New API Token" - this downloads `kaggle.json`
   - Place `kaggle.json` in `~/.kaggle/` directory:
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Download the competition data**:
   ```bash
   cd /Users/saba/Desktop/diabetes-kaggle-comp
   kaggle competitions download -c playground-series-s5e12
   ```

4. **Extract the zip file**:
   ```bash
   unzip playground-series-s5e12.zip -d playground-series-s5e12/
   ```

## Option 2: Manual Download

1. Go to: https://www.kaggle.com/competitions/playground-series-s5e12/data
2. Download all data files
3. Extract to: `./playground-series-s5e12/` directory

## Option 3: Download External Dataset (Optional but Recommended)

The external dataset improves performance significantly:

```bash
kaggle datasets download -d alexteboul/diabetes-health-indicators-dataset
unzip diabetes-health-indicators-dataset.zip -d diabetes-health-indicators-dataset/
```

## Expected File Structure

After downloading, your directory should look like:

```
diabetes-kaggle-comp/
├── comprehensive-diabetes-prediction.ipynb
├── playground-series-s5e12/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── diabetes-health-indicators-dataset/  (optional)
    └── diabetes_dataset.csv
```

## Verify Download

Run this to check if files exist:
```bash
cd /Users/saba/Desktop/diabetes-kaggle-comp
ls -lh playground-series-s5e12/*.csv
```

