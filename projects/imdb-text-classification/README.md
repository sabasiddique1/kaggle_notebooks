## Text Classification Pipeline

End-to-end sentiment classification on the IMDB reviews dataset. The notebook covers preprocessing, BoW/TF-IDF features, word2vec + UMAP visualization, ML baselines, and a BiLSTM model with evaluation.

## Dataset
- IMDB Dataset of 50K Movie Reviews (Kaggle): https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Project Structure
- `notebook84b0f7b11b.ipynb` — full pipeline notebook
- `src/` — small helper module(s)
- `requirements.txt`

## How to Run

### Kaggle
1. Upload the notebook.
2. Add the dataset above to the notebook.
3. Run all cells.

### Local
```bash
pip install -r requirements.txt
jupyter notebook notebook84b0f7b11b.ipynb
```

## Results (short)
- TF-IDF + LinearSVC is the strongest ML baseline.
- BiLSTM with embeddings provides a deep learning comparison.
- See the notebook for metrics and the comparison table.
