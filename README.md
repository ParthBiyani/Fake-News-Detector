# Neural Sentinel: Fake News Detection with NLP + ML

Neural Sentinel is a notebook-based pipeline for detecting fake news using Natural Language Processing (NLP) and classical Machine Learning. It combines two labeled news corpora (Fake and True), cleans and normalizes the text, builds TF-IDF features, and trains a Linear Support Vector Classifier (LinearSVC) to distinguish real vs. fake articles. The project also includes visual analytics via word clouds to highlight linguistic differences between the two classes.

This repository is organized as a single, reproducible notebook that walks through the full workflow: data loading, exploratory visualization, text cleaning, feature engineering, model training, evaluation, and a simple interactive prediction step.

## What This Project Does
- Loads two datasets of news articles (Fake.csv and True.csv).
- Labels fake articles as 0 and real articles as 1, then merges both into a unified dataset.
- Visualizes class balance and topic distribution.
- Cleans text using:
	- lowercasing
	- contraction normalization
	- regex-based character cleanup
	- stopword removal (spaCy + NLTK)
	- lemmatization (NLTK WordNet)
- Builds TF-IDF vectors from text and trains a LinearSVC classifier.
- Evaluates the model with accuracy and a confusion matrix.
- Generates word clouds for real vs. fake news to provide interpretability.

## Project Structure
- Fake_news_detector.ipynb: end-to-end notebook (main project).
- Fake.csv, True.csv: labeled datasets of news articles.
- requirements.txt: Python dependencies.
- Danger sign.png, Thumbs up.png: masks for word cloud visualizations.
- spacy_env/: local virtual environment (already included in this workspace).

## How The Notebook Is Organized
1) Data procurement: load datasets and label classes.
2) Data visualization: class and subject distribution plots.
3) Data cleanup: remove blanks, stopwords, normalize contractions, and lemmatize.
4) Word cloud: generate class-specific word clouds using custom masks.
5) Classifier model: train TF-IDF + LinearSVC pipeline and evaluate.
6) Prediction: input custom text and print fake/real prediction.

## Requirements
Install dependencies (recommended in a fresh virtual environment):

```bash
pip install -r requirements.txt
```

The notebook also expects the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Running The Project
Open and run the notebook in Jupyter or VS Code:

```bash
jupyter notebook Fake_news_detector.ipynb
```

Run the notebook cells from top to bottom. The final cell prompts for a news snippet and prints a classification.

## Important Notes About Paths
The notebook currently uses absolute Windows paths (for example, `D:\Projects\Fake-News-Detector\Fake.csv` and the word cloud images). If your project location differs, update these paths in the notebook to point to the files in this repository.

## Model Details
- Feature extraction: TF-IDF (bag-of-words with term weighting).
- Classifier: LinearSVC (fast and effective for sparse, high-dimensional text).
- Train/test split: 70/30 (random split).
- Metrics: accuracy and confusion matrix.

## Outputs You Can Expect
- Class distribution and subject distribution plots.
- Word clouds for real vs. fake news (masked by custom images).
- Accuracy score and confusion matrix.
- Interactive prediction of new, user-provided text.

## Dataset Schema
Both Fake.csv and True.csv include at least the following columns:
- `title`: headline of the article.
- `text`: full article content (used for training).
- `subject`: topic category (used for visualization).
- `date`: publication date.

Only `text` and the generated `category` label are used for modeling.

## Why LinearSVC + TF-IDF
TF-IDF captures the relative importance of terms across the corpus and produces sparse, high-dimensional vectors. LinearSVC is a strong baseline for text classification, particularly with large vocabularies, and performs well with TF-IDF features.

## Limitations
- This is a supervised model trained on a fixed dataset; generalization to new domains depends on the similarity of language and topics.
- The pipeline does not use deep contextual embeddings (e.g., BERT), so it may miss subtle semantic cues.

## Future Improvements
- Persist the trained model with joblib for reuse.
- Add cross-validation and hyperparameter tuning.
- Replace the simple regex-based normalization with a more robust tokenizer.
- Provide a small CLI or web demo for interactive inference.
