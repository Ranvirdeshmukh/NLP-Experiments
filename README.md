
Welcome to my NLP repository! This project is a collection, where I explored various Natural Language Processing techniques. The repository is divided into four major exercises:

## Overview

1. **Clustering Shakespeare**
   - **Objective:** Process Shakespeare's plays using TF-IDF and group them via KMeans clustering.
   - **Visualization:** Generate a hierarchical clustering dendrogram.
   - **Extras:** Predict clusters for new text snippets.

2. **Spanish FastText Word Embeddings**
   - **Objective:** Download and utilize a Spanish FastText model.
   - **Tasks:** Retrieve similar words for "hombre" and "mujer", perform analogy operations (e.g., `rey - hombre + mujer`), and visualize selected word embeddings using t-SNE.

3. **N-gram Language Modeling and Perplexity**
   - **Objective:** Build n-gram language models (from unigram to four-gram) using a Sherlock corpus.
   - **Tasks:** Generate text samples, compute MLE probabilities for selected n-grams, and evaluate the perplexity of various test sentences.

4. **Naïve Bayes Sentiment Analysis**
   - **Objective:** Perform sentiment analysis on review datasets using a Naïve Bayes classifier.
   - **Tasks:** Feature extraction from unigrams and bigrams, model training/testing, and performance evaluation (accuracy, precision, recall, F-measure).
   - **Data Sources:** Amazon and Google review datasets.

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `nltk`
  - `fasttext`
  - `gdown`
  - `scipy`
- Other tools: Git

Make sure to install the required libraries using `pip` if they are not already installed:
```bash
pip install numpy matplotlib scikit-learn nltk fasttext gdown scipy
