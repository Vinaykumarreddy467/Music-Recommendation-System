
---

# Music Recommendation System

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Technologies Used](#technologies-used)
5. [Getting Started](#getting-started)
6. [Data Preparation and Model Training](#data-preparation-and-model-training)
7. [Model Building](#model-building)
8. [Saving Models](#saving-models)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview

The **Music Recommendation System** utilizes natural language processing (NLP) and machine learning algorithms to suggest songs based on user input. This application analyzes user queries and employs a recommendation engine to provide relevant music suggestions.

---

## Features

- User-friendly interface for inputting text queries (e.g., song titles, artists, genres).
- NLP-powered query analysis to extract relevant keywords and entities.
- Knowledge graph-based recommendation engine considering user preferences and music relationships.
- Recommendations based on song lyrics and metadata.

---

## How It Works

1. **User Input**: The user inputs a text query (e.g., "songs like Happy by Pharrell Williams").
2. **Query Analysis**: NLP algorithms analyze the query to extract keywords (e.g., "Happy", "Pharrell Williams", "pop").
3. **Recommendation Generation**:
   - A knowledge graph is queried to find related songs and artists.
   - The recommendation engine generates a list of suggested songs based on user preferences and music relationships.
   - Results are ranked by relevance and popularity.

---

## Technologies Used

- Python
- NLTK (Natural Language Toolkit)
- scikit-learn
- Streamlit for web application
- Music streaming platform APIs (e.g., Spotify)

---

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Step 1: Clone the Repository

- Open a terminal or command prompt.
- Run the command:
  ```bash
  git clone https://github.com/Vinaykumarreddy467/Music-Recommendation-System.git
  ```

### Step 2: Install Dependencies

- Navigate to the cloned repository directory.
- Run the following commands to install necessary packages:

  ```bash
  pip install pandas
  pip install nltk
  pip install scikit-learn
  pip install streamlit
  pip install spotipy
  ```

### Step 3: Set Up Music Streaming API Credentials

- Create a file named `credentials.json` in the repository directory.
- Add your music streaming platform API credentials to the file (e.g., Spotify client ID and secret).

### Step 4: Run the Application

- Navigate to the repository directory:
  ```bash
  cd <repository-directory>
  ```
- Run the command:
  ```bash
  python -m streamlit run app.py
  ```

### Step 5: Access the Application

- Open a web browser and navigate to `http://localhost:5000`.
- Input a text query (e.g., "songs like Happy by Pharrell Williams") and press Enter.
- View the recommended songs and listen to them directly from the music streaming platform.

---

## Data Preparation and Model Training

1. **Data Collection**: Obtain the dataset (e.g., from Kaggle).
   - Example Dataset Link: [Kaggle Dataset](https://www.kaggle.com/datasets/notsh...)

2. **Text Preprocessing**:
   - Clean and preprocess the text by removing special characters, punctuation, and converting all letters to lowercase.
   - Tokenize the descriptions into individual words or phrases.
   - Remove stopwords (common words like "and," "the," etc.).

3. **Feature Extraction**:
   - Convert the tokenized descriptions into numerical representations using techniques like **TF-IDF** or word embeddings (Word2Vec, GloVe).

4. **Building a Recommender Model**:
   - Use **Content-Based Filtering** to analyze the song descriptions.
   - Calculate similarity scores between songs based on their preprocessed descriptions.
   - Recommend songs that have similar descriptions to those the user has interacted with in the past.

5. **User Interaction and Recommendations**:
   - Allow users to input their preferences (e.g., a sample song).
   - Use the selected song's description for recommendation.
   - Rank the songs based on similarity scores and present the top recommendations to the user.

---

## Model Building

This section describes how to build the recommendation model using the provided code:

1. **Load Dataset**: Use `pandas` to load the song dataset.
2. **Data Cleaning and Sampling**: Sample the dataset and clean the song lyrics.
3. **Tokenization and Stemming**: Tokenize lyrics into individual words and apply stemming.
4. **TF-IDF Vectorization**: Transform the cleaned text into a TF-IDF matrix.
5. **Cosine Similarity Calculation**: Compute similarity scores based on the TF-IDF matrix.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import pickle

# Load dataset
df = pd.read_csv("spotify_millsongdata.csv")
df = df.sample(5000).drop('link', axis=1).reset_index(drop=True)

# Text preprocessing
df['text'] = df['text'].str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ', regex=True)
stemmer = PorterStemmer()

def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)

df['text'] = df['text'].apply(lambda x: tokenization(x))

# TF-IDF Vectorization
tfidvector = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidvector.fit_transform(df['text'])

# Cosine Similarity
similarity = cosine_similarity(matrix)
```

---

## Saving Models

After building the model, save the processed song data and the similarity matrix for use in the Streamlit application:

```python
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(df, open('df.pkl', 'wb'))
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

---

## License

This project is licensed under the MIT License.

---
