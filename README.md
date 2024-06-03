# Information Retrieval System

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
The Information Retrieval System is designed to efficiently retrieve relevant answers to user queries from a dataset of questions and answers. This project implements various text preprocessing techniques and utilizes TF-IDF vectorization for indexing and querying.

## Features
- Text cleaning and normalization
- Tokenization and stopword removal
- Stemming and lemmatization
- Synonym expansion using WordNet
- TF-IDF vectorization for indexing
- Query processing and ranking using cosine similarity

## Installation
1. **Clone the repository**: Download the project from GitHub.
2. **Create and activate a virtual environment**: Set up a virtual environment to manage dependencies.
2. **Download NLTK data**: Ensure that all required NLTK datasets are available for text processing.

## Usage
1. **Preprocess the data**: Prepare the dataset by cleaning and normalizing the text.
2. **Index the preprocessed data using TF-IDF**: Create a TF-IDF matrix to index the preprocessed text.
3. **Process and query the system**: Implement the query system to process user queries and retrieve relevant answers based on the TF-IDF index.

## Data Preprocessing
The preprocessing steps include:
- **Cleaning**: Removing punctuation, HTML tags, and brackets to reduce noise.
- **Normalization**: Converting text to lowercase, expanding abbreviations, and correcting spelling mistakes.
- **Tokenization**: Splitting text into individual words or tokens.
- **Stopword Removal**: Eliminating common words that do not contribute to the meaning.
- **Stemming and Lemmatization**: Reducing words to their base or root form.
- **Synonym Expansion**: Using WordNet to expand synonyms and enhance retrieval performance.

## Evaluation
The performance of the Information Retrieval System is evaluated using metrics such as:
- **Precision**: The fraction of relevant instances among the retrieved instances.
- **Recall**: The fraction of relevant instances that have been retrieved over the total amount of relevant instances.
- **F1-Score**: The harmonic mean of precision and recall.
- **Mean Average Precision (MAP)**: A measure of the quality of the retrieval process.

## Acknowledgements
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)
