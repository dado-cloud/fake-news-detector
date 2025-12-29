# 📰 Fake News Detector (ML Pipeline + Streamlit App)

A complete end-to-end **Fake vs Real News classification system** built using machine learning.  
The project covers the full ML lifecycle: preprocessing, training, inference, and deployment via a Streamlit web app.

---

##  Project Overview

Fake news poses serious risks to public trust and information integrity.  
This project aims to automatically classify news articles as **REAL** or **FAKE** using natural language processing and machine learning.

**Key highlights:**
- Clean, modular ML pipeline (not a notebook-only project)
- Reusable preprocessing and inference code
- Trained TF-IDF + Linear SVM model
- Interactive Streamlit web application
- Portfolio-ready structure

---

##  Machine Learning Approach

### Pipeline
1. **Text Preprocessing**
   - Lowercasing
   - Punctuation removal
   - Contraction expansion
   - Stopword removal
   - Stemming

2. **Feature Engineering**
   - TF-IDF Vectorization
   - Unigrams + Bigrams

3. **Model**
   - Linear Support Vector Machine (LinearSVC)
   - Chosen for strong performance on high-dimensional sparse text data

4. **Evaluation**
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion Matrix

---

## Dataset 
This project uses the TextDB3 (Fake or Real News) dataset, which is publicly available on Kaggle. The dataset is designed for binary text classification tasks, specifically for detecting fake news vs. real news based on textual content.

The dataset contains news articles labeled as either fake or real, making it well-suited for training and evaluating Natural Language Processing (NLP) and machine learning models for misinformation detection.

🔹 Key Characteristics

Domain: News & Media

Task Type: Binary Classification

Data Type: Textual data

Language: English

🔹 Dataset Structure

Each record in the dataset represents a single news article and includes the following features:

title – The headline of the news article

text – The full body text of the article

label – The class label indicating the authenticity of the news

FAKE → Fake news

REAL → Real news

🔹 Use Cases

This dataset is commonly used for:

Fake news detection systems

Text classification experiments

NLP preprocessing techniques (tokenization, vectorization, TF-IDF, embeddings)

Model comparison (e.g., Naive Bayes, Logistic Regression, LSTM, Transformers)

🔹 Data Source

The dataset was collected from multiple online news sources and curated for research and educational purposes.

🔗 Dataset link: https://www.kaggle.com/datasets/hassanamin/textdb3
