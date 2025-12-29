# 📰 Fake News Detector  
**Machine Learning Pipeline + Streamlit App**

the app link: https://fake-news-detector-dp62flhfseki4ankcwkr77.streamlit.app/

An end-to-end **Fake vs. Real News classification system** built using machine learning and natural language processing.  
This project covers the full ML lifecycle — from text preprocessing and model training to deployment in an interactive **Streamlit web app**.

---

##  Project Overview

Fake news spreads rapidly and can significantly impact public trust and decision-making.  
This project aims to automatically classify news articles as **REAL** or **FAKE** using machine learning techniques.

The goal is not only to achieve good performance, but also to build a **clean, reusable, and deployable ML pipeline**.

### Key Highlights
- End-to-end machine learning workflow  
- Clean and modular preprocessing pipeline  
- TF-IDF + Linear SVM text classification model  
- Interactive Streamlit web application  

---

##  Machine Learning Approach

###  Text Preprocessing
The news text is cleaned and normalized using common NLP techniques:
- Lowercasing  
- Punctuation removal  
- Contraction expansion  
- Stopword removal  
- Stemming  

###  Feature Engineering
- TF-IDF vectorization  
- Unigrams and bigrams to capture individual words and short phrases  

###  Model
- **Linear Support Vector Machine (LinearSVC)**  
- Selected for its strong performance on high-dimensional, sparse text data  

### * Evaluation
The model is evaluated using:
- Accuracy  
- Precision, Recall, and F1-score  
- Confusion Matrix  

---

##  Dataset

This project uses the **TextDB3 (Fake or Real News)** dataset, which is publicly available on Kaggle.  
The dataset is designed for **binary text classification**, making it suitable for fake news detection tasks.

###  Key Characteristics
- **Domain:** News & Media  
- **Task Type:** Binary Classification  
- **Data Type:** Text  
- **Language:** English  

###  Dataset Structure
Each record in the dataset represents a single news article and includes:
- **`title`** – The headline of the news article  
- **`text`** – The full body text of the article  
- **`label`** – The authenticity label  
  - `FAKE` → Fake news  
  - `REAL` → Real news  

###  Use Cases
This dataset is commonly used for:
- Fake news detection systems  
- Text classification experiments  
- NLP preprocessing and feature engineering (TF-IDF, embeddings)  
- Model comparison (Naive Bayes, Logistic Regression, LSTM, Transformers)  

###  Data Source
The dataset was collected from multiple online news sources and curated for research and educational purposes.

 **Dataset link:**  
https://www.kaggle.com/datasets/hassanamin/textdb3  

---

##  Streamlit Web App

The trained model is deployed using **Streamlit**, allowing users to:
- Enter or paste a news article  
- Instantly receive a **REAL** or **FAKE** prediction  
- Interact with the model through a simple and user-friendly interface  

---

##  Final Notes

This project demonstrates a practical application of machine learning for misinformation detection, with a strong focus on:
- Clean code structure  
- Reproducibility  
- Real-world deployment  

It is well-suited for **students, researchers, and anyone interested in NLP and applied machine learning**.


