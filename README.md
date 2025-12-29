Fake News Detector

This project is a Fake vs Real News detection system built using machine learning and natural language processing.
It takes raw news text as input and predicts whether the article is REAL or FAKE, then presents the result through a simple Streamlit web interface.

The goal of this project was to move beyond notebooks and build something that looks and feels like a real ML product.

 Why this project?

Fake news has become a serious problem, affecting public trust, media credibility, and decision-making.
This project explores how machine learning can help identify misleading or false news articles automatically.

It was built as a portfolio project to demonstrate practical ML skills, clean code structure, and deployment readiness.

How it works 
1. Text preprocessing

Each news article goes through a cleaning process:

Convert text to lowercase

Remove punctuation

Expand contractions 

Remove common stopwords

Apply stemming

This helps reduce noise and improve model performance.

2. Feature extraction

Text is converted into numerical features using TF-IDF

Both unigrams and bigrams are used to capture important word patterns

3. Model

A Linear Support Vector Machine (LinearSVC) is trained

This model works very well for high-dimensional text data like TF-IDF vectors

4. Prediction

The trained model predicts whether an article is REAL or FAKE

A confidence-like score is shown based on the model’s decision function


 How to run the project locally
1. Create and activate a virtual environment

python -m venv .venv
.venv\Scripts\activate

2. Install dependencies

pip install -r requirements.txt

3. Train the model

python -m src.train --data data/fake_or_real_news.csv

This step trains the model and saves all required files into the models/ folder.

4. Run a prediction from the terminal

python -m src.predict --text "This is a sample news article"

5. Launch the Streamlit app

streamlit run app/streamlit_app.py

The app will open in your browser and allow you to paste text or upload a .txt file.

 Streamlit Web App

The web app allows users to:

Paste a news article or upload a text file

Get a REAL / FAKE prediction

See a confidence-like indicator for the prediction

Note: The model uses LinearSVC, which does not output true probabilities.
The confidence shown is a transformed decision score for user interpretation.

 Dataset

Binary labeled dataset (REAL / FAKE)

English-language news articles

Roughly balanced classes

Used for learning and experimentation purposes

 Limitations

TF-IDF does not capture deep semantic meaning

English-only dataset

No transformer-based models 

Confidence score is not a calibrated probability

 Future improvements

Add explainability (LIME / SHAP) to highlight influential words

Experiment with transformer models (BERT)

Improve confidence calibration

Deploy publicly using Hugging Face Spaces

Add logging and monitoring

 About the author

Daad Alhassan
Data Science / Machine Learning Portfolio Project

This project demonstrates:

End-to-end ML pipeline design

Clean and modular code structure

Practical deployment skills

Real-world problem solving

