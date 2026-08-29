# 📰 Fake News Detection using NLP

A Machine Learning project that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP), TF-IDF Vectorization, and Support Vector Machine (SVM).

The project also includes an interactive **Streamlit web application** where users can enter a news title and article content to get a prediction.

---

## 📌 Project Overview

Fake news has become a major problem with the rapid growth of online news and social media.

In this project, Natural Language Processing techniques are used to process news articles and extract meaningful textual features. A Support Vector Machine classifier is then trained to classify articles into two categories:

- **0 → Fake News**
- **1 → Real News**

---

## 🎯 Objectives

- Clean and preprocess raw news text
- Apply NLP techniques to news articles
- Convert text into numerical features using TF-IDF
- Train and compare multiple Machine Learning models
- Evaluate model performance using classification metrics
- Perform error analysis
- Build a reusable prediction pipeline
- Create an interactive Streamlit application

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## 🔄 NLP Pipeline

The project follows this workflow:

```text
Raw News Data
      ↓
Data Cleaning
      ↓
Duplicate Removal
      ↓
Text Preprocessing
      ↓
Tokenization
      ↓
Stopword Removal
      ↓
Lemmatization
      ↓
TF-IDF Vectorization
      ↓
Machine Learning Model
      ↓
Prediction