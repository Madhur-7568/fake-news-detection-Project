# 📰 Fake News Detection using NLP

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

🚀 **[Live Demo](https://madhur-fake-news-detector-project.streamlit.app/)**

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

---

## 📊 Model Performance

The final SVM model achieved excellent performance on the test dataset.

| Metric | Score |
|---|---:|
| Accuracy | **99.90%** |
| Precision | **99.81%** |
| Recall | **100.00%** |
| F1 Score | **99.91%** |

### 🔄 Cross-Validation

- Mean CV Accuracy: **99.29%**
- Standard Deviation: **0.00099**

### 🧩 Confusion Matrix

```text
[[3574    8]
 [   0 4239]]

 