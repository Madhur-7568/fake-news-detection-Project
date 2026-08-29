# 📰 Fake News Detection using NLP & Machine Learning

A Natural Language Processing (NLP) and Machine Learning project that classifies news articles as **Fake** or **Real** using **TF-IDF** and a **Support Vector Machine (SVM)**.

The project also includes a Streamlit web application for making predictions on new news articles.

---

## 🚀 Project Overview

The goal of this project is to build an end-to-end text classification system:

```text
News Title + News Content
          ↓
     Text Cleaning
          ↓
      Tokenization
          ↓
    Stopword Removal
          ↓
      Lemmatization
          ↓
       TF-IDF
          ↓
         SVM
          ↓
   Fake / Real Prediction
```

---

## 📊 Dataset

The original dataset contained **44,898 rows**.

After removing duplicate rows, the dataset contained:

- **44,689 rows** after the initial duplicate-removal stage
- **39,105 rows** after removing exact duplicate `title + text` combinations

Final class distribution after cleaning:

- Fake News (`0`)
- Real News (`1`)

The dataset contains news-related fields such as:

- `title`
- `text`
- `subject`
- `date`
- `label`

---

## 🧹 Text Preprocessing

The text preprocessing pipeline includes:

1. Lowercasing
2. Removing unnecessary punctuation/symbols
3. Tokenization using NLTK
4. Stopword removal
5. Lemmatization using `WordNetLemmatizer`

Example:

```text
playing → play
played  → play
plays   → play
studies → study
running → run
```

The processed title and article content are combined into the final text used by the classifier.

---

## 🔢 Feature Extraction

### TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical features.

The training data was transformed into:

```text
X_train: (35751, 50000)
X_test : (8938, 50000)
```

---

## 🤖 Machine Learning Model

The final classifier uses:

**Support Vector Machine (SVM)**

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Cross-validation
- Hyperparameter tuning
- Error analysis

---

## 📈 Model Evaluation

### Initial SVM Evaluation

```text
Accuracy : 0.995860
Precision: 0.995522
Recall   : 0.995757
F1 Score : 0.995639
```

Confusion Matrix:

```text
[[4677   19]
 [  18 4224]]
```

---

## 🔍 Duplicate & Leakage Analysis

Duplicate analysis was performed to make the evaluation more reliable.

Exact duplicate `title + text` rows removed:

```text
5584
```

After cleaning:

```text
Before: 44689
After : 39105
Removed: 5584
```

Duplicate-title analysis showed:

```text
Rows with duplicate titles after cleaning: 716
Titles having different labels: 0
```

An article-level train/test overlap check was also performed:

```text
Exact train-test article overlap: 0
```

However, title overlap across the train/test split was observed:

```text
Train titles: 31017
Test titles : 7801
Overlapping titles: 89
```

This is documented because duplicate or repeated titles can make standard random-split performance look stronger than performance on genuinely unseen articles.

---

## 🔬 Cross Validation

5-fold cross-validation produced:

```text
CV Scores:
[0.99156118 0.99258407 0.99232835 0.99360696 0.99437412]

Mean CV Accuracy: 0.9928909347
Std CV Accuracy : 0.0009890848
```

---

## ⚙️ Hyperparameter Tuning

Grid search was performed for the SVM parameter `C`.

Best result:

```text
Best Parameters:
{'svm__C': 1}

Best CV Accuracy:
0.9920214806290756
```

---

## 🧪 Final Model Evaluation

The final model achieved:

```text
Final Accuracy : 0.998977
Final Precision: 0.998116
Final Recall   : 1.000000
Final F1 Score : 0.999057
```

Confusion Matrix:

```text
[[3574    8]
 [   0 4239]]
```

These results should be interpreted together with the duplicate/title-overlap analysis above rather than treated as evidence that the classifier is perfect.

---

## ❌ Error Analysis

The model's mistakes were manually analyzed.

For the grouped evaluation:

```text
Wrong predictions: 57

Fake → Real: 31
Real → Fake: 26
```

This analysis helps identify difficult examples and understand where a text classifier can fail.

Examples of difficult cases included articles with:

- Sensational headlines
- Political language
- Breaking-news wording
- Similar writing patterns between classes
- Short or ambiguous text

---

## 🧠 Decision Score

The SVM provides a decision score using:

```python
model.decision_function()
```

Interpretation in this project:

```text
Negative score → Fake News
Positive score → Real News
```

The decision score is **not a probability**.

---

## 🌐 Streamlit Web App

The project includes a simple web application built with Streamlit.

Users can enter:

- News Title
- News Content

and receive:

- Fake/Real prediction
- Decision score
- Model information

Run the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
Fake-News-Detection/
│
├── app.py
├── fake_news_svm_pipeline.pkl
├── Fake-News-Detection.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Support Vector Machine
- Joblib
- Streamlit
- Jupyter Notebook

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

---

## ⚠️ Disclaimer

This project is an educational machine-learning classifier. A prediction of **Fake News** does not establish that a real-world claim is false, and a **Real News** prediction does not verify the claim independently.

The model learns patterns from its training dataset and can make incorrect predictions, especially on unfamiliar topics, writing styles, or claims.

---

## 👨‍💻 Project Focus

This project was built to practice and demonstrate:

- NLP preprocessing
- Text classification
- TF-IDF feature engineering
- SVM classification
- Model evaluation
- Cross-validation
- Hyperparameter tuning
- Error analysis
- Model serialization
- Streamlit deployment
