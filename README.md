# 📩 SMS Spam Classifier (NLP + Machine Learning)

🔗 **Live Demo:** https://sms-spam-classifier-3wekgcax8upybi8dks8hnd.streamlit.app/
🔗 **Project Repository:**https://github.com/Raj456999/SMS-SPAM-CLASSIFIER/ 

## 🚀 Overview
An end-to-end SMS Spam Classification system that detects whether a message is **Spam** or **Not Spam (Ham)** using Natural Language Processing and Machine Learning.

The project demonstrates a complete ML pipeline including preprocessing, feature engineering, model selection, evaluation, and deployment via a web application.

---

## 🎯 Features
- Real-time SMS classification
- Probability-based predictions with confidence score
- Context-aware detection using n-grams
- Comparison of multiple ML models
- Interactive UI built with Streamlit

---

## 🧠 Problem Statement
Traditional spam filters often fail on unseen or context-dependent phrases.  
This project explores how machine learning models handle:

- Word frequency vs contextual understanding  
- Generalization to unseen spam patterns  
- Trade-offs between precision and recall  

---

## ⚙️ Tech Stack

### 🐍 Language
- Python

### 📚 Libraries
- **NLP:** NLTK  
- **Feature Engineering:** TF-IDF Vectorizer (Scikit-learn)  
- **ML Models:**
  - Logistic Regression
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes
  - Support Vector Machine (SVC)
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - KNN

- **Model Tuning:** GridSearchCV  
- **Deployment:** Streamlit  
- **Data Handling:** Pandas, NumPy  

---

## 🔄 Pipeline

### 1. Data Preprocessing
- Lowercasing
- Tokenization
- Stopword removal
- Removal of punctuation and non-alphanumeric tokens

> Note: Stemming was removed to preserve phrase-level semantics like "lucky draw"

---

### 2. Feature Engineering
- TF-IDF Vectorization
- N-grams (1,2) to capture contextual phrases

---

### 3. Model Training
Multiple models were trained and evaluated.  
Final model selected:

> **Logistic Regression** (best balance of performance and probability calibration)

---

### 4. Evaluation
- Accuracy
- Precision
- Recall

### Key Observation:
Model performance is highly dependent on dataset diversity.

---

### 5. Deployment
- Built using Streamlit
- User inputs SMS
- Outputs:
  - Prediction (Spam / Not Spam)
  - Confidence score
  - Visual progress bar

---

## 📊 Example

# Input:
Congratulations! You won a free car in lucky draw
# Output:
<img width="1143" height="640" alt="image" src="https://github.com/user-attachments/assets/48622454-6fbc-41a2-94cb-71eaaa96c86e" />

- Prediction: Spam
- Confidence: 72.24%


---

## ⚠️ Challenges
- Dataset bias affecting generalization
- Handling unseen spam patterns
- Maintaining consistency between model and vectorizer during deployment

---

## 🚀 Future Work
- Integrate transformer models (BERT)
- Use larger real-world datasets
- Improve generalization with data augmentation
- Deploy scalable backend (FastAPI)

---

## 🧪 Installation

```bash
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
pip install -r requirements.txt
streamlit run app.py
```
📦 Requirements:
- streamlit
- scikit-learn
- pandas
- numpy
- nltk

👨‍💻 Author 
- Satteraju Palla (Aspiring Data scientist)


⭐ Support

If you found this useful, give it a ⭐ on GitHub!

