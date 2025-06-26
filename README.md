 📰 Fake News Detection Using Machine Learning – Project Report
1. Introduction**

In the digital age, the spread of fake news has become a critical issue. Fake news can mislead the public, create social unrest, and manipulate opinions.
This project aims to develop a machine learning model that can accurately classify news articles as **REAL** or **FAKE** using Natural Language Processing (NLP) techniques and supervised learning algorithms.


🎯 **2. Objectives**

* To clean and preprocess news article data.
* To vectorize the textual content using TF-IDF.
* To train and evaluate multiple machine learning models.
* To accurately predict whether a given news article is real or fake.
* To build a modular and scalable solution.

🧩 **3. Dataset**

 📁 Source:

* Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

 📊 Dataset Details:

* **Files used**: `Fake.csv`, `True.csv`
* **Total records**: \~45,000
* **Columns**:

  * `title`: Title of the article
  * `text`: Full content
  * `subject`: Topic category
  * `date`: Date of publication

 🔖 Labels:

* `FAKE` – Label for fake news articles
* `REAL` – Label for real news articles

⚙️ **4. Methodology**

 🔄 Data Preparation:

* Merged `Fake.csv` and `True.csv`
* Assigned labels accordingly
* Selected only the `text` and `label` columns
* Cleaned the text using regular expressions and NLTK (stop words removal, punctuation cleaning, etc.)

🧪 Data Splitting:

* Split the data into training and testing sets (80:20 ratio)

🧠 Text Vectorization:

* Used `TfidfVectorizer` to convert text into numerical features

 🧠 Machine Learning Models:

* **PassiveAggressiveClassifier**
* **Logistic Regression**
* **Multinomial Naive Bayes**



📈 **5. Evaluation Metrics**

* **Accuracy Score**
* **Confusion Matrix**
* **Classification Report (Precision, Recall, F1-score)**

---

🏆 **6. Results**

| Model                   | Accuracy |
| ----------------------- | -------- |
| Passive Aggressive      | \~96%    |
| Logistic Regression     | \~95%    |
| Multinomial Naive Bayes | \~92%    |

> Passive Aggressive Classifier showed the best performance in terms of speed and accuracy.

---

 🧪 **7. Sample Prediction**

Tested the model with real-time custom news text input to verify whether the model can classify unseen data.

```python
sample = "Breaking: NASA discovers water on Mars!"
print(model.predict(vectorizer.transform([clean_text(sample)])))
```

---

 🚀 **8. Future Enhancements**

* Integrate deep learning models like **BERT** or **LSTM**
* Create a **Streamlit** or **Flask** based web interface
* Extend to multi-class classification (e.g., satire, clickbait)
* Deploy the model using **Heroku** or **Streamlit Cloud**

---

 📚 **9. Conclusion**

This project successfully demonstrates how Natural Language Processing and machine learning techniques can be applied to detect fake news with high accuracy. The model is effective and scalable for real-world applications.

---

📎 **10. References**

* [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* scikit-learn Documentation
* NLTK Official Documentation

---

Would you like this as a downloadable `.docx` or `.pdf`? Or do you want to use this for your GitHub README?
