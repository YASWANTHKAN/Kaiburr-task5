## 🧠 **Kaiburr Assessment 2025 — Task 5**

### Text Classification on Consumer Complaint Dataset

---

### 📘 **Project Overview**

This project performs **multi-class text classification** on the official [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database).
The goal is to automatically categorize consumer complaints into four categories using NLP and machine learning.

| ID | Category                           |
| -- | ---------------------------------- |
| 0  | Credit reporting, repair, or other |
| 1  | Debt collection                    |
| 2  | Consumer Loan                      |
| 3  | Mortgage                           |

---

### ⚙️ **Workflow Summary**

1. **Data Loading & EDA**

   * Loaded ~11 million records from the CFPB dataset.
   * Filtered for 4 major categories and cleaned missing text fields.

2. **Feature Engineering**

   * Mapped product names to numeric labels.
   * Split dataset: 60% training, 20% validation, 20% testing.

3. **Text Preprocessing**

   * Applied **TF-IDF Vectorization** (`ngram_range=(1,2)`, `max_features=20,000`)
   * Removed English stopwords and lowercased all text.

4. **Model Training & Comparison**
   Implemented and compared four models using a unified pipeline:

   * Logistic Regression
   * Multinomial Naive Bayes
   * Random Forest Classifier
   * Linear SVM

5. **Evaluation Metrics**

   * Accuracy, Precision, Recall, F1-Score
   * Bar chart comparing model performance
   * Saved the best model for deployment

---

### 🧪 **Model Comparison Results**

| Model                   | Accuracy            | Notes                                       |
| ----------------------- | ------------------- | ------------------------------------------- |
| **Logistic Regression** | **0.9699 (96.99%)** | Best balance between precision and recall   |
| **Naive Bayes**         | 0.9476 (94.76%)     | Fast but weaker on minority class           |
| **Random Forest**       | 0.9094 (90.94%)     | Heavy runtime; underperforms on sparse data |
| **SVM (LinearSVC)**     | 0.9695 (96.95%)     | Excellent, close to Logistic Regression     |

📈 **Best Model:** Logistic Regression (Accuracy: 0.9699)
💾 Saved as: `best_complaint_model.pkl`

---

### 🖥️ **Classification Reports**

**Logistic Regression**

```
accuracy  = 0.9699
macro avg = precision 0.90 | recall 0.82 | f1-score 0.86
weighted avg = precision 0.97 | recall 0.97 | f1-score 0.97
```

**Naive Bayes**

```
accuracy  = 0.9476
macro avg = precision 0.84 | recall 0.76 | f1-score 0.75
weighted avg = precision 0.95 | recall 0.95 | f1-score 0.95
```

**Random Forest**

```
accuracy  = 0.9094
macro avg = precision 0.62 | recall 0.58 | f1-score 0.59
weighted avg = precision 0.90 | recall 0.91 | f1-score 0.90
```

**SVM**

```
accuracy  = 0.9695
macro avg = precision 0.90 | recall 0.82 | f1-score 0.85
weighted avg = precision 0.97 | recall 0.97 | f1-score 0.97
```

---

### 📊 **Model Accuracy Comparison**

![Model Accuracy Chart](screenshots/model_comparison.png)

> Logistic Regression slightly outperformed SVM and was chosen as the final production model due to stability and interpretability.

---

### 🧩 **Prediction Examples**

| Complaint (Shortened)                                                  | Predicted Category |
| ---------------------------------------------------------------------- | ------------------ |
| “I am being charged an incorrect fee on my mortgage payment.”          | Mortgage           |
| “This company keeps calling my family about a debt I don't recognize.” | Debt collection    |
| “I looked at my credit report and there is a fraudulent account open.” | Debt collection    |
| “I applied for a car loan and was denied without reason.”              | Consumer Loan      |

---

### 🚀 **How to Run**

#### 🧩 Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

#### ▶️ Execute

1. Open `kaiburr-task5-consumercomplaint-classification.ipynb`
2. Run all cells sequentially
3. Compare model outputs & accuracy chart
4. The best model is saved automatically as `best_complaint_model.pkl`

---

### 🧠 **Tools & Libraries**

| Purpose        | Library                         |
| -------------- | ------------------------------- |
| Data Handling  | `pandas`, `numpy`               |
| Model Training | `scikit-learn`                  |
| Visualization  | `matplotlib`                    |
| Model Saving   | `joblib`                        |
| Environment    | Jupyter Notebook / Python 3.11+ |

---

### 📄 **Result Summary**

> Logistic Regression achieved **96.99% accuracy**, outperforming other classifiers.
> It provides high precision and recall across all categories, making it the optimal choice for large-scale complaint classification.

---

### ✍️ **Author**

**Name:** YASWANTH KANCHARLA
---
