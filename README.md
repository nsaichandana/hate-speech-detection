# Hate Speech Detection using Weighted DistilBERT

An end-to-end NLP project that detects and classifies social media text into **Hate Speech**, **Offensive Language**, and **Neither** using Machine Learning and Transformer-based Deep Learning models.

## Live Demo

🚀 Streamlit App: https://hate-speechdetection.streamlit.app/

🤗 Hugging Face Model: https://huggingface.co/nsaichandana/hate-speech-weighted-distilbert

---

## Project Overview

Social media platforms process millions of posts every day, making manual moderation difficult and time-consuming.

This project leverages Natural Language Processing (NLP) and Deep Learning to automatically identify harmful content and classify it into:

* Hate Speech
* Offensive Language
* Neither

---

## Key Highlights

* Processed and analyzed **24,783 social media posts**
* Compared **5 Machine Learning and Deep Learning models**
* Applied **Weighted Loss** to handle severe class imbalance
* Improved minority-class hate speech detection performance
* Built a real-time text classification system
* Developed and deployed a Streamlit web application
* Hosted the trained model on Hugging Face

### Best Model Performance

| Metric            | Score |
| ----------------- | ----- |
| Accuracy          | 90.7% |
| Macro F1 Score    | 0.776 |
| Balanced Accuracy | 0.798 |

---

## Dataset

### Hate Speech and Offensive Language Dataset

Dataset Size: **24,783 Tweets**

Class Distribution:

| Class              | Samples |
| ------------------ | ------- |
| Hate Speech        | 1,430   |
| Offensive Language | 19,190  |
| Neither            | 4,163   |

Dataset Source:

https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

---

## Project Pipeline

```text
Raw Tweets
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Train/Test Split
    ↓
Machine Learning Models
    ↓
DistilBERT
    ↓
Weighted DistilBERT
    ↓
Evaluation & Comparison
    ↓
Inference System
    ↓
Streamlit Deployment
```

## Data Preprocessing

The following preprocessing steps were applied:

* Lowercasing
* URL Removal
* Mention Removal
* Special Character Removal
* Contraction Expansion
* Whitespace Normalization

Additional engineered features:

* Message Length
* Word Count
* Capital Letter Ratio

---

## Models Evaluated

### Machine Learning Models

* Logistic Regression
* Random Forest
* XGBoost

### Transformer Models

* DistilBERT
* Weighted DistilBERT

---

## Model Comparison

| Model               | Accuracy | Macro F1 |
| ------------------- | -------- | -------- |
| Logistic Regression | 0.850    | 0.720    |
| Random Forest       | 0.880    | 0.650    |
| XGBoost             | 0.900    | 0.700    |
| DistilBERT          | 0.918    | 0.744    |
| Weighted DistilBERT | 0.907    | 0.776    |

---

## Why Weighted DistilBERT?

Although DistilBERT achieved the highest overall accuracy, Weighted DistilBERT significantly improved performance on the minority Hate Speech class.

By incorporating class-weighted loss, the model achieved better balance across all classes while maintaining strong overall performance.

---

## Example Predictions

### Example 1

Input:

```text
Today is a beautiful day.
```

Prediction:

```text
Neither
```

### Example 2

Input:

```text
You are such an idiot.
```

Prediction:

```text
Offensive Language
```

### Example 3

Input:

```text
We need to kill all immigrants in this city.
```

Prediction:

```text
Hate Speech
```

---

## Repository Structure

```text
hate-speech-detection/
│
├── models/
├── notebooks/
├── results/
├── streamlit_app/
├── README.md
└── .gitignore
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* Streamlit
* Google Colab

---

## Future Improvements

* Explainable AI (XAI)
* Multi-language hate speech detection
* Larger and more balanced datasets
* DeBERTa-based architecture
* Real-time social media monitoring

---

## Author

**Nunna Saichandana**

B.Tech Computer Science Engineering
Sathyabama Institute of Science and Technology

LinkedIn:
https://www.linkedin.com/in/nunna-saichandana-8ab188326/
