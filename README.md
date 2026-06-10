# Hate Speech Detection using Machine Learning and Weighted DistilBERT

An NLP-based hate speech detection system that classifies social media posts into **Hate Speech**, **Offensive Language**, and **Neither** using Machine Learning and Transformer-based Deep Learning models.

---

## Project Highlights

* Processed and analyzed **24,783 social media posts**
* Compared **5 Machine Learning and Deep Learning models**
* Applied **class imbalance handling** using weighted loss
* Improved minority-class hate speech detection performance
* Built a real-time inference pipeline
* Developed a deployment-ready Streamlit application
* Achieved **90.7% Accuracy** and **0.776 Macro F1 Score**

---

## Problem Statement

Social media platforms generate millions of posts every day. Detecting harmful content manually is difficult, time-consuming, and challenging to scale.

The goal of this project is to automatically classify social media posts into:

* Hate Speech
* Offensive Language
* Neither

using Natural Language Processing (NLP) and Deep Learning techniques.

---

## Dataset

This project uses the Hate Speech and Offensive Language Dataset.

### Dataset Statistics

* Total Samples: **24,783 Tweets**
* Classes:

  * Hate Speech
  * Offensive Language
  * Neither

### Dataset Source

https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

---

## Data Availability

The raw and processed datasets are not included in this repository to keep the repository lightweight.

To reproduce this project:

1. Download the dataset from the Kaggle link above.
2. Create the following directories:

```text
raw_data/
processed_data/
```

3. Place the downloaded dataset inside:

```text
raw_data/
```

4. Run the preprocessing notebook to generate cleaned and processed data.

---

## Project Structure

```text
hate-speech-detection/
│
├── raw_data/                    # Downloaded dataset (not uploaded)
│
├── processed_data/              # Cleaned datasets (not uploaded)
│
├── models/
│   └── weighted_distilbert/
│       ├── config.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── notebooks/
│   ├── 01_Hate_Speech_Detection_Full_Pipeline.ipynb
│   └── inference.ipynb
│
├── results/
│   ├── experiment_log.csv
│   ├── model_comparison.csv
│   └── model_comparison.xlsx
│
├── streamlit_app/
│   └── app.py
│
├── README.md
└── .gitignore
```

### Folder Description

| Folder         | Purpose                                                         |
| -------------- | --------------------------------------------------------------- |
| raw_data       | Original dataset downloaded from Kaggle                         |
| processed_data | Cleaned and transformed datasets generated during preprocessing |
| models         | Tokenizer and model configuration files                         |
| notebooks      | Training, preprocessing, evaluation, and inference notebooks    |
| results        | Experiment logs and model comparison results                    |
| streamlit_app  | Deployment code for the web application                         |

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
Classical ML Models
(Logistic Regression, Random Forest, XGBoost)
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

---

## Data Preprocessing

The following preprocessing steps were applied:

* Lowercasing
* URL Removal
* Mention Removal
* Special Character Removal
* Contraction Expansion
* Whitespace Normalization

### Additional Engineered Features

* Message Length
* Word Count
* Capital Letter Ratio

---

## Models Implemented

### Classical Machine Learning Models

1. Logistic Regression
2. Random Forest
3. XGBoost

### Transformer-Based Deep Learning Models

4. DistilBERT
5. Weighted DistilBERT

---

## Results

| Model               | Accuracy | Macro F1 | Balanced Accuracy |
| ------------------- | -------: | -------: | ----------------: |
| Logistic Regression |    0.850 |    0.720 |             0.807 |
| Random Forest       |    0.880 |    0.650 |             0.623 |
| XGBoost             |    0.900 |    0.700 |             0.704 |
| DistilBERT          |    0.918 |    0.744 |             0.721 |
| Weighted DistilBERT |    0.907 |    0.776 |             0.798 |

---

## Best Model

### Weighted DistilBERT

Performance:

* Accuracy: **90.7%**
* Macro F1 Score: **0.776**
* Balanced Accuracy: **0.798**

### Why It Was Selected

Although DistilBERT achieved the highest overall accuracy, Weighted DistilBERT significantly improved minority-class hate speech detection by addressing class imbalance through weighted loss.

This resulted in a better balance between overall performance and fairness across classes.

---

## Example Predictions

### Example 1

**Input**

```text
Today is a beautiful day
```

**Prediction**

```text
Neither
```

**Confidence**

```text
97.8%
```

---

### Example 2

**Input**

```text
What the fuck are you doing?
```

**Prediction**

```text
Offensive Language
```

**Confidence**

```text
92.0%
```

---

## Model Access

The complete trained model weights are hosted separately due to GitHub file size limitations.

### Hugging Face Model


```text
https://huggingface.co/nsaichandana/hate-speech-weighted-distilbert
```

---

## Repository Structure

```text
hate-speech-detection/
│
├── models/
│   └── weighted_distilbert/
│
├── notebooks/
│
├── results/
│
├── streamlit_app/
│
├── README.md
└── .gitignore
```

---

## Future Work

* Streamlit Cloud Deployment
* Hugging Face Integration
* Explainable AI (XAI)
* Multi-Language Hate Speech Detection
* Real-Time Social Media Monitoring

---

## Author

### Nunna Saichandana

B.Tech Computer Science Engineering

Sathyabama Institute of Science and Technology

LinkedIn:
https://www.linkedin.com/in/nunna-saichandana-8ab188326/
