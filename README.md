# Credit Card Fraud Detection with Unsupervised Learning

This project explores **unsupervised anomaly detection techniques** for identifying fraudulent credit card transactions using the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset is highly imbalanced, with fraud making up only **0.172%** of the transactions. Our goal is to detect anomalies **without using class labels during training**, and evaluate post-hoc using appropriate metrics.

---

## Models Implemented

### LSTM Autoencoder
- Learns temporal patterns in transaction features
- Uses reconstruction error to flag anomalies

### Dense Autoencoder
- Faster fully-connected version of LSTM autoencoder
- Suitable for tabular anomaly detection, but less powerful than LSTM

### One-Class SVM
- Learns a decision boundary around normal transactions
- Sensitive to parameter tuning and scale

### Isolation Forest ✅ (Best performer)
- Tree-based ensemble that isolates anomalies quickly
- Tuned using `contamination`, `n_estimators`, and `max_samples`

---

## Evaluation Metrics

We used **Precision**, **Recall**, **F1 Score**, and most importantly:

### AUPRC — Area Under the Precision-Recall Curve

| Metric     | Description                                |
|------------|--------------------------------------------|
| Precision  | How many flagged transactions were fraud   |
| Recall     | How many frauds were caught                |
| F1 Score   | Balance of precision and recall            |
| AUPRC      | Best for imbalanced classification problems|

> A random classifier would have an AUPRC of ~0.0017 (fraud rate). Our best model achieved **0.2595 AUPRC**, which is over 150x better.

---

## File Overview

| File | Description |
|------|-------------|
| `anomaly_detection_final_project.py` | Main training & evaluation script |
| `creditcard.csv` (not included) | Kaggle dataset, [Download here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| `README.md` | You're reading it! |
| `requirements.txt` |  Dependencies list for pip or Colab |

---

## Visuals Included
- t-SNE projections of fraud vs normal
- Anomaly score distributions
- Precision-Recall curves for each model
- Confusion matrix comparisons

---

##  How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/credit-card-fraud-unsupervised.git
cd credit-card-fraud-unsupervised

# (Optional) Set up virtual environment
pip install -r requirements.txt

# Run main script
python anomaly_detection_final_project.py
