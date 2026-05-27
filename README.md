# Fraud Detection using Self-Organizing Maps and ANN

A two-phase hybrid model for detecting fraudulent credit card applications — first using an unsupervised **Self-Organizing Map (SOM)** to identify anomalies, then training a supervised **Artificial Neural Network (ANN)** to score every customer by their fraud probability.

## Problem Statement

Traditional rule-based fraud detection systems struggle with novel fraud patterns. This project uses unsupervised learning to detect outliers in credit card application data — customers whose profiles don't fit any legitimate pattern — and then builds a supervised model to rank all customers by their likelihood of fraud.

## Dataset

**File:** `Credit_Card_Applications.csv`

Each row represents a credit card application with 15 anonymised customer attributes (features 0–14) and a final column indicating whether the application was **approved (1)** or **not approved (0)**.

## Methodology

### Phase 1 — Unsupervised Fraud Detection (SOM)

A **Self-Organizing Map** is trained on the scaled application data to learn the topology of legitimate vs. anomalous applications.

- **SOM Grid:** 10 × 10
- **Input Length:** 15 features
- **Sigma:** 1.0 (neighbourhood radius)
- **Learning Rate:** 0.5
- **Iterations:** 100

After training, the **distance map (U-Matrix)** is visualised using a colour-coded heatmap. Neurons with high mean inter-neuron distances (bright regions) represent outlier clusters — likely fraudulent applications. Customers mapped to those outlier neurons are extracted as suspected frauds.

### Phase 2 — Supervised Fraud Scoring (ANN)

The fraud labels discovered by the SOM are used as a target variable to train a small ANN, enabling a probability score for every customer.

**ANN Architecture:**
```
Input Layer  →  14 customer features
Hidden Layer →  2 neurons, ReLU activation
Output Layer →  1 neuron, Sigmoid activation (fraud probability)
```

**Training:**
- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Batch Size:** 32
- **Epochs:** 5

The final output is a list of all customers sorted by their predicted fraud probability.

## Key Insight

This project demonstrates a practical **unsupervised → supervised pipeline**: the SOM provides weak labels from raw, unlabelled data, which are then used to train a more interpretable scoring model. This is a common real-world approach when labelled fraud data is scarce.

## Requirements

```
numpy
pandas
matplotlib
minisom
tensorflow
scikit-learn
```

Install with:
```bash
pip install numpy pandas matplotlib minisom tensorflow scikit-learn
```

## Usage

```bash
python Fraud_Detection.py
```

Ensure `Credit_Card_Applications.csv` is in the same directory as the script.

## Output

- A visual U-Matrix heatmap with customer markers (red circles = rejected applications, green squares = approved)
- A printed list of suspected fraud **Customer IDs** from the SOM phase
- A ranked table of all customers sorted by **predicted fraud probability** from the ANN
