# Business Failure Predictor

**Machine learning model that predicts company bankruptcy risk using financial metrics.**

---

## What This Does

Analyzes 18 financial ratios to predict if a company is likely to fail. Trained on 78,000+ American companies from 1999-2018.

Key features:
- 82.6% AUC score (strong predictive power)
- Identifies most important financial warning signs
- Provides risk scores from 0-100%
- Random Forest classifier

---

## Results

Model Performance:
- AUC: 0.8262
- Accuracy: 92%
- Trained on 62,945 companies
- Tested on 15,737 companies

Top Risk Indicators:
- X8 (most important financial ratio)
- X6 (second most important)
- X15, X3, X11 (also significant)

---

## How to Run

Train the model:
```bash
python train_model.py
```

Make predictions:
```bash
python predict.py
```

---

## What's Inside

- `train_model.py` - Trains Random Forest and Gradient Boosting models
- `predict.py` - Makes predictions on new companies
- `data/` - 78,682 company records with financial metrics
- `visualizations/` - 4 charts showing model performance
- `model/` - Saved model, scaler, and metadata

---

## Model Details

Algorithm: Random Forest Classifier
- 100 trees
- Max depth: 15
- Balanced class weights
- Handles imbalanced data (6.6% failure rate)

Features: 18 financial ratios (X1-X18)
- Likely include: debt ratios, profitability metrics, liquidity ratios, asset turnover

---

## Visualizations

Created 4 charts with neon orange styling:
1. Feature Importance - Shows which metrics matter most
2. Confusion Matrix - Model accuracy breakdown
3. ROC Curve - Predictive power visualization
4. Risk Distribution - How scores separate healthy vs failing companies

---

## Real-World Use

This model could help:
- Banks assess loan risk
- Investors identify troubled companies
- Business owners monitor their own health
- Accountants flag at-risk clients

Inspired by patterns observed in construction/contracting where overextension and budget issues signal trouble.

---

## Limitations

- Based on 1999-2018 data (pre-COVID)
- Financial ratios only (no industry context)
- US companies (may not generalize globally)
- Can't predict black swan events

Use as one tool among many for risk assessment.

---

## Built With

- Python
- scikit-learn - Machine learning
- pandas - Data processing
- matplotlib - Visualizations

---

## Author

Monse Rojo
- Portfolio: monserojo.com
- GitHub: @cyberyimi
- LinkedIn: linkedin.com/in/monse-rojo-6b70b3397/

---

Built with machine learning and financial data.
