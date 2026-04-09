# Transaction Fraud Detection

A machine learning project to classify financial transactions as fraudulent or legitimate, built around a performance-based business model where revenue depends directly on model accuracy.

---

## Business problem

Blocker Fraud Company offers a fraud detection service with the following revenue structure:

- **+25%** of the value of each transaction correctly identified as fraud
- **+5%** of the value of each transaction incorrectly flagged as fraud
- **−100%** refund for each fraudulent transaction missed by the model

This means the company profits from correct detections and absorbs the cost of misses — making model precision and recall the core business levers.

---

## Solution strategy

1. **Data description** — study the raw data, handle missing values, compute descriptive statistics
2. **Feature engineering** — create new features guided by domain hypotheses
3. **Data filtering** — remove columns and rows irrelevant to the business problem
4. **Exploratory data analysis** — univariate, bivariate, and multivariate analysis; hypothesis testing
5. **Data preparation** — encoding, rescaling, and resampling for model training
6. **Feature selection** — use Boruta to identify the most predictive features
7. **Model training** — train and cross-validate multiple algorithms
8. **Hyperparameter tuning** — optimize the best-performing model
9. **Business evaluation** — test on unseen data and translate results into financial impact
10. **Deployment** — serve predictions via a Flask API

---

## Models evaluated (cross-validation)

| Model | Balanced accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Dummy | 0.499 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Logistic regression | 0.565 +/- 0.009 | 1.000 +/- 0.000 | 0.129 +/- 0.017 | 0.229 +/- 0.027 |
| K-nearest neighbors | 0.705 +/- 0.037 | 0.942 +/- 0.022 | 0.409 +/- 0.074 | 0.568 +/- 0.073 |
| Support vector machine | 0.595 +/- 0.013 | 1.000 +/- 0.000 | 0.190 +/- 0.026 | 0.319 +/- 0.037 |
| Random forest | 0.865 +/- 0.017 | 0.972 +/- 0.014 | 0.731 +/- 0.033 | 0.834 +/- 0.022 |
| **XGBoost** | **0.880 +/- 0.016** | **0.963 +/- 0.008** | **0.761 +/- 0.033** | **0.850 +/- 0.023** |
| LightGBM | 0.701 +/- 0.089 | 0.180 +/- 0.100 | 0.407 +/- 0.175 | 0.241 +/- 0.128 |

---

## Final model performance

XGBoost was selected and tuned. Results on unseen data:

| Balanced accuracy | Precision | Recall | F1 | Kappa |
|---|---|---|---|---|
| 0.915 | 0.944 | 0.829 | 0.883 | 0.883 |

---

## Business results

| | Value |
|---|---|
| Revenue from correct fraud detections (25%) | R$ 60,613,782.88 |
| Revenue from false positives (5%) | R$ 183,866.98 |
| Refunds for missed fraud (−100%) | R$ 3,546,075.42 |
| **Net profit with model** | **R$ 57,251,574.44** |
| Net result without model | R$ −246,001,206.94 |

---

## Tech stack

- Python
- Scikit-learn, XGBoost
- Pandas, NumPy
- Boruta
- Flask

---

## Next steps

- Test additional hypotheses from the EDA phase
- Experiment with oversampling and subsampling techniques to improve recall
- Deploy the Flask API to a cloud platform

---

## License

MIT — see [LICENSE](LICENSE) for details.
