# Customer Churn Prediction Using Data Mining

Predict which telecom customers are likely to cancel their service, and find out why, using the public IBM Telco Customer Churn dataset (7,043 customers, 21 columns). Four classifiers (logistic regression, random forest, XGBoost, RBF-kernel SVM) are compared with and without SMOTE oversampling, scored with recall, precision, F1 and ROC-AUC, and the final model is explained with SHAP.

The full write-up is in [`report/Customer_Churn_Prediction_Project_Report.pdf`](report/Customer_Churn_Prediction_Project_Report.pdf).

## Headline result

Logistic regression without resampling had the best cross-validated ROC-AUC (0.846) and scored **ROC-AUC 0.842** on the held-out test set. At a decision threshold of 0.32, chosen on training data only, it catches **73% of churners at 53% precision**.

## Repository layout

```
churn-prediction-project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md            how to download the dataset (file not included)
├── src/
│   └── churn_pipeline.py    cleaning, CV comparison, final model, SHAP
├── results/
│   └── results.json         raw output of the run reported below
└── report/
    └── Customer_Churn_Prediction_Project_Report.pdf
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put Telco-Customer-Churn.csv in data/ (see data/README.md), then:
python src/churn_pipeline.py
```

By default the script reads `data/Telco-Customer-Churn.csv` and writes `results/results.json`. You can pass other paths: `python src/churn_pipeline.py <input.csv> <output.json>`. The run takes a few minutes on a laptop because of the SVM and the 16 cross-validation fits.

## Method in brief

1. **Cleaning.** `TotalCharges` is text in the raw file; its 11 blank values all belong to customers with 0 months of tenure and are set to 0. `customerID` is dropped.
2. **Split.** Stratified 80/20 train/test split (5,634 / 1,409 customers; 374 churners in the test set), seed 42.
3. **Preprocessing.** Standard-scale the numeric columns, one-hot encode the categorical columns.
4. **Model comparison.** 5-fold stratified cross-validation on the training set for each of four models, with no resampling and with SMOTE. SMOTE sits inside the pipeline, so it only touches training folds and never leaks into validation data.
5. **Final model.** The combination with the highest cross-validated ROC-AUC is refit on the full training set. The decision threshold is picked by maximising F1 on out-of-fold training predictions, then the model is scored once on the test set.
6. **Explanation.** Mean absolute SHAP values on the test set.

## Results

All numbers come from `results/results.json`, produced by `src/churn_pipeline.py` with scikit-learn 1.9.1, imbalanced-learn 0.14.2, XGBoost 3.4.1 and SHAP 0.54.1.

### Cross-validation (training set, 5-fold, threshold 0.5)

| Model | Resampling | Recall | Precision | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | None | 0.544 | 0.652 | 0.592 | 0.846 |
| Logistic Regression | SMOTE | 0.788 | 0.526 | 0.630 | 0.845 |
| Random Forest | None | 0.486 | 0.627 | 0.547 | 0.820 |
| Random Forest | SMOTE | 0.557 | 0.589 | 0.573 | 0.819 |
| XGBoost | None | 0.515 | 0.612 | 0.559 | 0.825 |
| XGBoost | SMOTE | 0.573 | 0.590 | 0.581 | 0.822 |
| SVM (RBF) | None | 0.498 | 0.681 | 0.575 | 0.799 |
| SVM (RBF) | SMOTE | 0.718 | 0.552 | 0.624 | 0.827 |

Recall, precision and F1 are for the churn class. SMOTE raises recall by pushing models to flag more customers, and lowers precision to match; it does not improve ROC-AUC for any model except the SVM.

### Final model on the held-out test set

Selected model: logistic regression, no resampling.

| Item | Threshold 0.5 | Threshold 0.32 (tuned) |
| --- | --- | --- |
| Recall (churn) | 0.559 | 0.730 |
| Precision (churn) | 0.657 | 0.533 |
| F1-score | 0.604 | 0.616 |
| ROC-AUC (threshold-free) | 0.842 | 0.842 |
| True positives / false negatives | 209 / 165 | 273 / 101 |
| False positives / true negatives | 109 / 926 | 239 / 796 |

### Churn rate by group (full dataset, overall 26.5%)

| Group | Churn rate |
| --- | --- |
| Month-to-month contract | 42.7% |
| One-year contract | 11.3% |
| Two-year contract | 2.8% |
| Tenure 0-12 months | 47.4% |
| Tenure 13-24 months | 28.7% |
| Tenure 25-48 months | 20.4% |
| Tenure 49+ months | 9.5% |
| Fibre-optic internet | 41.9% |
| DSL internet | 19.0% |
| No internet service | 7.4% |
| Electronic-check payment | 45.3% |
| Automatic bank transfer or credit card | 15.2% to 16.7% |
| No tech support (internet customers) | 41.6% |
| Tech support | 15.2% |
| No online security (internet customers) | 41.8% |
| Online security | 14.6% |

### Top 10 features by mean absolute SHAP value

| Rank | Feature | Mean abs. SHAP (log-odds) |
| --- | --- | --- |
| 1 | Tenure (months) | 1.123 |
| 2 | Monthly charges | 0.514 |
| 3 | Total charges | 0.437 |
| 4 | Fibre-optic internet | 0.324 |
| 5 | Month-to-month contract | 0.292 |
| 6 | DSL internet | 0.267 |
| 7 | Two-year contract | 0.257 |
| 8 | No paperless billing | 0.163 |
| 9 | No multiple phone lines | 0.136 |
| 10 | Has dependents | 0.111 |

Tenure, monthly charges and total charges are closely related, so SHAP splits credit among them; read their ranks together.

## What the results suggest

- Move month-to-month customers onto longer contracts with discounts or bundled perks.
- Run an onboarding and check-in programme for customers in their first 12 months.
- Bundle security and tech support into fibre plans, and investigate fibre service quality.
- Encourage automatic payment with a small incentive.

## Limitations

- Models use scikit-learn defaults. No hyperparameter search, class-weight run or engineered features (such as tenure bands) were done, and the gaps between models are small, so the ranking could change with tuning.
- The dataset is a single undated snapshot of a fictional company, so it says nothing about seasonality or real-world transfer.
- There is no cost data, so the threshold is set by F1 and not by money saved.
- SMOTE on one-hot columns creates fractional category values; SMOTENC is worth testing.
- No charts (ROC curves, SHAP summary plot) are generated by the script yet.

## Ideas for future work

- Hyperparameter tuning with `RandomizedSearchCV`; class weights, SMOTENC and ADASYN.
- Cost-benefit threshold selection and probability calibration.
- Survival analysis on a dataset with dated records.
- A Streamlit dashboard listing the highest-risk customers with their top risk factors.

## Data and licence note

The dataset is not redistributed here; see `data/README.md` for where to download it and check its terms. Add a licence file for your own code before publishing (for example MIT).
