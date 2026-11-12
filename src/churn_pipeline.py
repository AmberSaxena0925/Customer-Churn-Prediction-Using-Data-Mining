"""Customer churn prediction on the IBM Telco dataset.

Usage: python src/churn_pipeline.py [input.csv] [output.json]
"""
import sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (recall_score, precision_score, f1_score, roc_auc_score,
                             confusion_matrix, precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

path = sys.argv[1] if len(sys.argv) > 1 else "data/Telco-Customer-Churn.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "results/results.json"
df = pd.read_csv(path)
n_blank = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
churn_rate = float((df["Churn"] == "Yes").mean())
n_churn = int((df["Churn"] == "Yes").sum())
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns="customerID")

# EDA numbers
eda = {
    "rows": len(df), "cols": df.shape[1] + 1, "n_churn": n_churn, "churn_rate": churn_rate,
    "blank_totalcharges": n_blank,
    "by_contract": df.groupby("Contract")["Churn"].mean().round(3).to_dict(),
    "by_internet": df.groupby("InternetService")["Churn"].mean().round(3).to_dict(),
    "by_payment": df.groupby("PaymentMethod")["Churn"].mean().round(3).to_dict(),
    "by_techsupport": df.groupby("TechSupport")["Churn"].mean().round(3).to_dict(),
    "by_security": df.groupby("OnlineSecurity")["Churn"].mean().round(3).to_dict(),
    "by_tenure_band": df.groupby(pd.cut(df["tenure"], [-1, 12, 24, 48, 100],
                                        labels=["0-12", "13-24", "25-48", "49+"]),
                                 observed=True)["Churn"].mean().round(3).to_dict(),
}

X, y = df.drop(columns="Churn"), df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in X.columns if c not in num_cols]
def prep():
    return ColumnTransformer([("num", StandardScaler(), num_cols),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])

def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(probability=True, random_state=42),
    }

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["recall", "precision", "f1", "roc_auc"]
rows = []
for rs in ["None", "SMOTE"]:
    for name, clf in models().items():
        steps = [("prep", prep())]
        if rs == "SMOTE":
            steps.append(("smote", SMOTE(random_state=42)))
        steps.append(("clf", clf))
        sc = cross_validate(Pipeline(steps), X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        r = {"model": name, "resampling": rs}
        r.update({m: float(sc[f"test_{m}"].mean()) for m in scoring})
        rows.append(r)
        print(r, flush=True)

# Final model: best by CV ROC-AUC among all combos, then threshold chosen on CV out-of-fold predictions (F1)
best = max(rows, key=lambda r: r["roc_auc"])
print("BEST", best, flush=True)
from sklearn.model_selection import cross_val_predict
steps = [("prep", prep())]
if best["resampling"] == "SMOTE":
    steps.append(("smote", SMOTE(random_state=42)))
steps.append(("clf", models()[best["model"]]))
pipe = Pipeline(steps)
oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
ths = np.linspace(0.1, 0.9, 81)
f1s = [f1_score(y_train, oof >= t) for t in ths]
th = float(ths[int(np.argmax(f1s))])
pipe.fit(X_train, y_train)
p = pipe.predict_proba(X_test)[:, 1]
def evalat(t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {"threshold": float(t), "recall": recall_score(y_test, pred), "precision": precision_score(y_test, pred),
            "f1": f1_score(y_test, pred), "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}
final = {"model": best["model"], "resampling": best["resampling"], "roc_auc": float(roc_auc_score(y_test, p)),
         "at_0.5": evalat(0.5), "at_tuned": evalat(th), "n_test": int(len(y_test)),
         "n_test_churn": int(y_test.sum())}

# Feature importance via SHAP-free permutation-ish: use SHAP if tree model, else LR coefficients
top = []
try:
    import shap
    Xt = pipe.named_steps["prep"].transform(X_test)
    if hasattr(Xt, "toarray"): Xt = Xt.toarray()
    names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if best["model"] in ("XGBoost", "Random Forest"):
        sv = shap.TreeExplainer(clf).shap_values(Xt)
        if isinstance(sv, list): sv = sv[1]
        sv = np.array(sv)
        if sv.ndim == 3: sv = sv[:, :, 1]
    elif best["model"] == "Logistic Regression":
        sv = shap.LinearExplainer(clf, pipe.named_steps["prep"].transform(X_train).toarray()
                                  if hasattr(pipe.named_steps["prep"].transform(X_train), "toarray")
                                  else pipe.named_steps["prep"].transform(X_train)).shap_values(Xt)
    else:
        raise RuntimeError("no fast explainer")
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(-imp)[:10]
    top = [(str(names[i]), float(imp[i])) for i in order]
except Exception as e:
    print("SHAP failed:", e)
final["top_features"] = top

json.dump({"eda": eda, "cv": rows, "final": final}, open(out, "w"), indent=2)
print("DONE")
