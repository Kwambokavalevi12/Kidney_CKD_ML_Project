"""
Kidney CKD classification project
Downloads public NHANES August 2021-August 2023 XPT files, creates
survey-based kidney-status groups, trains supervised models and saves metrics.

IMPORTANT:
- This is an educational research prototype, NOT a clinical diagnostic tool.
- NHANES is cross-sectional. One measurement cannot prove CKD persistence >=3 months.
- Creatinine/eGFR and urine albumin-creatinine ratio are used ONLY to define the
  research outcome and are excluded from predictors to prevent target leakage.
"""
from pathlib import Path
import warnings
import urllib.request
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results"
MODELS = ROOT / "models"
for folder in (RAW, OUT, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/"
FILES = ["DEMO_L", "BIOPRO_L", "ALB_CR_L", "BMX_L", "BPXO_L",
         "DIQ_L", "SMQ_L", "PAQ_L", "ALQ_L", "GHB_L", "BPQ_L"]

def fetch_xpt(stem):
    path = RAW / f"{stem}.xpt"
    if not path.exists() or path.stat().st_size < 1000:
        url = BASE_URL + f"{stem}.xpt"
        print(f"Downloading {stem} from CDC...")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=90) as response, open(path, "wb") as f:
            f.write(response.read())
    return pd.read_sas(path, format="xport", encoding="utf-8")

def safe_mean(df, cols):
    present = [c for c in cols if c in df.columns]
    return df[present].mean(axis=1) if present else np.nan

def merge_nhanes():
    frames = {}
    for stem in FILES:
        try:
            frames[stem] = fetch_xpt(stem)
            print(f"  {stem}: {len(frames[stem]):,} records")
        except Exception as exc:
            # Only optional questionnaire/examination files may be skipped.
            if stem in ("DEMO_L", "BIOPRO_L", "ALB_CR_L"):
                raise RuntimeError(f"Required CDC file {stem} could not be loaded: {exc}") from exc
            warnings.warn(f"Optional file {stem} skipped: {exc}")

    df = frames["DEMO_L"].copy()
    for stem, frame in frames.items():
        if stem == "DEMO_L":
            continue
        keep = [c for c in frame.columns if c == "SEQN"]
        # retain only variables needed from each component
        desired = {
            "BIOPRO_L": ["LBXSCR", "LBXSGL", "LBXSBU", "LBXSUA"],
            "ALB_CR_L": ["URDACT"],
            "BMX_L": ["BMXBMI", "BMXWAIST"],
            "BPXO_L": ["BPXOSY1","BPXOSY2","BPXOSY3","BPXODI1","BPXODI2","BPXODI3"],
            "DIQ_L": ["DIQ010"],
            "SMQ_L": ["SMQ020","SMQ040"],
            "PAQ_L": ["PAD680","PAQ605","PAQ620"],
            "ALQ_L": ["ALQ111","ALQ121","ALQ130"],
            "GHB_L": ["LBXGH"],
            "BPQ_L": ["BPQ020"],
        }.get(stem, [])
        keep += [c for c in desired if c in frame.columns]
        if len(keep) > 1:
            df = df.merge(frame[keep].drop_duplicates("SEQN"), on="SEQN", how="left")

    return df

def add_outcome_and_features(df):
    required = ["RIDAGEYR", "RIAGENDR", "LBXSCR", "URDACT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing essential NHANES variables: " + ", ".join(missing))

    # CKD-EPI 2021 race-free creatinine equation; serum creatinine in mg/dL.
    scr = pd.to_numeric(df["LBXSCR"], errors="coerce")
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    female = df["RIAGENDR"].eq(2)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr / kappa
    egfr = 142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** (-1.200) * (0.9938 ** age)
    egfr = egfr * np.where(female, 1.012, 1.0)
    uacr = pd.to_numeric(df["URDACT"], errors="coerce")

    # Three project classes. These are cross-sectional survey categories,
    # not confirmed clinical CKD stages.
    valid = egfr.notna() & uacr.notna() & age.ge(18) & scr.gt(0) & uacr.ge(0)
    label = pd.Series(index=df.index, dtype="object")
    advanced = egfr.lt(45)
    early = ((egfr.ge(45) & egfr.lt(60)) | (egfr.ge(60) & uacr.ge(30)))
    no_evidence = egfr.ge(60) & uacr.lt(30)
    label.loc[valid & advanced] = "Advanced kidney impairment"
    label.loc[valid & early] = "Early kidney abnormality"
    label.loc[valid & no_evidence] = "No kidney abnormality detected"
    df = df.loc[valid & label.notna()].copy()
    df["target"] = label.loc[df.index]

    # Feature engineering. Kidney label variables (creatinine, eGFR, UACR)
    # are deliberately not used as predictors.
    df["mean_systolic_bp"] = safe_mean(df, ["BPXOSY1","BPXOSY2","BPXOSY3"])
    df["mean_diastolic_bp"] = safe_mean(df, ["BPXODI1","BPXODI2","BPXODI3"])

    def yes_no(series, yes_values=(1,), no_values=(2,)):
        s = pd.to_numeric(series, errors="coerce")
        return s.map(lambda x: "Yes" if x in yes_values else ("No" if x in no_values else np.nan))

    df["diabetes_reported"] = yes_no(df["DIQ010"]) if "DIQ010" in df else np.nan
    df["hypertension_reported"] = yes_no(df["BPQ020"]) if "BPQ020" in df else np.nan
    df["ever_smoked_100"] = yes_no(df["SMQ020"]) if "SMQ020" in df else np.nan
    df["current_smoker"] = pd.to_numeric(df["SMQ040"], errors="coerce").map(
        {1:"Every day", 2:"Some days", 3:"Not at all"}) if "SMQ040" in df else np.nan
    df["sex"] = df["RIAGENDR"].map({1:"Male", 2:"Female"})
    df["race_ethnicity"] = df["RIDRETH3"].astype("Int64").astype(str) if "RIDRETH3" in df else "Unknown"
    df["age"] = age
    df["bmi"] = pd.to_numeric(df["BMXBMI"], errors="coerce") if "BMXBMI" in df else np.nan
    df["hba1c"] = pd.to_numeric(df["LBXGH"], errors="coerce") if "LBXGH" in df else np.nan
    df["glucose"] = pd.to_numeric(df["LBXSGL"], errors="coerce") if "LBXSGL" in df else np.nan
    df["blood_urea_nitrogen"] = pd.to_numeric(df["LBXSBU"], errors="coerce") if "LBXSBU" in df else np.nan
    df["sedentary_minutes"] = pd.to_numeric(df["PAD680"], errors="coerce") if "PAD680" in df else np.nan
    df["vigorous_work_activity"] = yes_no(df["PAQ605"]) if "PAQ605" in df else np.nan
    df["moderate_work_activity"] = yes_no(df["PAQ620"]) if "PAQ620" in df else np.nan
    df["alcohol_ever"] = yes_no(df["ALQ111"]) if "ALQ111" in df else np.nan
    df["mean_systolic_bp"] = pd.to_numeric(df["mean_systolic_bp"], errors="coerce")
    df["mean_diastolic_bp"] = pd.to_numeric(df["mean_diastolic_bp"], errors="coerce")

    numeric = ["age","bmi","hba1c","glucose","blood_urea_nitrogen",
               "mean_systolic_bp","mean_diastolic_bp","sedentary_minutes"]
    categorical = ["sex","race_ethnicity","diabetes_reported","hypertension_reported",
                   "ever_smoked_100","current_smoker","vigorous_work_activity",
                   "moderate_work_activity","alcohol_ever"]
    feature_cols = [c for c in numeric + categorical if c in df.columns and df[c].notna().any()]
    data = df[feature_cols + ["target"]].copy()
    # Exclude impossible values; retain missing values for pipeline imputation.
    for col, lo, hi in [("age",18,120),("bmi",10,90),("hba1c",2,20),
                        ("glucose",20,800),("mean_systolic_bp",60,300),
                        ("mean_diastolic_bp",30,200),("sedentary_minutes",0,1439)]:
        if col in data:
            data.loc[~data[col].between(lo,hi), col] = np.nan
    return data, numeric, categorical

def main():
    print("\n=== NHANES CKD classification project ===")
    raw = merge_nhanes()
    data, numeric, categorical = add_outcome_and_features(raw)
    data.to_csv(OUT / "modeling_dataset.csv", index=False)
    print(f"\nEligible adult records with outcome: {len(data):,}")
    print("Class counts:\n", data["target"].value_counts().to_string())
    if data["target"].nunique() < 2:
        raise ValueError("Fewer than two outcome classes. Check source data/labels.")

    X = data.drop(columns="target")
    y = data["target"]
    X_train, X_test, y_train_text, y_test_text = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_test = label_encoder.transform(y_test_text)
    numeric_features = [c for c in numeric if c in X.columns]
    categorical_features = [c for c in categorical if c in X.columns]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=8,
                                                  class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=3,
                                                 class_weight="balanced_subsample",
                                                 random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                                 subsample=0.85, colsample_bytree=0.85,
                                 objective="multi:softprob", eval_metric="mlogloss",
                                 random_state=42, n_jobs=-1)
    }

    results = []
    fitted = {}
    for name, estimator in models.items():
        print(f"\nTraining {name}...")
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0)
        }
        try:
            proba = pipe.predict_proba(X_test)
            row["roc_auc_ovr_macro"] = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        except Exception:
            row["roc_auc_ovr_macro"] = np.nan
        results.append(row)
        fitted[name] = pipe
        print(row)

    scores = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    scores.to_csv(OUT / "model_comparison.csv", index=False)
    best_name = scores.iloc[0]["model"]
    best = fitted[best_name]
    pred = best.predict(X_test)

    report = classification_report(y_test, pred, labels=np.arange(len(label_encoder.classes_)),
                                   target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(OUT / "classification_report.csv")
    (OUT / "run_summary.json").write_text(json.dumps({
        "records_after_labeling": int(len(data)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "class_counts": {str(k): int(v) for k,v in y.value_counts().items()},
        "best_model": best_name,
        "selection_metric": "macro F1-score",
        "best_model_test_metrics": scores.iloc[0].to_dict(),
        "important_note": "Single-visit NHANES categories are not confirmed CKD diagnoses."
    }, indent=2, default=float), encoding="utf-8")

    labels = list(label_encoder.classes_)
    cm = confusion_matrix(y_test, pred, labels=np.arange(len(labels)))
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9,5))
    plot_df = scores.set_index("model")[["accuracy","precision_macro","recall_macro","f1_macro"]]
    plot_df.plot(kind="bar", ylim=(0,1), rot=20)
    plt.ylabel("Score")
    plt.title("Test-set model comparison")
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=180)
    plt.close()

    # Permutation importance is computed on held-out data for model interpretation.
    print("\nCalculating permutation importance...")
    perm = permutation_importance(best, X_test, y_test, n_repeats=8,
                                  random_state=42, scoring="f1_macro", n_jobs=-1)
    importance = pd.DataFrame({"feature": X.columns,
                               "importance_mean": perm.importances_mean,
                               "importance_std": perm.importances_std}).sort_values(
                                   "importance_mean", ascending=False)
    importance.to_csv(OUT / "permutation_importance.csv", index=False)
    plt.figure(figsize=(9,6))
    top = importance.head(12).sort_values("importance_mean")
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Decrease in macro F1 after permutation")
    plt.title("Permutation feature importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=180)
    plt.close()

    joblib.dump(best, MODELS / "best_ckd_classifier.joblib")
    print("\nSUCCESS. Results saved in:", OUT)
    print("Best model:", best_name)
    print(scores.to_string(index=False))
    print("\nDo not describe this prototype as a clinically validated diagnostic system.")

if __name__ == "__main__":
    main()
