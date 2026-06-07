"""
Train the DDoS detection model and save deployment artifacts.

Usage:
    python train.py --dataset datasets/dataset.csv

Outputs:
    model/model.pkl
    model/scaler.pkl
    model/features.pkl
    model/metrics.pkl
    model/feature_importance.csv
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, average_precision_score


def load_and_preprocess(dataset_path: str):
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.drop_duplicates(inplace=True)

    if "Label" not in df.columns:
        raise ValueError("Dataset must contain a Label column with BENIGN and DDoS values.")

    df["Label"] = df["Label"].map({"BENIGN": 0, "DDoS": 1})
    df.dropna(subset=["Label"], inplace=True)
    df["Label"] = df["Label"].astype(int)

    X = df.drop("Label", axis=1).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = df["Label"]
    return X, y


def build_model():
    rf_model = RandomForestClassifier(
        n_estimators=80,
        max_depth=25,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    et_model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=25,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    hgb_model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42
    )
    return VotingClassifier(
        estimators=[("rf", rf_model), ("et", et_model), ("hgb", hgb_model)],
        voting="soft",
        n_jobs=-1
    )


def main(dataset_path: str):
    os.makedirs("model", exist_ok=True)
    X, y = load_and_preprocess(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, test_size=0.25, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": auc(fpr, tpr),
        "average_precision": average_precision_score(y_test, y_prob),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "total_features": int(X.shape[1])
    }

    rf_importance = model.named_estimators_["rf"].feature_importances_
    et_importance = model.named_estimators_["et"].feature_importances_
    avg_importance = (rf_importance + et_importance) / 2
    feature_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": avg_importance
    }).sort_values(by="Importance", ascending=False)

    joblib.dump(model, "model/model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(list(X.columns), "model/features.pkl")
    joblib.dump(metrics, "model/metrics.pkl")
    feature_importance_df.to_csv("model/feature_importance.csv", index=False)

    print("Training complete. Saved files in model/ folder.")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/dataset.csv", help="Path to training CSV file")
    args = parser.parse_args()
    main(args.dataset)
