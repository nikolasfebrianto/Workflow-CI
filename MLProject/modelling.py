"""Manual MLflow training for UCI Breast Cancer Wisconsin classification.

Run examples:
    python modelling.py --data-dir breast_cancer_preprocessing

DagsHub example:
    export DAGSHUB_USERNAME="your_username"
    export DAGSHUB_REPO="your_repo"
    export DAGSHUB_TOKEN="your_token"
    python modelling.py --data-dir breast_cancer_preprocessing
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COLUMN = "is_malignant"
RANDOM_STATE = 42


def setup_mlflow(experiment_name: str) -> None:
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    dagshub_repo = os.getenv("DAGSHUB_REPO")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    if dagshub_username and dagshub_repo:
        tracking_uri = f"https://dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow"
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", dagshub_username)
        if dagshub_token:
            os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", dagshub_token)
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

    mlflow.set_experiment(experiment_name)


def load_dataset(data_dir: str | Path):
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"train.csv/test.csv tidak ditemukan di {data_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = pd.crosstab(y_test, y_pred, dropna=False).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy().ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_malignant": precision_score(y_test, y_pred, zero_division=0),
        "recall_malignant": recall_score(y_test, y_pred, zero_division=0),
        "specificity_benign": specificity,
        "f1_malignant": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
    }


def save_artifacts(model, X_test, y_test, artifact_dir: Path) -> dict[str, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax,
        colorbar=False,
        display_labels=["benign", "malignant"],
    )
    ax.set_title("Confusion Matrix - Breast Cancer")
    paths["confusion_matrix"] = artifact_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(paths["confusion_matrix"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC Curve - Breast Cancer")
    paths["roc_curve"] = artifact_dir / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(paths["roc_curve"], dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Malignant Class")
    paths["precision_recall_curve"] = artifact_dir / "precision_recall_curve.png"
    fig.tight_layout()
    fig.savefig(paths["precision_recall_curve"], dpi=160)
    plt.close(fig)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["benign", "malignant"],
        output_dict=True,
        zero_division=0,
    )
    report_path = artifact_dir / "classification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["classification_report"] = report_path

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_path = artifact_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    paths["feature_importance"] = importance_path

    top_feature_path = artifact_dir / "top_20_feature_importance.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    top20 = importance_df.head(20).sort_values("importance")
    ax.barh(top20["feature"], top20["importance"])
    ax.set_title("Top 20 Feature Importance")
    fig.tight_layout()
    fig.savefig(top_feature_path, dpi=160)
    plt.close(fig)
    paths["top_20_feature_importance"] = top_feature_path

    pred_sample = pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred,
        "probability_malignant": y_proba,
    }).head(100)
    pred_path = artifact_dir / "prediction_sample.csv"
    pred_sample.to_csv(pred_path, index=False)
    paths["prediction_sample"] = pred_path

    return paths


def train(args) -> dict:
    setup_mlflow(args.experiment_name)
    X_train, X_test, y_train, y_test = load_dataset(args.data_dir)

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "class_weight": args.class_weight,
        "random_state": RANDOM_STATE,
        "dataset": "breast_cancer_wisconsin_diagnostic_preprocessed",
        "source": "UCI Machine Learning Repository Dataset ID 17",
    }

    model = RandomForestClassifier(**{k: v for k, v in params.items() if k not in {"dataset", "source"}})
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    artifact_dir = Path(args.artifact_dir)
    artifact_paths = save_artifacts(model, X_test, y_test, artifact_dir)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", X_train.shape[1])

        for artifact_path in artifact_paths.values():
            mlflow.log_artifact(str(artifact_path), artifact_path.parent.name)

        metrics_path = artifact_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path), artifact_dir.name)

        signature = infer_signature(X_train.head(20), model.predict_proba(X_train.head(20)))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=args.registered_model_name if args.registered_model_name else None,
        )

        if args.model_output:
            output_path = Path(args.model_output)
            if output_path.exists():
                import shutil
                shutil.rmtree(output_path)
            mlflow.sklearn.save_model(
                sk_model=model,
                path=str(output_path),
                signature=signature,
                input_example=X_train.head(5),
            )
            mlflow.log_artifact(str(output_path), "saved_model_copy")

        print("Run ID:", run.info.run_id)
        print(json.dumps(metrics, indent=2))
        return {"run_id": run.info.run_id, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description="Train UCI Breast Cancer model with manual MLflow logging")
    parser.add_argument("--data-dir", default="breast_cancer_preprocessing")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--experiment-name", default="Breast_Cancer_Advanced_Manual_Logging")
    parser.add_argument("--run-name", default="random_forest_manual_logging")
    parser.add_argument("--registered-model-name", default="BreastCancerRandomForest")
    parser.add_argument("--model-output", default="")
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--class-weight", default="balanced")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
