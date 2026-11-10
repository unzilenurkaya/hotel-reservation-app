"""Evaluate grouped splits and save the selected fitted pipeline."""

import hashlib
import json
import os
import platform
import tempfile
from importlib.metadata import version

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from reservation import (
    ARTIFACT_PATH,
    ARTIFACT_VERSION,
    CATEGORICAL,
    DATA_PATH,
    FEATURES,
    LABELS,
    NUMERIC,
)

SEED = 42


def load_data(path=DATA_PATH):
    frame = pd.read_csv(path)
    X = frame.loc[:, FEATURES]
    y = frame["booking_status"].map(LABELS)
    if y.isna().any() or set(y) != set(LABELS.values()):
        raise ValueError("booking_status must contain the two known labels")
    if X.isna().any().any():
        raise ValueError("The selected booking fields must be complete")
    if frame["Booking_ID"].isna().any() or not frame["Booking_ID"].is_unique:
        raise ValueError("Booking_ID must be present and unique")
    return X, y


def split_data(X, y):
    # A reservation ID is unique. Identical model inputs still repeat across IDs.
    groups = X.groupby(FEATURES, sort=False, dropna=False).ngroup()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(outer.split(X, y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    folds = list(inner.split(X.iloc[train], y.iloc[train], groups.iloc[train]))
    return train, test, folds, groups


def make_pipeline(model):
    logistic = isinstance(model, LogisticRegression)
    numeric = StandardScaler() if logistic else "passthrough"
    transform = ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        # Keep numeric zeros explicit for trees, especially XGBoost.
        sparse_threshold=1.0 if logistic else 0.0,
    )
    return Pipeline([("preprocessing", transform), ("model", model)])


def holdout_metrics(pipeline, X, y):
    positive = LABELS["Canceled"]
    predicted = pipeline.predict(X)
    score = pipeline.predict_proba(X)[:, list(pipeline.classes_).index(positive)]
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision_canceled": precision_score(y, predicted, pos_label=positive, zero_division=0),
        "recall_canceled": recall_score(y, predicted, pos_label=positive, zero_division=0),
        "f1_canceled": f1_score(y, predicted, pos_label=positive, zero_division=0),
        "average_precision_canceled": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "confusion_matrix_actual_by_predicted_0_1": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def save_model(artifact, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after a complete write, so Streamlit never reads a partial file.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = temporary.name
    try:
        joblib.dump(artifact, temporary_path, compress=3)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def build_artifact(pipeline, model_name, training_features):
    encoder = pipeline.named_steps["preprocessing"].named_transformers_["cat"]
    return {
        "schema_version": ARTIFACT_VERSION,
        "pipeline": pipeline,
        "selected_model": model_name,
        "defaults": training_features.iloc[0].to_dict(),
        "bounds": {
            name: (float(training_features[name].min()), float(training_features[name].max()))
            for name in NUMERIC
        },
        "categories": dict(zip(CATEGORICAL, (values.tolist() for values in encoder.categories_))),
    }


def train():
    X, y = load_data()
    train_indices, test_indices, folds, groups = split_data(X, y)
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=2),
        "XGBoost": XGBClassifier(random_state=SEED, n_jobs=2, tree_method="hist", eval_metric="logloss"),
    }
    pipelines = {name: make_pipeline(model) for name, model in candidates.items()}
    results = {}
    for name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring="f1", error_score="raise")
        results[name] = {"cv_f1_canceled_folds": scores.tolist(), "cv_f1_canceled_mean": float(scores.mean())}
        print(f"{name} | eğitim içi iptal F1 {scores.mean():.4f}", flush=True)

    # Freeze model selection before evaluating any holdout predictions.
    selected = max(results, key=lambda name: results[name]["cv_f1_canceled_mean"])
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        holdout = holdout_metrics(pipeline, X_test, y_test)
        results[name]["holdout"] = holdout
        print(f"{name} | test doğruluğu {holdout['accuracy']:.2%} | iptal F1 {holdout['f1_canceled']:.4f}")

    metrics = {
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("pandas", "numpy", "scikit-learn", "scipy", "xgboost", "joblib")},
        "seed": SEED,
        "rows": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_canceled": int(y_train.sum()),
        "test_canceled": int(y_test.sum()),
        "feature_groups": int(groups.nunique()),
        "shared_train_test_groups": len(set(groups.iloc[train_indices]) & set(groups.iloc[test_indices])),
        "holdout_majority_accuracy": float((y_test == LABELS["Not_Canceled"]).mean()),
        "features": FEATURES,
        "target_encoding": LABELS,
        "selection_metric": "mean grouped training CV F1 for Canceled",
        "selected_model": selected,
        "models": results,
    }
    artifact = build_artifact(pipelines[selected], selected, X_train)
    save_model(artifact, ARTIFACT_PATH)
    ARTIFACT_PATH.with_name("evaluation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Seçilen model | {selected}")
    print(f"Ayrı test kümesi | {len(X_test)} kayıt")
    holdout = results[selected]["holdout"]
    print(f"Test doğruluğu | {holdout['accuracy']:.2%}")
    print(f"Test iptal F1 | {holdout['f1_canceled']:.4f}")
    print(f"Model ve değerlendirme | {ARTIFACT_PATH.parent}")


if __name__ == "__main__":
    train()
