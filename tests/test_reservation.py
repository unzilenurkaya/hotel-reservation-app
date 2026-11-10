import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from streamlit.testing.v1 import AppTest

from audit_legacy import audit
from reservation import DATA_PATH, FEATURES, FIELDS, LABELS, ROOT, predict_cancellation
from train import build_artifact, load_data, make_pipeline, save_model, split_data

APP_PATH = ROOT / "app.py"
PUBLISHED_EVALUATION = ROOT / "docs" / "evaluation.json"


def assert_ui_predictions(case, artifact_path, pipeline, sample):
    predictions = pipeline.predict(sample)
    case.assertEqual(set(predictions), {0, 1})
    with patch("reservation.ARTIFACT_PATH", artifact_path), patch(
        "reservation.predict_cancellation", wraps=predict_cancellation
    ) as inference:
        app = AppTest.from_file(str(APP_PATH)).run()
        case.assertFalse(app.exception)
        for expected in (0, 1):
            with case.subTest(predicted_class=expected):
                booking = sample.iloc[np.flatnonzero(predictions == expected)[0]]
                for name, _, kind in FIELDS:
                    value = booking[name]
                    if kind == "category":
                        app.selectbox(key=name).set_value(str(value))
                    elif kind == "binary":
                        app.selectbox(key=name).set_value(int(value))
                    else:
                        app.number_input(key=name).set_value(float(value) if kind == "float" else int(value))
                app.button[0].click().run()
                case.assertFalse(app.exception)
                messages = app.warning if expected else app.success
                case.assertFalse(app.success if expected else app.warning)
                case.assertEqual(len(messages), 1)
                case.assertEqual(messages[0].value, "Model tahmini · İptal" if expected else "Model tahmini · İptal edilmez")
                sent = inference.call_args.args[1]
                case.assertEqual(list(sent.columns), FEATURES)
                pd.testing.assert_series_equal(sent.iloc[0], booking, check_names=False)
                score = pipeline.predict_proba(sent)[0, list(pipeline.classes_).index(1)]
                case.assertEqual(app.metric[0].value, f"%{score * 100:.1f}".replace(".", ","))


class DataTests(unittest.TestCase):
    def test_legacy_population_and_overlap(self):
        result = audit()
        self.assertEqual(result["raw_rows"], 36275)
        self.assertEqual(result["historical_label_counts"], {"Not_Canceled": 10507, "Canceled": 6651})
        self.assertEqual(result["historical_encoding"], {"Canceled": 0, "Not_Canceled": 1})
        self.assertEqual(result["historical_test_rows_with_input_pattern_in_train"], 1210)

    def test_no_equal_inputs_cross_holdout_or_cv_folds(self):
        X, y = load_data()
        train, test, folds, groups = split_data(X, y)
        self.assertEqual(len(X), len(train) + len(test))
        self.assertEqual(len(X), len(set(train) | set(test)))
        self.assertEqual(list(X.columns), FEATURES)
        self.assertEqual(LABELS, {"Not_Canceled": 0, "Canceled": 1})
        self.assertEqual(int(y.sum()), 11885)
        self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[test]))
        train_groups = groups.iloc[train]
        for left, right in folds:
            self.assertFalse(set(train_groups.iloc[left]) & set(train_groups.iloc[right]))

    def test_score_uses_canceled_class_even_if_class_order_changes(self):
        model = Mock(
            classes_=np.array([1, 0]),
            **{"predict.return_value": np.array([1]), "predict_proba.return_value": np.array([[0.81, 0.19]])},
        )
        self.assertEqual(predict_cancellation(model, pd.DataFrame()), (1, 0.81))

    def test_readme_model_table_matches_published_evaluation(self):
        recorded = json.loads(PUBLISHED_EVALUATION.read_text())
        readme = (ROOT / "README.md").read_text().replace("**", "")
        self.assertEqual(recorded["data_sha256"], hashlib.sha256(DATA_PATH.read_bytes()).hexdigest())
        self.assertEqual(recorded["target_encoding"], LABELS)
        self.assertEqual(recorded["features"], FEATURES)
        self.assertEqual(
            recorded["selected_model"],
            max(recorded["models"], key=lambda name: recorded["models"][name]["cv_f1_canceled_mean"]),
        )
        for name, result in recorded["models"].items():
            with self.subTest(model=name):
                holdout = result["holdout"]
                (tn, fp), (fn, tp) = holdout["confusion_matrix_actual_by_predicted_0_1"]
                total = tn + fp + fn + tp
                self.assertEqual(total, recorded["test_rows"])
                self.assertEqual(fn + tp, recorded["test_canceled"])
                self.assertAlmostEqual((tn + tp) / total, holdout["accuracy"])
                self.assertAlmostEqual(2 * tp / (2 * tp + fp + fn), holdout["f1_canceled"])
                row = (
                    f"| {name} | {result['cv_f1_canceled_mean']:.4f} | %{holdout['accuracy'] * 100:.2f} | "
                    f"{holdout['f1_canceled']:.4f} | {holdout['roc_auc']:.4f} |"
                )
                self.assertIn(row.replace(".", ","), readme)
        self.assertIn("gerçek rezervasyon kararları için doğrulanmış bir sistem değildir", readme)
        self.assertTrue((ROOT / "docs" / "streamlit-preview.jpg").is_file())


class AppTests(unittest.TestCase):
    def test_missing_or_corrupt_artifact_has_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "missing.joblib"
            with patch("reservation.ARTIFACT_PATH", model_path):
                app = AppTest.from_file(str(APP_PATH)).run()
                self.assertFalse(app.exception)
                self.assertIn("python train.py", app.info[0].value)
                model_path.write_bytes(b"not a model")
                app = AppTest.from_file(str(APP_PATH)).run()
                self.assertFalse(app.exception)
                self.assertIn("python train.py", app.error[0].value)

    def test_ui_labels_and_inputs_with_small_pipeline(self):
        X, y = load_data()
        sample = X.sample(128, random_state=88)
        pipeline = make_pipeline(DecisionTreeClassifier(random_state=42)).fit(sample, y.loc[sample.index])
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "test.joblib"
            save_model(build_artifact(pipeline, "Decision Tree", sample), model_path)
            assert_ui_predictions(self, model_path, pipeline, sample)
