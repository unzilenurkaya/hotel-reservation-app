import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from audit_legacy import audit
from reservation import ARTIFACT_PATH, CATEGORICAL, DATA_PATH, FEATURES, FIELDS, LABELS, ROOT, predict_cancellation
from train import load_data, split_data

APP_PATH = ROOT / "app.py"


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


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ARTIFACT_PATH.is_file():
            raise RuntimeError("Run python train.py before the tests")
        cls.artifact = joblib.load(ARTIFACT_PATH)

    def test_model_selection_and_artifact_match(self):
        metrics = json.loads(ARTIFACT_PATH.with_name("evaluation.json").read_text())
        model = max(metrics["models"], key=lambda name: metrics["models"][name]["cv_f1_canceled_mean"])
        self.assertEqual(metrics["selected_model"], model)
        self.assertEqual(self.artifact["selected_model"], model)
        self.assertEqual(metrics["data_sha256"], hashlib.sha256(DATA_PATH.read_bytes()).hexdigest())
        pipeline = self.artifact["pipeline"]
        self.assertEqual(list(pipeline.feature_names_in_), FEATURES)
        self.assertEqual(list(pipeline.classes_), [0, 1])
        encoder = pipeline.named_steps["preprocessing"].named_transformers_["cat"]
        self.assertEqual(self.artifact["categories"], dict(zip(CATEGORICAL, (c.tolist() for c in encoder.categories_))))

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

    def test_both_ui_labels_and_complete_raw_inputs_with_real_model(self):
        X, _ = load_data()
        sample = X.sample(300, random_state=88)
        pipeline = self.artifact["pipeline"]
        predictions = pipeline.predict(sample)
        self.assertEqual(set(predictions), {0, 1})
        with patch("reservation.predict_cancellation", wraps=predict_cancellation) as inference:
            app = AppTest.from_file(str(APP_PATH)).run()
            self.assertFalse(app.exception)
            for expected in (0, 1):
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
                self.assertFalse(app.exception)
                messages = app.warning if expected else app.success
                self.assertEqual(messages[0].value, "Model tahmini · İptal" if expected else "Model tahmini · İptal edilmez")
                sent = inference.call_args.args[1]
                self.assertEqual(list(sent.columns), FEATURES)
                pd.testing.assert_series_equal(sent.iloc[0], booking, check_names=False)
                score = pipeline.predict_proba(sent)[0, list(pipeline.classes_).index(1)]
                self.assertEqual(app.metric[0].value, f"%{score * 100:.1f}".replace(".", ","))


if __name__ == "__main__":
    unittest.main()
