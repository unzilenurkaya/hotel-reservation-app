"""Reconstruct the historical notebook's data selection without loading model.pkl."""

import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from reservation import DATA_PATH


def audit():
    frame = pd.read_csv(DATA_PATH)
    target = "booking_status"
    cleaned = frame.copy()
    steps = []
    for column in frame.select_dtypes(exclude="object"):
        q1, q3 = cleaned[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        keep = cleaned[column].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        steps.append({
            "column": column,
            "iqr": float(iqr),
            "removed_by_class": cleaned.loc[~keep, target].value_counts().to_dict(),
            "remaining_rows": int(keep.sum()),
        })
        cleaned = cleaned.loc[keep].copy()

    # Match the original pre-split frequency grouping, encoding and column removal.
    for column in frame.select_dtypes(include="object").columns.drop("Booking_ID"):
        freq = cleaned[column].value_counts(normalize=True)
        cleaned.loc[cleaned[column].isin(freq[freq < 0.01].index), column] = "Other"

    target_counts = cleaned[target].value_counts().to_dict()
    category_columns = cleaned.select_dtypes(include="object").columns
    binary = [column for column in category_columns if cleaned[column].nunique() == 2]
    encoder = LabelEncoder()
    mapping = {}
    for column in binary:
        cleaned[column] = encoder.fit_transform(cleaned[column])
        if column == target:
            mapping = {label: int(index) for index, label in enumerate(encoder.classes_)}

    # Booking_ID dummies are dropped by the notebook. Skip creating those columns here.
    cleaned = cleaned.drop(columns="Booking_ID")
    encoded = pd.get_dummies(cleaned, drop_first=True)
    encoded = encoded.loc[:, encoded.nunique().ne(1) & ~encoded.isna().any()]
    X = encoded.drop(columns=target)
    y = encoded[target]
    train, test, _, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_patterns = set(train.itertuples(index=False, name=None))
    test_patterns = list(test.itertuples(index=False, name=None))
    calendar = frame[["arrival_year", "arrival_month", "arrival_date"]].copy()
    calendar.columns = ["year", "month", "day"]

    return {
        "raw_rows": len(frame),
        "raw_label_counts": frame[target].value_counts().to_dict(),
        "raw_missing_values": int(frame.isna().sum().sum()),
        "raw_unique_booking_ids": frame["Booking_ID"].nunique(),
        "invalid_calendar_dates": int(pd.to_datetime(calendar, errors="coerce").isna().sum()),
        "raw_identical_input_rows_beyond_first": int(frame.drop(columns=["Booking_ID", target]).duplicated().sum()),
        "iqr_steps": steps,
        "historical_label_counts": target_counts,
        "historical_encoding": mapping,
        "historical_features": list(X.columns),
        "historical_train_rows": len(train),
        "historical_test_rows": len(test),
        "historical_test_label_counts": {str(label): int(count) for label, count in test_y.value_counts().items()},
        "historical_distinct_test_patterns_also_in_train": len(set(test_patterns) & train_patterns),
        "historical_test_rows_with_input_pattern_in_train": sum(row in train_patterns for row in test_patterns),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
