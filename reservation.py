"""Shared raw feature schema and target contract for training and inference."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Hotel Reservations.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "reservation_model.joblib"
ARTIFACT_VERSION = 1
LABELS = {"Not_Canceled": 0, "Canceled": 1}

# Names match the original CSV. The UI uses these exact fields and types.
FIELDS = (
    ("lead_time", "Rezervasyon ile varış arasındaki gün", "int"),
    ("avg_price_per_room", "Gecelik ortalama oda fiyatı", "float"),
    ("market_segment_type", "Pazar segmenti", "category"),
    ("arrival_month", "Varış ayı", "int"),
    ("no_of_adults", "Yetişkin sayısı", "int"),
    ("no_of_children", "Çocuk sayısı", "int"),
    ("no_of_week_nights", "Hafta içi gece sayısı", "int"),
    ("no_of_weekend_nights", "Hafta sonu gece sayısı", "int"),
    ("room_type_reserved", "Rezerve edilen oda tipi", "category"),
    ("type_of_meal_plan", "Yemek planı", "category"),
    ("required_car_parking_space", "Otopark talebi", "binary"),
    ("no_of_special_requests", "Özel istek sayısı", "int"),
)
FEATURES = [name for name, _, _ in FIELDS]
CATEGORICAL = [name for name, _, kind in FIELDS if kind == "category"]
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]


def predict_cancellation(pipeline, booking):
    """Return the predicted class and the model score for Canceled."""
    cancellation = LABELS["Canceled"]
    score_column = list(pipeline.classes_).index(cancellation)
    predicted = int(pipeline.predict(booking)[0])
    score = float(pipeline.predict_proba(booking)[0, score_column])
    return predicted, score
