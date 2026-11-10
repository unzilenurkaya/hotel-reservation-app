"""Streamlit demo using the fitted preprocessing and classifier together."""

from pickle import UnpicklingError

import joblib
import pandas as pd
import streamlit as st

from reservation import ARTIFACT_PATH, ARTIFACT_VERSION, FEATURES, FIELDS, LABELS, predict_cancellation

st.set_page_config(page_title="Otel Rezervasyon İptal Tahmini", page_icon="🏨", layout="wide")
st.title("Otel Rezervasyon İptal Tahmini")
st.markdown("Rezervasyon bilgilerini girerek modelin iptal tahminini ve iptal skorunu inceleyin.")


@st.cache_resource
def load_artifact(path, modified_ns):
    # The timestamp invalidates the cache after local retraining.
    return joblib.load(path)


def booking_field(name, label, kind, artifact):
    default = artifact["defaults"][name]
    if kind == "category":
        options = artifact["categories"][name]
        return st.selectbox(label, options, index=options.index(default), key=name)
    if kind == "binary":
        return st.selectbox(label, [0, 1], index=int(default), format_func=lambda value: "Evet" if value else "Hayır", key=name)
    lower, upper = artifact["bounds"][name]
    if kind == "float":
        return st.number_input(label, min_value=lower, max_value=upper, value=float(default), step=0.01, key=name)
    return st.number_input(label, min_value=int(lower), max_value=int(upper), value=int(default), step=1, key=name)


try:
    artifact = load_artifact(ARTIFACT_PATH, ARTIFACT_PATH.stat().st_mtime_ns)
except FileNotFoundError:
    st.info("Model dosyası henüz oluşturulmadı. Proje klasöründe `python train.py` çalıştırın.")
    st.stop()
except (OSError, ImportError, ValueError, EOFError, KeyError, UnpicklingError):
    st.error("Model dosyası açılamadı. Proje bağımlılıklarını kurup `python train.py` ile yeniden oluşturun.")
    st.stop()

if not isinstance(artifact, dict) or artifact.get("schema_version") != ARTIFACT_VERSION:
    st.error("Model sürümü bu uygulamayla uyumlu değil. `python train.py` ile yeni model oluşturun.")
    st.stop()

st.caption(f"Kullanılan model · {artifact['selected_model']}")
with st.form("reservation"):
    columns = st.columns(3)
    values = {}
    for index, (name, label, kind) in enumerate(FIELDS):
        with columns[index % len(columns)]:
            values[name] = booking_field(name, label, kind, artifact)
    submitted = st.form_submit_button("Tahmin et", type="primary")

if submitted:
    booking = pd.DataFrame([values], columns=FEATURES)
    predicted, score = predict_cancellation(artifact["pipeline"], booking)
    if predicted == LABELS["Canceled"]:
        st.warning("Model tahmini · İptal")
    else:
        st.success("Model tahmini · İptal edilmez")
    st.metric("İptal model skoru", f"%{score * 100:.1f}".replace(".", ","))
    st.caption("Skor kalibre edilmemiştir. Bu uygulama 2017–2018 verisine dayanan bir portfolyo demosudur.")
