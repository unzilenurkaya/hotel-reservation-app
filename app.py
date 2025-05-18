import streamlit as st
import pandas as pd
import joblib

# Modeli yükle
model = joblib.load("model.pkl")

st.title("🛎️ Otel Rezervasyon İptal Tahmini")

# Kullanıcıdan giriş al
st.sidebar.header("Müşteri Bilgileri")

lead_time = st.sidebar.number_input("Lead Time (gün)", min_value=0)
adr = st.sidebar.number_input("Ortalama Günlük Ücret (ADR)")
total_of_special_requests = st.sidebar.slider("Özel İstek Sayısı", 0, 5)
required_car_parking_spaces = st.sidebar.selectbox("Otopark Gerekiyor mu?", ["Evet", "Hayır"])
deposit_type = st.sidebar.selectbox("Depozito Tipi", ["No Deposit", "Non Refund", "Refundable"])
market_segment = st.sidebar.selectbox("Pazar Segmenti", ["Online TA", "Offline TA/TO", "Groups", "Direct"])

# Girdi verisini işleme
data = pd.DataFrame({
    "lead_time": [lead_time],
    "adr": [adr],
    "total_of_special_requests": [total_of_special_requests],
    "required_car_parking_spaces": [1 if required_car_parking_spaces == "Evet" else 0],
    "deposit_type": [deposit_type],
    "market_segment": [market_segment]
})

# One-hot encoding (modelin eğitimde kullandığı gibi olmalı!)
data = pd.get_dummies(data)

# Eksik sütunları doldurma (modelin beklediği sütun sayısı olmalı!)
expected_cols = model.feature_names_in_
for col in expected_cols:
    if col not in data.columns:
        data[col] = 0

data = data[expected_cols]

# Tahmin
if st.button("Tahmin Et"):
    prediction = model.predict(data)[0]
    if prediction == 1:
        st.error("❌ Bu rezervasyonun iptal edilme ihtimali yüksek.")
    else:
        st.success("✅ Bu rezervasyonun iptal edilme ihtimali düşük.")
