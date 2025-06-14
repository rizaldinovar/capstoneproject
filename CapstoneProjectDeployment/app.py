import streamlit as st
import pandas as pd
import joblib

# === Load Model ===
log_reg = joblib.load('logisticregression.pkl')
rf = joblib.load('randomforest.pkl')
gb = joblib.load('gradientboosting.pkl')

# === Load Encoder dan Fitur Kolom ===
label_encoder = joblib.load('labelencoder.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# === Model Dictionary ===
model_dict = {
    "Logistic Regression": log_reg,
    "Random Forest": rf,
    "Gradient Boosting": gb
}

# === Fungsi Prediksi ===
def predict_obesity(input_df, selected_model):
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    model = model_dict[selected_model]
    prediction = model.predict(input_encoded)[0]
    pred_label = label_encoder.inverse_transform([prediction])[0]
    return pred_label

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Prediksi Obesitas", layout="wide")
st.markdown("""
    <style>
        .title {text-align: center; font-size: 36px; font-weight: bold; color: #333;}
        .subtitle {text-align: center; font-size: 18px; color: #555; margin-bottom: 30px;}
        .stButton>button {background-color: #4CAF50; color: white; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Prediksi Tingkat Obesitas</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Masukkan informasi pribadi dan kebiasaan harian Anda untuk mengetahui prediksi tingkat obesitas.</div>', unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.title("Pilih Model")
selected_model = st.sidebar.radio("Model Machine Learning:", list(model_dict.keys()))

# === Formulir Input ===
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Usia (tahun)", min_value=1, max_value=120)
        Gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        Height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, step=0.01)
        Weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=300.0, step=0.1)
        CALC = st.selectbox("Seberapa sering Anda mengonsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"])
        FAVC = st.selectbox("Apakah Anda sering mengonsumsi makanan tinggi kalori?", ["yes", "no"])
        FCVC = st.slider("Seberapa sering Anda makan sayuran setiap kali makan? (0=tidak pernah, 3=selalu)", 0, 3)
        NCP = st.number_input("Berapa kali Anda makan besar per hari?", min_value=1.0, max_value=10.0, step=0.1)

    with col2:
        SCC = st.selectbox("Apakah Anda memantau asupan kalori harian Anda?", ["yes", "no"])
        SMOKE = st.selectbox("Apakah Anda merokok?", ["yes", "no"])
        CH2O = st.number_input("Berapa liter air yang Anda minum per hari?", min_value=0.0, max_value=10.0, step=0.1)
        family_history_with_overweight = st.selectbox("Apakah ada anggota keluarga yang mengalami kelebihan berat badan?", ["yes", "no"])
        FAF = st.number_input("Seberapa sering Anda melakukan aktivitas fisik? (jam per minggu)", min_value=0.0, max_value=40.0, step=0.5)
        TUE = st.number_input("Berapa jam/hari Anda menggunakan perangkat teknologi (HP, TV, dll)?", min_value=0, max_value=24)
        CAEC = st.selectbox("Apakah Anda makan camilan di antara waktu makan?", ["no", "Sometimes", "Frequently", "Always"])
        MTRANS = st.selectbox("Jenis transportasi utama yang biasa Anda gunakan", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

    submitted = st.form_submit_button("🚀 Prediksi")

# === Hasil Prediksi ===
if submitted:
    user_data = pd.DataFrame([{
        "Age": Age,
        "Gender": Gender,
        "Height": Height,
        "Weight": Weight,
        "CALC": CALC,
        "FAVC": FAVC,
        "FCVC": FCVC,
        "NCP": NCP,
        "SCC": SCC,
        "SMOKE": SMOKE,
        "CH2O": CH2O,
        "family_history_with_overweight": family_history_with_overweight,
        "FAF": FAF,
        "TUE": TUE,
        "CAEC": CAEC,
        "MTRANS": MTRANS
    }])

    prediction_result = predict_obesity(user_data, selected_model)
    st.markdown("## 🧾 Hasil Prediksi")
    st.success(f"🎯 Tingkat Obesitas Anda diprediksi sebagai: **{prediction_result}**")
