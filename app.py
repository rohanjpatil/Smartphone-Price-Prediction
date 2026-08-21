import datetime

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Smartphone Price Predictor", page_icon="📱", layout="centered")

MODEL_PATH = "smartphone_price_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    pipeline = load_model()
    model_loaded = True
except FileNotFoundError:
    pipeline = None
    model_loaded = False

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.write(
        "This app predicts a smartphone's price (in INR) from its specifications "
        "using a regularized linear regression model trained on ~3,200 smartphone "
        "listings. The model was selected via cross-validation and is deployed as "
        "a single scikit-learn pipeline (preprocessing + model)."
    )
    st.markdown("**Problem type:** Regression")
    st.markdown("**Target:** `price_inr`")
    if not model_loaded:
        st.error(f"Model file `{MODEL_PATH}` not found. Place it next to this script.")

st.title("📱 Smartphone Price Predictor")
st.write("Enter the specs of a smartphone below to estimate its market price in INR.")

# ------------------------------------------------------------------
# Input widgets
# ------------------------------------------------------------------
BRANDS = [
    "Apple", "Asus", "Google", "Honor", "Infinix", "Karbonn", "Lava", "Micromax",
    "Motorola", "Nothing", "OnePlus", "Oppo", "Poco", "Realme", "Samsung", "Sony",
    "Tecno", "Vivo", "Xiaomi", "iQOO", "itel",
]
BUILD_MATERIALS = ["Glass+Metal", "Glass+Titanium", "Metal", "Plastic"]
OS_OPTIONS = ["Android", "iOS"]

st.subheader("Specifications")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", BRANDS)
    release_year = st.number_input("Release year", min_value=2015, max_value=2026, value=2024, step=1)
    ram_gb = st.number_input("RAM (GB)", min_value=1, max_value=32, value=8, step=1)
    storage_gb = st.number_input("Storage (GB)", min_value=8, max_value=2048, value=128, step=8)
    battery_mah = st.number_input("Battery (mAh)", min_value=1000, max_value=8000, value=4000, step=50)
    screen_size_inch = st.number_input("Screen size (inches)", min_value=4.0, max_value=8.5, value=6.1, step=0.01, format="%.2f")
    refresh_rate_hz = st.selectbox("Refresh rate (Hz)", [60, 90, 120, 144, 165], index=2)
    processor_score = st.number_input("Processor benchmark score", min_value=10000, max_value=2000000, value=500000, step=1000)
    build_material = st.selectbox("Build material", BUILD_MATERIALS)

with col2:
    rear_camera_mp = st.number_input("Rear camera (MP)", min_value=1, max_value=250, value=48, step=1)
    num_rear_cameras = st.number_input("Number of rear cameras", min_value=1, max_value=6, value=2, step=1)
    front_camera_mp = st.number_input("Front camera (MP)", min_value=1.0, max_value=100.0, value=13.0, step=1.0)
    fast_charging_watt = st.number_input("Fast charging (W)", min_value=0.0, max_value=250.0, value=25.0, step=1.0)
    weight_g = st.number_input("Weight (g)", min_value=100.0, max_value=350.0, value=185.0, step=1.0)
    os_choice = st.selectbox("Operating system", OS_OPTIONS)
    has_5g = st.checkbox("Has 5G", value=True)
    has_nfc = st.checkbox("Has NFC", value=True)

predict_clicked = st.button("Predict Price", type="primary", disabled=not model_loaded)

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if predict_clicked:
    try:
        # Recreate the exact engineered features used during training
        current_year = datetime.datetime.now().year
        phone_age = current_year - release_year
        total_camera_mp = rear_camera_mp + front_camera_mp
        high_refresh = int(refresh_rate_hz >= 90)

        input_df = pd.DataFrame([{
            "brand": brand,
            "release_year": release_year,
            "ram_gb": ram_gb,
            "storage_gb": storage_gb,
            "battery_mah": battery_mah,
            "screen_size_inch": screen_size_inch,
            "refresh_rate_hz": refresh_rate_hz,
            "processor_score": processor_score,
            "rear_camera_mp": rear_camera_mp,
            "num_rear_cameras": num_rear_cameras,
            "front_camera_mp": front_camera_mp,
            "fast_charging_watt": fast_charging_watt,
            "has_5g": int(has_5g),
            "has_nfc": int(has_nfc),
            "build_material": build_material,
            "weight_g": weight_g,
            "os": os_choice,
            "phone_age": phone_age,
            "total_camera_mp": total_camera_mp,
            "high_refresh": high_refresh,
        }])

        prediction = pipeline.predict(input_df)[0]

        st.success(f"### Estimated price: ₹{prediction:,.0f}")
        with st.expander("Show input passed to the model"):
            st.dataframe(input_df.T.rename(columns={0: "value"}))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption(
    "This is a statistical estimate based on historical listings, not a guaranteed "
    "market price."
)
