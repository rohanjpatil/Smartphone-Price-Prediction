import datetime

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Smartphone Price Predictor",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "smartphone_price_model.pkl"

# ------------------------------------------------------------------
# Global styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        :root {
            --accent-1: #7C3AED;
            --accent-2: #2563EB;
            --accent-3: #EC4899;
        }

        /* Hide default streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .stApp {
            background:
                radial-gradient(1200px 600px at 10% -10%, rgba(124,58,237,0.14), transparent 60%),
                radial-gradient(1200px 600px at 110% 10%, rgba(37,99,235,0.14), transparent 55%),
                #F7F7FB;
        }

        /* Hero banner */
        .hero {
            padding: 2.1rem 2rem;
            border-radius: 22px;
            background: linear-gradient(120deg, var(--accent-1) 0%, var(--accent-2) 55%, var(--accent-3) 100%);
            box-shadow: 0 18px 40px -18px rgba(76, 29, 149, 0.55);
            margin-bottom: 1.6rem;
            color: white;
        }
        .hero h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 2.1rem;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.5px;
        }
        .hero p {
            margin: 0;
            font-size: 1rem;
            opacity: 0.92;
            max-width: 46rem;
        }
        .hero-badges { margin-top: 0.9rem; }
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.35);
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 500;
            margin-right: 0.5rem;
            backdrop-filter: blur(6px);
        }

        /* Real bordered containers used as "cards" (st.container(border=True, key=...)).
           Streamlit tags the wrapper with a class like "st-key-card_identity" —
           match any element whose class starts with "st-key-card_" so this
           works regardless of exactly which DOM node carries the class. */
        div[class*="st-key-card_"] {
            background: #FFFFFF !important;
            border: 1px solid rgba(124,58,237,0.12) !important;
            border-radius: 18px !important;
            box-shadow: 0 10px 30px -20px rgba(30, 30, 60, 0.35);
            padding: 0.4rem 0.2rem 0.8rem 0.2rem;
        }
        .section-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.05rem;
            color: #2B2440;
            margin: 0.2rem 0 0.8rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Locked / derived field (read-only badge, e.g. OS tied to brand) */
        .locked-field {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            background: #F3F0FB;
            border: 1px solid #DDD6FE;
            color: #5B21B6;
            border-radius: 10px;
            padding: 0.5rem 0.75rem;
            font-size: 0.95rem;
            font-weight: 500;
        }
        .locked-hint {
            font-size: 0.76rem;
            color: #8B8699;
            margin-top: 0.25rem;
        }

        /* Inputs */
        div[data-baseweb="select"] > div, .stNumberInput input {
            border-radius: 10px !important;
        }

        /* --- Slider enhancements (covers both st.slider and st.select_slider,
           which share the same underlying BaseWeb slider component) --- */

        /* Give the slider room to breathe so the value bubble never clips */
        div[data-testid="stSlider"] {
            padding-top: 0.6rem;
            margin-bottom: 0.2rem;
        }

        /* Track rail (the full-width background bar) */
        div[data-baseweb="slider"] > div:first-child {
            height: 8px !important;
            border-radius: 999px !important;
            background: #EAE6F7 !important;
        }

        /* Filled portion of the track, left of the thumb — gradient instead of flat color */
        div[data-baseweb="slider"] div[data-testid*="stSliderTrack"],
        .stSlider > div > div > div > div {
            height: 8px !important;
            border-radius: 999px !important;
            background: linear-gradient(90deg, var(--accent-2), var(--accent-1), var(--accent-3)) !important;
        }

        /* Thumb handle — bigger, with a soft glow ring */
        div[data-baseweb="slider"] div[role="slider"] {
            width: 22px !important;
            height: 22px !important;
            background: #FFFFFF !important;
            border: 3px solid var(--accent-1) !important;
            box-shadow: 0 2px 8px rgba(124,58,237,0.45), 0 0 0 4px rgba(124,58,237,0.12) !important;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        div[data-baseweb="slider"] div[role="slider"]:hover,
        div[data-baseweb="slider"] div[role="slider"]:focus {
            transform: scale(1.15);
            box-shadow: 0 4px 12px rgba(124,58,237,0.55), 0 0 0 6px rgba(124,58,237,0.16) !important;
        }

        /* Value bubble that floats above the thumb while dragging */
        [data-testid*="ThumbValue"] {
            background: linear-gradient(120deg, var(--accent-1), var(--accent-2)) !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            padding: 0.15rem 0.55rem !important;
            box-shadow: 0 4px 10px rgba(124,58,237,0.35);
        }

        /* Min/max end labels — quiet, no stray highlight box */
        [data-testid*="TickBarMin"],
        [data-testid*="TickBarMax"] {
            background: transparent !important;
            color: #A6A2B8 !important;
            font-size: 0.75rem !important;
        }

        /* Live readout card shown under a slider */
        .slider-readout {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 0.5rem;
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }

        /* Predict button */
        div.stButton > button {
            width: 100%;
            border: none;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.05rem;
            color: white;
            background: linear-gradient(120deg, var(--accent-1), var(--accent-2));
            box-shadow: 0 12px 24px -10px rgba(37, 99, 235, 0.55);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 30px -10px rgba(37, 99, 235, 0.65);
            color: white;
        }
        div.stButton > button:disabled {
            background: #C9C6D8;
            box-shadow: none;
        }

        /* Result card */
        .result-card {
            text-align: center;
            padding: 2rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white;
            box-shadow: 0 20px 45px -18px rgba(5, 150, 105, 0.55);
            margin: 1rem 0 1.4rem 0;
        }
        .result-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            opacity: 0.85;
            margin-bottom: 0.3rem;
        }
        .result-price {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 3rem;
            letter-spacing: -1px;
        }
        .result-sub {
            margin-top: 0.5rem;
            font-size: 0.85rem;
            opacity: 0.9;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1F1147 0%, #2B1C63 100%);
        }
        /* Only force light text on the sidebar's own headings/markdown/cards —
           NOT on st.success/st.error/st.info boxes, which need their own
           readable text color against their own colored backgrounds. */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div.sidebar-card,
        section[data-testid="stSidebar"] div.sidebar-card * {
            color: #EDE9FE !important;
        }
        section[data-testid="stSidebar"] .sidebar-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.9rem;
        }
        section[data-testid="stSidebar"] .stat-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            padding: 0.15rem 0;
        }
        section[data-testid="stSidebar"] .stat-value {
            font-weight: 600;
            color: #C4B5FD !important;
        }
        /* Keep alert boxes (success/error/info) using Streamlit's own
           high-contrast colors instead of the sidebar override above. */
        section[data-testid="stSidebar"] div[data-testid="stAlert"],
        section[data-testid="stSidebar"] div[data-testid="stAlert"] * {
            color: inherit !important;
        }

        /* Explicit, theme-independent colors for form controls so labels
           and values are never left to an ambiguous inherited/dark-mode
           default. */
        .stApp label,
        .stApp .stMarkdown p,
        .stApp span {
            color: #2B2440;
        }
        .stApp input,
        .stApp textarea {
            color: #2B2440 !important;
            background-color: #FFFFFF !important;
        }
        .stApp div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #2B2440 !important;
        }
        .stApp div[data-baseweb="select"] * {
            color: #2B2440 !important;
        }
        .stApp div[data-baseweb="popover"] li {
            color: #2B2440 !important;
            background-color: #FFFFFF !important;
        }
        .stApp [data-testid="stNumberInput"] button {
            color: #2B2440 !important;
        }

        .stCaption, .footnote {
            text-align: center;
            color: #8B8699 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    st.markdown("### 📱 &nbsp;About this project")
    st.markdown(
        """
        <div class="sidebar-card">
        Predicts a smartphone's market price (INR) from its specs using a
        regularized linear regression pipeline trained on ~3,200 listings,
        selected via cross-validation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="stat-row"><span>Problem type</span><span class="stat-value">Regression</span></div>
            <div class="stat-row"><span>Target</span><span class="stat-value">price_inr</span></div>
            <div class="stat-row"><span>Training rows</span><span class="stat-value">~3,200</span></div>
            <div class="stat-row"><span>Model</span><span class="stat-value">Linear (regularized)</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_loaded:
        st.error(f"Model file `{MODEL_PATH}` not found. Place it next to this script.")
    else:
        st.success("Model loaded and ready ✓")

# ------------------------------------------------------------------
# Hero header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📱 Smartphone Price Predictor</h1>
        <p>Dial in the specs below and get an instant, data-driven estimate
        of a smartphone's market price in India.</p>
        <div class="hero-badges">
            <span class="badge">⚡ Instant prediction</span>
            <span class="badge">📊 Trained on real listings</span>
            <span class="badge">🇮🇳 Price in INR</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Valid option sets (business logic lives here, not just in the UI)
# ------------------------------------------------------------------
BRANDS = [
    "Apple", "Asus", "Google", "Honor", "Infinix", "Karbonn", "Lava", "Micromax",
    "Motorola", "Nothing", "OnePlus", "Oppo", "Poco", "Realme", "Samsung", "Sony",
    "Tecno", "Vivo", "Xiaomi", "iQOO", "itel",
]
BUILD_MATERIALS = ["Glass+Metal", "Glass+Titanium", "Metal", "Plastic"]

# Only real-world RAM sizes are selectable — no arbitrary values like 9GB.
RAM_OPTIONS = [1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 32]

# Only real-world storage tiers are selectable.
STORAGE_OPTIONS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def os_for_brand(brand_name: str) -> str:
    """Apple only ships iOS; every other brand in this dataset ships Android."""
    return "iOS" if brand_name == "Apple" else "Android"


def performance_tier(score: int):
    """Map a benchmark score to a human-readable performance tier + color."""
    if score < 150_000:
        return "🟢 Entry-level", "#059669", "#D1FAE5"
    elif score < 500_000:
        return "🔵 Mid-range", "#2563EB", "#DBEAFE"
    elif score < 1_000_000:
        return "🟣 High-end", "#7C3AED", "#EDE9FE"
    else:
        return "🔴 Flagship", "#DB2777", "#FCE7F3"


REFRESH_RATE_LABELS = {
    60: "Standard — everyday use",
    90: "Smooth — noticeably fluid",
    120: "Very smooth — great for gaming",
    144: "Ultra smooth — competitive gaming",
    165: "Esports-grade — maximum fluidity",
}


# --- Identity & Design ---
with st.container(border=True, key="card_identity"):
    st.markdown('<div class="section-title">🏷️ Identity &amp; Design</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        brand = st.selectbox("Brand", BRANDS)
    with c2:
        release_year = st.number_input("Release year", min_value=2015, max_value=2026, value=2024, step=1)
    with c3:
        build_material = st.selectbox("Build material", BUILD_MATERIALS)

    c4, c5 = st.columns(2)
    with c4:
        weight_g = st.number_input("Weight (g)", min_value=100.0, max_value=350.0, value=185.0, step=1.0)
    with c5:
        # OS is derived from brand, not user-selectable, so an invalid
        # combination (e.g. Apple + Android) can never be constructed.
        os_choice = os_for_brand(brand)
        st.markdown("Operating system", unsafe_allow_html=False)
        st.markdown(f'<div class="locked-field">🔒 {os_choice}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="locked-hint">Fixed by brand — Apple ships iOS, every other brand ships Android.</div>',
            unsafe_allow_html=True,
        )

# --- Performance ---
with st.container(border=True, key="card_performance"):
    st.markdown('<div class="section-title">⚙️ Performance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ram_gb = st.selectbox("RAM (GB)", RAM_OPTIONS, index=RAM_OPTIONS.index(8))
    with c2:
        storage_gb = st.selectbox("Storage (GB)", STORAGE_OPTIONS, index=STORAGE_OPTIONS.index(128))

    processor_score = st.slider(
        "Processor benchmark score", min_value=10000, max_value=2000000, value=500000, step=5000
    )
    tier_label, tier_color, tier_bg = performance_tier(processor_score)
    st.markdown(
        f"""
        <div class="slider-readout" style="background:{tier_bg}; color:{tier_color};">
            {tier_label} &nbsp;·&nbsp; {processor_score:,} pts
        </div>
        """,
        unsafe_allow_html=True,
    )

    c3, c4 = st.columns(2)
    with c3:
        has_5g = st.checkbox("📶 Has 5G", value=True)
    with c4:
        has_nfc = st.checkbox("💳 Has NFC", value=True)

# --- Display & Battery ---
with st.container(border=True, key="card_display"):
    st.markdown('<div class="section-title">🔋 Display &amp; Battery</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        screen_size_inch = st.number_input(
            "Screen size (inches)", min_value=4.0, max_value=8.5, value=6.1, step=0.01, format="%.2f"
        )
    with c2:
        refresh_rate_hz = st.select_slider("Refresh rate (Hz)", options=[60, 90, 120, 144, 165], value=120)
        st.markdown(
            f"""
            <div class="slider-readout" style="background:#EDE9FE; color:#7C3AED;">
                🎮 {REFRESH_RATE_LABELS[refresh_rate_hz]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        battery_mah = st.number_input("Battery (mAh)", min_value=1000, max_value=8000, value=4000, step=50)
    with c4:
        fast_charging_watt = st.number_input("Fast charging (W)", min_value=0.0, max_value=250.0, value=25.0, step=1.0)

# --- Camera ---
with st.container(border=True, key="card_camera"):
    st.markdown('<div class="section-title">📷 Camera</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        rear_camera_mp = st.number_input("Rear camera (MP)", min_value=1, max_value=250, value=48, step=1)
    with c2:
        num_rear_cameras = st.number_input("Number of rear cameras", min_value=1, max_value=6, value=2, step=1)
    with c3:
        front_camera_mp = st.number_input("Front camera (MP)", min_value=1.0, max_value=100.0, value=13.0, step=1.0)

st.write("")
predict_clicked = st.button("✨ Predict Price", type="primary", disabled=not model_loaded)

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

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated Market Price</div>
                <div class="result-price">₹{prediction:,.0f}</div>
                <div class="result-sub">{brand} · {ram_gb}GB RAM · {storage_gb}GB storage · {os_choice} · {release_year}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.balloons()

        with st.expander("🔍 Show input passed to the model"):
            st.dataframe(input_df.T.rename(columns={0: "value"}), use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown(
    '<p class="footnote">This is a statistical estimate based on historical listings, '
    "not a guaranteed market price.</p>",
    unsafe_allow_html=True,
)
