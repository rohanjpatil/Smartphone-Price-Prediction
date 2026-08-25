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
            margin-bottom: 1.4rem;
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

        /* Quick presets bar */
        .presets-label {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            color: #2B2440;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
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
            padding: 0.6rem 0.3rem 0.9rem 0.3rem;
            transition: box-shadow 0.15s ease, transform 0.15s ease;
        }
        div[class*="st-key-card_"]:hover {
            box-shadow: 0 16px 38px -18px rgba(30, 30, 60, 0.5);
        }
        .section-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1.05rem;
            color: #2B2440;
            margin: 0.2rem 0 0.9rem 0;
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

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: #EFEBFA;
            padding: 6px;
            border-radius: 14px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            color: #6B6480 !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(120deg, var(--accent-1), var(--accent-2)) !important;
            color: white !important;
        }
        .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
        .stTabs [data-baseweb="tab-border"] { display: none !important; }

        /* --- Slider polish ---
           Streamlit's slider internals use auto-generated, build-specific
           class names (e.g. st-emotion-cache-xxxxx) that are unreliable to
           target directly. Rather than fight that, we lean on Streamlit's
           own theme (primaryColor, set in .streamlit/config.toml) for the
           track/thumb color, and only add safe, layout-level polish here. */
        div[data-testid="stSlider"] {
            padding-top: 0.6rem;
            padding-bottom: 0.2rem;
        }

        /* Live readout card shown under a slider — fully custom-rendered,
           so its appearance never depends on Streamlit's internal markup. */
        .slider-readout {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 0.6rem;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        /* --- Buttons ---
           Streamlit tags any widget with a `key` using a "st-key-<key>" class,
           so each button family (primary CTA / presets / reset) gets its own
           scoped style instead of one generic rule fighting all of them. */
        div.stButton > button {
            border-radius: 12px;
            font-weight: 600;
            transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
        }
        div[class*="st-key-predict_btn"] button {
            width: 100%;
            border: none !important;
            border-radius: 14px !important;
            padding: 0.85rem 1rem !important;
            font-family: 'Poppins', sans-serif;
            font-size: 1.05rem !important;
            color: white !important;
            background: linear-gradient(120deg, var(--accent-1), var(--accent-2)) !important;
            box-shadow: 0 12px 24px -10px rgba(37, 99, 235, 0.55) !important;
        }
        div[class*="st-key-predict_btn"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 30px -10px rgba(37, 99, 235, 0.65) !important;
            color: white !important;
        }
        div[class*="st-key-predict_btn"] button:disabled {
            background: #C9C6D8 !important;
            box-shadow: none !important;
        }
        div[class*="st-key-preset_"] button {
            border: 2px solid var(--accent-1) !important;
            background: #FFFFFF !important;
            color: var(--accent-1) !important;
            border-radius: 999px !important;
            box-shadow: none !important;
        }
        div[class*="st-key-preset_"] button:hover {
            background: #F5F3FF !important;
            transform: translateY(-1px);
        }
        div[class*="st-key-reset_btn"] button {
            border: 1px solid #E5E1F0 !important;
            background: transparent !important;
            color: #8B8699 !important;
            border-radius: 999px !important;
            box-shadow: none !important;
        }
        div[class*="st-key-reset_btn"] button:hover {
            background: #F3F0FB !important;
            color: #5B21B6 !important;
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
        .result-tier {
            display: inline-block;
            margin-top: 0.7rem;
            padding: 0.25rem 0.85rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.35);
            font-size: 0.82rem;
            font-weight: 600;
        }
        .price-gauge {
            margin-top: 1.1rem;
            text-align: left;
        }
        .price-gauge-track {
            position: relative;
            height: 10px;
            border-radius: 999px;
            background: linear-gradient(90deg, #A7F3D0, #FDE68A, #FCA5A5);
        }
        .price-gauge-marker {
            position: absolute;
            top: -5px;
            width: 5px;
            height: 20px;
            background: white;
            border-radius: 3px;
            box-shadow: 0 0 0 2px rgba(0,0,0,0.18);
        }
        .price-gauge-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.72rem;
            opacity: 0.85;
            margin-top: 6px;
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

    st.markdown(
        """
        <div class="sidebar-card">
        💡 <b>Tip:</b> use a Quick Preset up top to jump-start the form,
        then fine-tune specs in the tabs below it.
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

REFRESH_RATE_LABELS = {
    60: "Standard — everyday use",
    90: "Smooth — noticeably fluid",
    120: "Very smooth — great for gaming",
    144: "Ultra smooth — competitive gaming",
    165: "Esports-grade — maximum fluidity",
}


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


def price_tier(price: float):
    """Map a predicted price to a market-segment label."""
    if price < 15_000:
        return "💚 Budget segment"
    elif price < 40_000:
        return "💙 Mid-range segment"
    elif price < 80_000:
        return "💜 Premium segment"
    else:
        return "❤️ Flagship segment"


# ------------------------------------------------------------------
# Default spec values + quick presets
# ------------------------------------------------------------------
DEFAULTS = {
    "brand": "Samsung",
    "release_year": 2024,
    "build_material": "Glass+Metal",
    "weight_g": 185.0,
    "ram_gb": 8,
    "storage_gb": 128,
    "processor_score": 500000,
    "has_5g": True,
    "has_nfc": True,
    "screen_size_inch": 6.1,
    "refresh_rate_hz": 120,
    "battery_mah": 4000,
    "fast_charging_watt": 25.0,
    "rear_camera_mp": 48,
    "num_rear_cameras": 2,
    "front_camera_mp": 13.0,
}

PRESETS = {
    "Budget": {
        "brand": "Xiaomi", "release_year": 2024, "build_material": "Plastic", "weight_g": 190.0,
        "ram_gb": 4, "storage_gb": 64, "processor_score": 120000, "has_5g": False, "has_nfc": False,
        "screen_size_inch": 6.5, "refresh_rate_hz": 90, "battery_mah": 5000, "fast_charging_watt": 18.0,
        "rear_camera_mp": 50, "num_rear_cameras": 2, "front_camera_mp": 8.0,
    },
    "Mid-range": {
        "brand": "Samsung", "release_year": 2024, "build_material": "Glass+Metal", "weight_g": 185.0,
        "ram_gb": 8, "storage_gb": 128, "processor_score": 500000, "has_5g": True, "has_nfc": True,
        "screen_size_inch": 6.4, "refresh_rate_hz": 120, "battery_mah": 4500, "fast_charging_watt": 33.0,
        "rear_camera_mp": 50, "num_rear_cameras": 3, "front_camera_mp": 16.0,
    },
    "Flagship": {
        "brand": "Apple", "release_year": 2025, "build_material": "Glass+Titanium", "weight_g": 221.0,
        "ram_gb": 12, "storage_gb": 256, "processor_score": 1800000, "has_5g": True, "has_nfc": True,
        "screen_size_inch": 6.7, "refresh_rate_hz": 120, "battery_mah": 4400, "fast_charging_watt": 27.0,
        "rear_camera_mp": 48, "num_rear_cameras": 3, "front_camera_mp": 12.0,
    },
}

# Initialize widget state once; safe to run on every rerun since it only
# fills in keys that aren't already set.
for _key, _val in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


def apply_preset(name):
    """Load a named preset (or None to reset to defaults) into widget state."""
    values = DEFAULTS if name is None else PRESETS[name]
    for k, v in values.items():
        st.session_state[k] = v
    st.rerun()


# ------------------------------------------------------------------
# Quick presets
# ------------------------------------------------------------------
st.markdown('<div class="presets-label">⚡ Quick presets</div>', unsafe_allow_html=True)
p1, p2, p3, p4 = st.columns([1, 1, 1, 0.85])
with p1:
    if st.button("💰 Budget", key="preset_budget", use_container_width=True):
        apply_preset("Budget")
with p2:
    if st.button("⚖️ Mid-range", key="preset_mid", use_container_width=True):
        apply_preset("Mid-range")
with p3:
    if st.button("🚀 Flagship", key="preset_flagship", use_container_width=True):
        apply_preset("Flagship")
with p4:
    if st.button("↺ Reset", key="reset_btn", use_container_width=True):
        apply_preset(None)

st.write("")

# ------------------------------------------------------------------
# Spec input tabs
# ------------------------------------------------------------------
tab_identity, tab_perf, tab_display, tab_camera = st.tabs(
    ["🏷️ Design", "⚙️ Performance", "🔋 Display & Battery", "📷 Camera"]
)

with tab_identity:
    with st.container(border=True, key="card_identity"):
        c1, c2, c3 = st.columns(3)
        with c1:
            brand = st.selectbox("Brand", BRANDS, key="brand")
        with c2:
            release_year = st.number_input(
                "Release year", min_value=2015, max_value=2026, step=1, key="release_year"
            )
        with c3:
            build_material = st.selectbox("Build material", BUILD_MATERIALS, key="build_material")

        c4, c5 = st.columns(2)
        with c4:
            weight_g = st.number_input(
                "Weight (g)", min_value=100.0, max_value=350.0, step=1.0, key="weight_g"
            )
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

with tab_perf:
    with st.container(border=True, key="card_performance"):
        c1, c2 = st.columns(2)
        with c1:
            ram_gb = st.selectbox("RAM (GB)", RAM_OPTIONS, key="ram_gb")
        with c2:
            storage_gb = st.selectbox("Storage (GB)", STORAGE_OPTIONS, key="storage_gb")

        processor_score = st.slider(
            "Processor benchmark score",
            min_value=10000,
            max_value=2000000,
            step=5000,
            format="%,d",
            key="processor_score",
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
            has_5g = st.checkbox("📶 Has 5G", key="has_5g")
        with c4:
            has_nfc = st.checkbox("💳 Has NFC", key="has_nfc")

with tab_display:
    with st.container(border=True, key="card_display"):
        c1, c2 = st.columns(2)
        with c1:
            screen_size_inch = st.number_input(
                "Screen size (inches)", min_value=4.0, max_value=8.5, step=0.01, format="%.2f",
                key="screen_size_inch",
            )
        with c2:
            refresh_rate_hz = st.segmented_control(
                "Refresh rate (Hz)",
                options=[60, 90, 120, 144, 165],
                selection_mode="single",
                key="refresh_rate_hz",
            )
            if refresh_rate_hz is None:
                refresh_rate_hz = DEFAULTS["refresh_rate_hz"]  # guard against deselecting the pill
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
            battery_mah = st.number_input(
                "Battery (mAh)", min_value=1000, max_value=8000, step=50, key="battery_mah"
            )
        with c4:
            fast_charging_watt = st.number_input(
                "Fast charging (W)", min_value=0.0, max_value=250.0, step=1.0, key="fast_charging_watt"
            )

with tab_camera:
    with st.container(border=True, key="card_camera"):
        c1, c2, c3 = st.columns(3)
        with c1:
            rear_camera_mp = st.number_input(
                "Rear camera (MP)", min_value=1, max_value=250, step=1, key="rear_camera_mp"
            )
        with c2:
            num_rear_cameras = st.number_input(
                "Number of rear cameras", min_value=1, max_value=6, step=1, key="num_rear_cameras"
            )
        with c3:
            front_camera_mp = st.number_input(
                "Front camera (MP)", min_value=1.0, max_value=100.0, step=1.0, key="front_camera_mp"
            )

st.write("")
predict_clicked = st.button(
    "✨ Predict Price", type="primary", disabled=not model_loaded, key="predict_btn"
)

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

        # Position marker for the price gauge, clamped to a 0–150K INR span
        gauge_max = 150_000
        gauge_pct = max(0.0, min(prediction / gauge_max, 1.0)) * 100

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated Market Price</div>
                <div class="result-price">₹{prediction:,.0f}</div>
                <div class="result-sub">{brand} · {ram_gb}GB RAM · {storage_gb}GB storage · {os_choice} · {release_year}</div>
                <div class="result-tier">{price_tier(prediction)}</div>
                <div class="price-gauge">
                    <div class="price-gauge-track">
                        <div class="price-gauge-marker" style="left:{gauge_pct}%;"></div>
                    </div>
                    <div class="price-gauge-labels">
                        <span>₹0</span><span>₹75K</span><span>₹150K+</span>
                    </div>
                </div>
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
