import streamlit as st
import json
import logging

# --- KONFIGURASI LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- KONFIGURASI HALAMAN (harus paling atas) ---
st.set_page_config(page_title="Deteksi Lagu Daerah", layout="wide")

# Sembunyikan sidebar bawaan Streamlit agar nav kustom tidak terganggu
st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD RESOURCES BERSAMA (di-cache global)
# ==========================================
@st.cache_data(show_spinner=False)
def load_geojson():
    with open("data/indonesia.geojson", "r") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_classifier():
    from modules.classifier import MusicClassifier
    return MusicClassifier(
        model_path="models/xgboost_tuned.pkl",
        encoder_path="models/label_encoder.pkl"
    )

# ==========================================
# KAMUS PEMETAAN (shared constant)
# ==========================================
IRSD_MAPPING = {
    "aceh": "Aceh",
    "jawa barat": "Jawa Barat",
    "jakarta raya": "Jakarta",
    "jawa tengah": "Jawa Tengah",
    "kalimantan barat": "Kalbar",
    "maluku": "Maluku",
    "papua": "Papua",
    "riau": "Riau",
    "sulawesi utara": "Sulawesi Utara",
    "sumatera barat": "Sumatera Barat"
}

# --- INITIALIZE SESSION STATE ---
defaults = {
    "menu": "Beranda", # Default diubah ke Beranda
    "prediction_result": None,
    "last_uploaded_file": None,
    "feature_dataframe": None,
    "selected_song_info": None,
    "current_region": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================================
# HEADER & NAVIGASI
# ==========================================
col_logo, col_nav = st.columns([1, 1.5])

with col_logo:
    st.markdown("""
        <h3 style='margin-bottom: 0;'>🎵 Lagu Daerah</h3>
        <p style='font-size: 14px; margin-top: -10px; color: gray;'>Prediksi Asal Lagu Daerah</p>
    """, unsafe_allow_html=True)

with col_nav:
    st.write("")
    st.write("")
    # Kolom navigasi dibagi 3 untuk Beranda, Deteksi, dan Peta
    nav1, nav2, nav3 = st.columns([1, 1, 1])
    
    with nav1:
        btn_type0 = "primary" if st.session_state.menu == "Beranda" else "secondary"
        if st.button("Beranda", type=btn_type0, use_container_width=True):
            st.session_state.menu = "Beranda"
            st.rerun()
            
    with nav2:
        btn_type1 = "primary" if st.session_state.menu == "Deteksi" else "secondary"
        if st.button("Deteksi", type=btn_type1, use_container_width=True):
            st.session_state.menu = "Deteksi"
            st.rerun()
            
    with nav3:
        btn_type2 = "primary" if st.session_state.menu == "Peta Interaktif" else "secondary"
        if st.button("Peta Interaktif", type=btn_type2, use_container_width=True):
            st.session_state.menu = "Peta Interaktif"
            st.rerun()

st.markdown("<hr style='border: 2px solid #fca311; margin-top: 0;'>", unsafe_allow_html=True)

# ==========================================
# ROUTING HALAMAN
# ==========================================
if st.session_state.menu == "Beranda":
    from pages.halaman_beranda import render
    render()

elif st.session_state.menu == "Deteksi":
    from pages.halaman_deteksi import render
    render(load_geojson(), load_classifier(), IRSD_MAPPING)

elif st.session_state.menu == "Peta Interaktif":
    from pages.halaman_peta import render
    render(load_geojson(), IRSD_MAPPING)