import math
import os
import json
import logging
import streamlit as st
import folium
from streamlit_folium import st_folium
from modules.processor import process_audio
from modules.database import get_songs_by_region
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="Dashboard Prediksi & Eksplorasi Lagu Daerah", layout="wide")

# ==========================================
# RESOURCE MANAGEMENT & GLOBAL CACHING
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

@st.cache_resource(show_spinner=False)
def _get_genai_client():
    from google import genai
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_data(show_spinner=False)
def _fetch_gemini_data(judul_lagu: str, asal_daerah: str) -> str:
    client = _get_genai_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            f"Berikan informasi singkat, menarik, dan edukatif tentang lagu daerah "
            f"'{judul_lagu}' dari {asal_daerah}. Jelaskan makna lagu "
            f"dan nilai budayanya. Gunakan bahasa yang santai namun informatif."
        ),
    )
    return response.text

def get_song_info_from_gemini(judul_lagu: str, asal_daerah: str) -> str:
    try:
        return _fetch_gemini_data(judul_lagu, asal_daerah)
    except Exception as e:
        # Deteksi apakah error mengandung kode status 503 (Service Unavailable)
        is_503 = False
        if hasattr(e, 'code') and e.code == 503:
            is_503 = True
        elif hasattr(e, 'status_code') and e.status_code == 503:
            is_503 = True
        elif "503" in str(e):
            is_503 = True
            
        # Pembedaan pesan error berdasarkan hasil deteksi
        if is_503:
            return (
                "⚠️ **Error 503: Server Gemini Sibuk (Service Unavailable).**\n\n"
                "Layanan Google Gemini AI saat ini sedang menerima beban lalu lintas data yang terlalu padat "
                "atau sedang dalam pemeliharaan sistem sementara.\n\n"
                "💡 *Solusi:* Silakan tunggu sekitar 10–30 detik, lalu klik ulang tombol **'Pilih Lagu'** untuk mencoba kembali."
            )
        else:
            return (
                f"❌ **Gagal Memuat Informasi Budaya.**\n\n"
                f"Terjadi kesalahan sistem saat menghubungi sistem Google GenAI.\n\n"
                f"*Detail Teknis: {e}*"
            )

@st.cache_data(show_spinner=False)
def _cached_songs(region: str):
    return get_songs_by_region(region)

# ==========================================
# KAMUS PEMETAAN & KONSTANTA UTAMA
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

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
defaults = {
    "prediction_result": None,
    "prediction_confidence": 0.0,
    "last_uploaded_file": None,
    "feature_dataframe": None,
    "selected_song_info": None,
    "current_region": None,
    "is_processing": False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================================
# HELPER UI & LOGIKA PETA
# ==========================================
def _build_dashboard_map(geojson_data: dict, current_region: str) -> folium.Map:
    m = folium.Map(
        location=[-2.5, 118], 
        zoom_start=5, 
        tiles="CartoDB positron",
        zoom_control=False,        
        dragging=False,            
        scrollWheelZoom=False,     
        doubleClickZoom=False,     
        touchZoom=False            
    )

    def style_fn(feature):
        state = str(feature["properties"].get("state", "")).strip().lower()
        mapped_name = IRSD_MAPPING.get(state, None)
        
        if mapped_name and current_region and mapped_name.lower() == current_region.lower():
            return {"fillColor": "#ff4b4b", "color": "black", "weight": 2.5, "fillOpacity": 0.9}
        elif state in IRSD_MAPPING:
            return {"fillColor": "#fca311", "color": "white", "weight": 1.5, "fillOpacity": 0.5}
        return {"fillColor": "#d3d3d3", "color": "white", "weight": 1, "fillOpacity": 0.3}

    folium.GeoJson(
        geojson_data,
        name="geojson_dashboard",
        style_function=style_fn,
        popup=folium.GeoJsonPopup(fields=["state"], labels=False),
        tooltip=folium.GeoJsonTooltip(fields=["state"], aliases=["Provinsi:"]),
    ).add_to(m)
    return m

def _parse_clicked_region(popup: dict) -> str | None:
    if not popup: return None
    if popup.get("last_active_drawing"):
        props = popup["last_active_drawing"].get("properties", {})
        return str(props.get("state", "")).strip().lower()
    if popup.get("last_object_clicked_popup"):
        raw = str(popup["last_object_clicked_popup"]).strip().lower()
        return raw.replace("provinsi:", "").strip()
    return None

def _section_header(title: str):
    st.markdown(f"""
        <div style='background-color:#14213d;padding:10px;border-radius:5px 5px 0 0;color:white;text-align:center;'>
            <h5 style='margin:0;'>{title}</h5>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# MAIN APPLICATION FLOW
# ==========================================

# 1. Load Data/Model Global
geojson_data = load_geojson()
classifier = load_classifier()

# 2. Render Header Layout
st.markdown("""
    <div style='text-align: center; margin-bottom: 0;'>
        <h2 style='margin-bottom: 0;'>🎵 Dashboard Analisis & Eksplorasi Lagu Daerah</h2>
        <p style='font-size: 15px; color: gray;'>Sistem Identifikasi Otomatis Menggunakan XGBoost dan Ekstraksi Audio Feature</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 2px solid #fca311; margin-top: 5px; margin-bottom: 25px;'>", unsafe_allow_html=True)

# 3. Strategi Seamless UI: Tangkap Interaksi Klik Peta di Paling Atas Pipeline Render
if "peta_dashboard_utama" in st.session_state and st.session_state["peta_dashboard_utama"] is not None:
    popup_data = st.session_state["peta_dashboard_utama"]
    clicked_name = _parse_clicked_region(popup_data)
    
    if clicked_name and clicked_name in IRSD_MAPPING:
        db_region_name = IRSD_MAPPING[clicked_name]
        if st.session_state.current_region != db_region_name:
            st.session_state.current_region = db_region_name
            st.session_state.selected_song_info = None

# ─────────────────────────────────────────────────────────────────────────
# BAGIAN 1: PANEL PREDIKSI (LAYOUT COMPACT KIRI-KANAN)
# ─────────────────────────────────────────────────────────────────────────

col_input, col_hasil = st.columns([1.2, 1], gap="large")

# 📥 KOLOM 1: INPUT AUDIO & TOMBOL
with col_input:
    st.markdown("##### 🎙️ Input Audio Lagu Daerah")
    uploaded_file = st.file_uploader(
        "Unggah file audio (.wav / .mp3)", 
        type=["wav", "mp3"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if uploaded_file.name != st.session_state.last_uploaded_file:
            st.session_state.prediction_result = None
            st.session_state.last_uploaded_file = uploaded_file.name
            st.rerun()
            
        if st.button(
            "🚀 Jalankan Klasifikasi", 
            type="primary", 
            use_container_width=True,
            disabled=st.session_state.is_processing
        ):
            st.session_state.is_processing = True
            st.rerun()


if st.session_state.is_processing and st.session_state.prediction_result is None:
    with st.spinner("Menganalisis karakteristik audio dengan XGBoost..."):
        features = process_audio(uploaded_file)
        if features is not None:
            st.session_state.feature_dataframe = features.copy()
            
            # Memanggil fungsi predict yang mengembalikan 2 nilai (Provinsi, Confidence)
            pred_res, confidence = classifier.predict(features)
                
            st.session_state.prediction_result = pred_res
            st.session_state.prediction_confidence = confidence
            
            # Sinkronisasi Nama Wilayah ke GeoJSON Peta
            matched_db_name = None
            for key_geojson, val_db in IRSD_MAPPING.items():
                if str(pred_res).strip().lower() in [key_geojson.lower(), val_db.lower()]:
                    matched_db_name = val_db
                    break
            
            st.session_state.current_region = matched_db_name if matched_db_name else pred_res
    
    st.session_state.is_processing = False 
    st.rerun()


# 📊 KOLOM 2: HASIL DETEKSI & CONFIDENCE SCORE
with col_hasil:
    st.markdown("##### 🔍 Hasil Analisis Model")
    
    if st.session_state.prediction_result is not None:
        # 1. Hasil Deteksi Provinsi
        st.markdown(f"Asal Daerah:\n### {st.session_state.current_region.upper()}")
        
        # 2. Nilai Confidence Score
        conf_val = st.session_state.prediction_confidence * 100
        st.write(f"**Tingkat Keyakinan Model:** {conf_val:.2f}%")
        st.progress(st.session_state.prediction_confidence)
        
    else:
        # Keadaan Standby (Sebelum Tombol Klasifikasi Diklik)
        st.info("💡 **Petunjuk:**\nSilakan unggah file musik di kolom kiri, lalu klik **Jalankan Klasifikasi** untuk melihat hasil prediksi XGBoost.")

st.markdown("---")
# ─────────────────────────────────────────────────────────────────────────
# BAGIAN 2: PETA UTAMA
# ─────────────────────────────────────────────────────────────────────────
st.markdown("##### 🗺️ Peta Geografis Lagu Daerah")
m_full = _build_dashboard_map(geojson_data, st.session_state.current_region)

with st.container(height=415, border=False):
    st_folium(
        m_full,
        use_container_width=True,
        height=400,
        key="peta_dashboard_utama",
        returned_objects=["last_active_drawing", "last_object_clicked_popup"]
    )

st.write("")

# ─────────────────────────────────────────────────────────────────────────
# BAGIAN 3: PANEL EKSPLORASI DATA & BUDAYA (BAWAH)
# ─────────────────────────────────────────────────────────────────────────
if st.session_state.current_region:
    reg_name = st.session_state.current_region
    df_db = _cached_songs(reg_name)
    
    col_list, col_desc = st.columns([1.3, 1.2])
    
    # --- Kolom Kiri: Tampilan Berbentuk Card ---
    with col_list:
        _section_header(f"DAFTAR LAGU: {reg_name.upper()}")
        with st.container(border=True, height=450):
            if df_db is not None and not df_db.empty:
                actual_cols = df_db.columns.tolist()
                title_col = "judul_Lagu" if "judul_Lagu" in actual_cols else "judul_lagu"
                
                # Pengaturan Paginasi Card List
                total_rows = len(df_db)
                rows_per_page = 3  
                max_pages = math.ceil(total_rows / rows_per_page)
                
                page_key = f"dash_page_{reg_name.replace(' ', '_').lower()}"
                if page_key not in st.session_state: st.session_state[page_key] = 1
                
                curr_page = st.session_state[page_key]
                df_page = df_db.iloc[(curr_page - 1) * rows_per_page : curr_page * rows_per_page]
                
                for _, row in df_page.iterrows():
                    song = row[title_col]
                    artist = row['artists'] if 'artists' in row.index else "Unknown Artist"
                    yt_url = row['youtube_url'] if 'youtube_url' in row.index else None
                    
                    is_selected = st.session_state.selected_song_info == song
                    
                    # --- UPDATE BAGIAN 3: Render Loop Komponen Card Single Column ---
                for _, row in df_page.iterrows():
                    song = row[title_col]
                    artist = row['artists'] if 'artists' in row.index else "Unknown Artist"
                    yt_url = row['youtube_url'] if 'youtube_url' in row.index else None
                    
                    is_selected = st.session_state.selected_song_info == song
                    
                    # Setiap lagu dibungkus dalam satu container vertikal utuh
                    with st.container(border=True):
                        # Baris 1: Judul Lagu (Kiri) & Tombol Pilih (Kanan) -> Sejajar
                        c_title, c_btn = st.columns([2.5, 1])
                        
                        with c_title:
                            # Menggunakan sedikit margin atas CSS agar teks judul sejajar secara vertikal dengan tombol
                            st.markdown(f"<div style='margin-top: 2px; font-size: 25px;'><b>🎵 {song.upper()}</b></div>", unsafe_allow_html=True)
                            
                        with c_btn:
                            btn_type = "primary" if is_selected else "secondary"
                            
                            # 💡 SOLUSI: Tambahkan parameter disabled=is_selected
                            if st.button(
                                "Pilih Lagu", 
                                key=f"btn_dash_{song}", 
                                type=btn_type, 
                                use_container_width=True,
                                disabled=is_selected  # Tombol akan mati jika lagu ini sedang aktif
                            ):
                                st.session_state.selected_song_info = song
                                st.rerun()
                        
                        # Baris 2 & 3: Info Penyanyi & Player Video (Berada di bawah baris judul, melebar penuh)
                        st.markdown(f"👤 Penyanyi: *{artist}*")
                        
                        if yt_url and str(yt_url).strip() != "":
                            st.video(yt_url)
                        else:
                            st.caption("⚠️ *Tautan video tidak tersedia untuk lagu ini*")
                # Navigasi Halaman Card
                st.write("---")
                c_prev, c_page, c_next = st.columns([1, 1.5, 1])
                with c_prev:
                    if st.button("◀", disabled=(curr_page == 1), use_container_width=True, key="prev_dash"):
                        st.session_state[page_key] -= 1
                        st.session_state.selected_song_info = None
                        st.rerun()
                with c_page:
                    st.markdown(f"<p style='text-align:center;font-size:12px;margin-top:5px;'>Hal {curr_page}/{max_pages}</p>", unsafe_allow_html=True)
                with c_next:
                    if st.button("▶", disabled=(curr_page == max_pages), use_container_width=True, key="next_dash"):
                        st.session_state[page_key] += 1
                        st.session_state.selected_song_info = None
                        st.rerun()
            else:
                st.info(f"Belum ada daftar lagu untuk wilayah {reg_name} di database.")

    # --- Kolom Kanan: Deskripsi Budaya Gemini ---
    with col_desc:
        _section_header("MAKNA & FILOSOFI BUDAYA (GEMINI AI)")
        with st.container(border=True, height=450):
            if st.session_state.selected_song_info:
                s_name = st.session_state.selected_song_info
                st.markdown(f"<h4 style='text-align:center;color:#ff4b4b;margin-bottom:0;'>{s_name.upper()}</h4>", unsafe_allow_html=True)
                st.divider()
                with st.spinner("Menggali sejarah dan nilai filosofis..."):
                    info = get_song_info_from_gemini(s_name, reg_name)
                    st.write(info)
            else:
                st.write("")
                st.markdown("<p style='text-align:center;color:gray;margin-top:100px;'>👈 Silakan klik tombol <b>'Pilih Lagu'</b> pada card di samping untuk membaca makna budaya serta filosofinya.</p>", unsafe_allow_html=True)
else:
    st.info("💡 **Petunjuk:** Silakan upload file audio di atas untuk mendeteksi otomatis, atau langsung klik salah satu provinsi berwarna kuning di atas peta untuk mengeksplorasi.")