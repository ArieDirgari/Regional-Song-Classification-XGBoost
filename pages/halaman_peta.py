"""
Halaman Peta Interaktif: Eksplorasi lagu daerah per provinsi dengan deskripsi AI.
"""
import math
import os
import streamlit as st
import folium
from streamlit_folium import st_folium
from modules.database import get_songs_by_region

# ── Lazy import Google GenAI hanya saat halaman ini dimuat ───────────────────
from dotenv import load_dotenv
load_dotenv()

CACHE_DIR = "cache_ai"
os.makedirs(CACHE_DIR, exist_ok=True)

# Client diinisialisasi sekali per session (bukan per-render)
@st.cache_resource(show_spinner=False)
def _get_genai_client():
    from google import genai
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


@st.cache_data(show_spinner=False)
def get_song_info_from_gemini(judul_lagu: str, asal_daerah: str) -> str:
    """Ambil deskripsi lagu dari Gemini, dicache di memori server Streamlit."""
    try:
        client = _get_genai_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=(
                f"Berikan informasi singkat, menarik, dan edukatif tentang lagu daerah "
                f"'{judul_lagu}' dari {asal_daerah}. Jelaskan makna lagu "
                f"dan nilai budayanya. Gunakan bahasa yang santai namun informatif."
            ),
        )
        return response.text
    except Exception as e:
        return f"Terjadi kendala saat memuat deskripsi: {e}"


@st.cache_data(show_spinner=False)
def _cached_songs(region: str):
    """Cache hasil query DB per region agar tidak query ulang saat rerun."""
    return get_songs_by_region(region)


def _build_interactive_map(geojson_data: dict, irsd_mapping: dict) -> folium.Map:
    """Bangun peta interaktif dengan highlight provinsi yang ada di database."""
    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="CartoDB positron")

    def style_fn(feature):
        state = str(feature["properties"].get("state", "")).strip().lower()
        if state in irsd_mapping:
            return {"fillColor": "#ff4b4b", "color": "white", "weight": 1.5, "fillOpacity": 0.8}
        return {"fillColor": "#d3d3d3", "color": "white", "weight": 1, "fillOpacity": 0.5}

    folium.GeoJson(
        geojson_data,
        name="geojson_irsd",
        style_function=style_fn,
        popup=folium.GeoJsonPopup(fields=["state"], labels=False),
        tooltip=folium.GeoJsonTooltip(fields=["state"], aliases=["Provinsi:"]),
    ).add_to(m)
    return m


def _parse_clicked_region(popup: dict) -> str | None:
    """Ekstrak nama provinsi (lowercase) dari hasil klik peta."""
    if not popup:
        return None
    if popup.get("last_active_drawing"):
        props = popup["last_active_drawing"].get("properties", {})
        return str(props.get("state", "")).strip().lower() or None
    if popup.get("last_object_clicked_popup"):
        raw = str(popup["last_object_clicked_popup"]).strip().lower()
        return raw.replace("provinsi:", "").strip() or None
    return None


def _render_song_list(df_db, db_region_name: str):
    """Render daftar lagu dengan paginasi di kolom tengah."""
    actual_cols = df_db.columns.tolist()
    title_col = "judul_Lagu" if "judul_Lagu" in actual_cols else "judul_lagu"

    total_rows = len(df_db)
    rows_per_page = 5
    max_pages = math.ceil(total_rows / rows_per_page)

    page_key = f"page_{db_region_name.replace(' ', '_').lower()}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    current_page = st.session_state[page_key]
    start_idx = (current_page - 1) * rows_per_page
    df_page = df_db.iloc[start_idx : start_idx + rows_per_page]

    st.caption("Klik lagu untuk detail")
    for song in df_page[title_col].tolist():
        btn_style = "primary" if st.session_state.selected_song_info == song else "secondary"
        if st.button(song, use_container_width=True, type=btn_style, key=f"btn_{song}"):
            st.session_state.selected_song_info = song
            st.rerun()

    # Paginasi
    st.write("---")
    c_prev, c_page, c_next = st.columns([1, 1.5, 1])
    with c_prev:
        if st.button("◀", disabled=(current_page == 1), use_container_width=True, key=f"prev_{page_key}"):
            st.session_state[page_key] -= 1
            st.session_state.selected_song_info = None
            st.rerun()
    with c_page:
        st.markdown(
            f"<p style='text-align:center;font-size:11px;margin-top:5px;'>Hal {current_page}/{max_pages}</p>",
            unsafe_allow_html=True,
        )
    with c_next:
        if st.button("▶", disabled=(current_page == max_pages), use_container_width=True, key=f"next_{page_key}"):
            st.session_state[page_key] += 1
            st.session_state.selected_song_info = None
            st.rerun()


def _section_header(title: str):
    st.markdown(
        f"""
        <div style='background-color:#fca311;padding:10px;border-radius:5px 5px 0 0;
                    color:white;text-align:center;'>
            <h5 style='margin:0;'>{title}</h5>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(geojson_data: dict, irsd_mapping: dict):
    """Entry point halaman Peta Interaktif."""
    st.markdown("<h2 style='text-align: center;'>Sistem Prediksi Asal</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>Klik area pada peta untuk mengeksplorasi lagu daerah</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ── Peta ditampilkan FULL WIDTH di atas, di luar kolom ───────────────────
    m_full = _build_interactive_map(geojson_data, irsd_mapping)
    st.markdown(
        "<style>[data-testid='stHorizontalBlock'] iframe{width:100%;}</style>",
        unsafe_allow_html=True,
    )
    popup = st_folium(
        m_full,
        use_container_width=True,   # ← mengisi penuh lebar halaman
        height=500,
        key="peta_irsd_main",
        returned_objects=[
            "last_active_drawing",
            "last_object_clicked_popup"
        ]
    )

    clicked_name = _parse_clicked_region(popup)

    col_list, col_desc = st.columns([1.2, 1.5])

    # ── Panel bawah: hanya tampil jika provinsi valid diklik ─────────────────
    if clicked_name and clicked_name in irsd_mapping:
        db_region_name = irsd_mapping[clicked_name]

        # Reset pilihan lagu jika provinsi berganti
        if st.session_state.current_region != db_region_name:
            st.session_state.current_region = db_region_name
            st.session_state.selected_song_info = None

        # Query DB (di-cache agar tidak ulang tiap rerun)
        df_db = _cached_songs(db_region_name)

        # ── Kolom Kiri: Daftar Lagu ───────────────────────────────────────
        with col_list:
            _section_header(db_region_name.upper())
            with st.container(border=True, height=385):
                if df_db is not None and not df_db.empty:
                    _render_song_list(df_db, db_region_name)
                else:
                    st.info("Belum ada lagu.")

        # ── Kolom Kanan: Deskripsi AI ─────────────────────────────────────
        with col_desc:
            _section_header("Deskripsi Lagu")
            with st.container(border=True, height=385):
                if st.session_state.selected_song_info:
                    song_name = st.session_state.selected_song_info
                    st.markdown(
                        f"<h4 style='text-align:center;color:#ff4b4b;'>{song_name.upper()}</h4>",
                        unsafe_allow_html=True,
                    )
                    st.divider()
                    with st.spinner("Memuat deskripsi..."):
                        info = get_song_info_from_gemini(song_name, db_region_name)
                        st.write(info)
                else:
                    st.write("")
                    st.write("")
                    st.markdown(
                        "<p style='text-align:center;color:gray;'>"
                        "👈 Pilih salah satu lagu dari daftar di samping untuk melihat maknanya."
                        "</p>",
                        unsafe_allow_html=True,
                    )

    elif clicked_name:
        # Provinsi diklik tapi tidak ada di database
        with col_list:
            st.warning("Wilayah di luar cakupan database.")
        with col_desc:
            st.empty()
    else:
        # Belum ada klik sama sekali
        with col_list:
            st.info("Klik provinsi pada peta.")
        with col_desc:
            st.empty()