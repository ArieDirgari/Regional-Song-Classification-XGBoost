import streamlit as st

def render():
    # CSS khusus untuk gaya desain kartu menu di Beranda
    st.markdown("""
        <style>
        .menu-box {
            border: 3px solid #fca311;
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            background-color: white;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        }
        .icon-style {
            font-size: 50px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Konten Judul Utama
    st.write("")
    st.write("")
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>Sistem Prediksi Asal</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #fca311; margin-top: 0; font-size: 45px;'>Lagu Daerah Indonesia</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; font-size: 16px;'>Berbasis MFCC dan XGBoost untuk mengklasifikasikan asal lagu daerah provinsi di Indonesia</p>", unsafe_allow_html=True)
    st.write("")
    st.write("")

    # Menu Pilihan Dua Kotak (UI Cards)
    col_space1, col_card1, col_space2, col_card2, col_space3 = st.columns([0.5, 2, 0.3, 2, 0.5])

    with col_card1:
        st.markdown("""
            <div class='menu-box'>
                <div class='icon-style'>🎙️</div>
                <h3>Upload Audio</h3>
                <p style='color: gray; font-size: 14px; min-height: 50px;'>Unggahan audio lagu untuk memprediksi asal daerah</p>
            </div>
        """, unsafe_allow_html=True)
        # Menggunakan logika routing session_state alih-alih switch_page
        if st.button("Mulai Deteksi ➔", use_container_width=True, key="go_deteksi"):
            st.session_state.menu = "Deteksi"
            st.rerun()

    with col_card2:
        st.markdown("""
            <div class='menu-box'>
                <div class='icon-style'>🗺️</div>
                <h3>Peta Interaktif</h3>
                <p style='color: gray; font-size: 14px; min-height: 50px;'>Jelajahi informasi lagu daerah berdasarkan wilayah provinsi</p>
            </div>
        """, unsafe_allow_html=True)
        # Menggunakan logika routing session_state alih-alih switch_page
        if st.button("Lihat Peta ➔", use_container_width=True, key="go_peta"):
            st.session_state.menu = "Peta Interaktif"
            st.rerun()