"""
Halaman Deteksi: Upload audio dan klasifikasi asal daerah lagu.
"""
import streamlit as st
import folium
from streamlit_folium import st_folium
from modules.processor import process_audio


def _build_result_map(geojson_data: dict, prediction: str, irsd_mapping: dict) -> folium.Map:
    """Buat peta mini yang menyorot provinsi hasil prediksi."""
    m = folium.Map(location=[-2.5, 118], zoom_start=4, tiles="CartoDB positron")

    # Cari kunci GeoJSON yang sesuai dengan hasil prediksi
    matched_keys = [k for k, v in irsd_mapping.items() if v.lower() == str(prediction).lower()]

    def style_fn(feature):
        geo_state = str(feature["properties"].get("state", "")).strip().lower()
        if matched_keys and geo_state == matched_keys[0]:
            return {"fillColor": "#ff4b4b", "color": "black", "weight": 2, "fillOpacity": 0.8}
        return {"fillColor": "#d3d3d3", "color": "white", "weight": 1, "fillOpacity": 0.5}

    folium.GeoJson(geojson_data, style_function=style_fn).add_to(m)
    return m


def render(geojson_data: dict, classifier, irsd_mapping: dict):
    """Entry point halaman Deteksi."""
    st.markdown("<h2 style='text-align: center;'>Sistem Prediksi Asal</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>Upload suara lagu daerah untuk mengetahui asal daerah</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ── Upload & Klasifikasi ──────────────────────────────────────────────────
    _, col_uploader, _ = st.columns([1, 2, 1])
    with col_uploader:
        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "Upload File Audio (*.wav / *.mp3)", type=["wav", "mp3"]
            )

            # Reset hasil jika file baru diunggah
            if uploaded_file != st.session_state.last_uploaded_file:
                st.session_state.prediction_result = None
                st.session_state.feature_dataframe = None
                st.session_state.last_uploaded_file = uploaded_file

            if uploaded_file is not None:
                st.audio(uploaded_file)
                if st.button("Jalankan Klasifikasi", use_container_width=True, type="primary"):
                    with st.spinner("Menganalisis karakteristik audio..."):
                        features = process_audio(uploaded_file)

                        if features is not None:

                            # simpan dataframe hasil ekstraksi
                            st.session_state.feature_dataframe = features.copy()

                            # klasifikasi
                            st.session_state.prediction_result = classifier.predict(features)

                # ── Tampilkan Hasil ───────────────────────────────────────────────────────
            if st.session_state.prediction_result is not None:

                prediction = st.session_state.prediction_result

                st.write("")
                st.write("")

                col_res, col_map_res = st.columns(2)

                with col_res:
                    with st.container(border=True):

                        st.markdown("<b>Hasil Deteksi</b>", unsafe_allow_html=True)

                        st.markdown(
                            f"""
                            <h2 style='text-align:center;
                                    margin-top:40px;
                                    margin-bottom:40px;'>
                                {prediction}
                            </h2>
                            """,
                            unsafe_allow_html=True
                        )

                        st.caption("Provinsi")

                with col_map_res:
                    with st.container(border=True):

                        st.markdown("<b>Lokasi Peta</b>", unsafe_allow_html=True)

                        m_res = _build_result_map(geojson_data,prediction,irsd_mapping
                        )

                        st_folium(
                            m_res,
                            width=400,
                            height=200,
                            key="map_result"
                        )

                st.caption("*Hasil prediksi berdasarkan model MFCC dan XGBoost")

                # =====================================================
                # HASIL EKSTRAKSI FITUR
                # =====================================================

                if st.session_state.feature_dataframe is not None:

                    feature_df = (
                        st.session_state.feature_dataframe
                        .T
                        .reset_index()
                    )

                    feature_df.columns = [
                        "Fitur",
                        "Nilai"
                    ]

                    st.write("")

                    with st.expander(
                        "📊 Lihat Hasil Ekstraksi Fitur",
                        expanded=False
                    ):

                        st.caption(
                            f"Jumlah fitur: {len(feature_df)}"
                        )

                        st.dataframe(
                            feature_df,
                            use_container_width=True,
                            height=600
                        )