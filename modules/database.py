import sqlite3
import pandas as pd

def get_songs_by_region(region_name):
    conn = sqlite3.connect('data/music_data.db')
    query = "SELECT artists, judul_Lagu, youtube_url FROM songs WHERE provinsi = ?"
    df = pd.read_sql(query, conn, params=(region_name,))
    conn.close()
    return df
import sqlite3
import json

def get_geometry_by_region(region_name):
    conn = sqlite3.connect('data/musicdata.db')
    cursor = conn.cursor()
    
    # Ambil kolom geometry_data berdasarkan nama region
    query = "SELECT geometry_data FROM geodata WHERE nama_provinsi = ?"
    cursor.execute(query, (region_name,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        # result[0] adalah string text dari database, kita ubah jadi dict python
        return json.loads(result[0])
    return None

def get_all_geometries():
    """Mengambil semua data geometri dari database untuk peta utama."""
    conn = sqlite3.connect('data/musicdata.db')
    cursor = conn.cursor()
    
    # Ambil semua baris geometri
    query = "SELECT nama_provinsi, geometry_data FROM geodata"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    features = []
    for row in rows:
        region_name = row[0]
        # Data di database Anda sudah berupa string Feature, kita ubah jadi dict
        feature = json.loads(row[1])
        # Pastikan key 'state' ada di properties agar tooltip tidak error
        feature['properties']['state'] = region_name
        features.append(feature)
    
    # Bungkus ke dalam format FeatureCollection GeoJSON
    return {
        "type": "FeatureCollection",
        "features": features
    }