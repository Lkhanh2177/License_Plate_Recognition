import sqlite3

DB_PATH = "license_plates.db"

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT,
        end_time TEXT,
        plate_origin TEXT,
        plate_text TEXT,
        image_crop BLOB
    )
""")

conn.commit()
conn.close()
