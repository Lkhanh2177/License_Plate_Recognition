import sqlite3

DB_PATH = "license_plates.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_origin TEXT,
                plate_text TEXT,
                start_time TEXT,
                end_time TEXT,
                image_crop BLOB,
                direction TEXT,
                camera_location TEXT
            )
        ''')
    conn.commit()
    conn.close()
