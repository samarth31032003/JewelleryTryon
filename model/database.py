# model/database.py
import sqlite3
import json
import os
import hashlib
import secrets
from utils.paths import DATA_DIR
from model.models import JewelryItem

DB_PATH = DATA_DIR / "jewelry.db"

class JewelryDB:
    def __init__(self):
        self.conn = None
        self._init_db()
        self._ensure_default_admin()

    def _init_db(self):
        """Creates tables and handles schema updates."""
        os.makedirs(DATA_DIR, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # 1. Base Table (Jewelry)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jewelry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                model_path TEXT NOT NULL,
                texture_path TEXT,
                thumbnail_path TEXT,
                settings TEXT  -- Stores JSON Slider Values
            )
        ''')
        
        # 2. Auth Table (Secure Password)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL
            )
        """)
        
        self.conn.commit()

    def _ensure_default_admin(self):
        """Sets default password 'admin' if no password exists."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT count(*) FROM auth")
        if cursor.fetchone()[0] == 0:
            print("[DB] First run detected. Setting default password: 'admin'")
            self.set_password("admin")

    # --- SECURITY METHODS ---

    def set_password(self, plain_password):
        """Hashes and saves a new password."""
        salt = secrets.token_hex(16) 
        # Hash = SHA256( salt + password )
        p_hash = hashlib.sha256((salt + plain_password).encode()).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM auth WHERE id=1") # Clear old
        cursor.execute("INSERT INTO auth (id, password_hash, salt) VALUES (1, ?, ?)", (p_hash, salt))
        self.conn.commit()

    def verify_password(self, plain_input):
        """Checks if input matches the stored hash."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT password_hash, salt FROM auth WHERE id=1")
        row = cursor.fetchone()
        if not row: return False
        
        stored_hash, salt = row
        input_hash = hashlib.sha256((salt + plain_input).encode()).hexdigest()
        
        return input_hash == stored_hash

    # --- JEWELRY CRUD OPERATIONS ---

    def add_item(self, name, category, model_path, texture_path=None, thumbnail_path=None):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO jewelry (name, category, model_path, texture_path, thumbnail_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, category, model_path, texture_path, thumbnail_path))
        self.conn.commit()

    def update_item_settings(self, item_id, settings_dict):
        """Saves slider values (JSON) for a specific item."""
        try:
            cursor = self.conn.cursor()
            json_str = json.dumps(settings_dict)
            cursor.execute('UPDATE jewelry SET settings = ? WHERE id = ?', (json_str, item_id))
            self.conn.commit()
            print(f" [DB] Saved settings for Item {item_id}")
        except Exception as e:
            print(f" [DB] ERROR saving settings: {e}")

    def get_all_items(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM jewelry')
        rows = cursor.fetchall()
        
        items = []
        for row in rows:
            # Map Row -> Object
            # Row: 0=id, 1=name, 2=cat, 3=path, 4=tex, 5=thumb, 6=settings
            settings = json.loads(row[6]) if row[6] else {}
            
            item = JewelryItem(
                id=row[0], name=row[1], category=row[2], 
                model_path=row[3], texture_path=row[4], 
                thumbnail_path=row[5], settings=settings
            )
            items.append(item)
        return items

    def delete_item(self, item_id):
        """Removes an item from the database by ID."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM jewelry WHERE id = ?", (item_id,))
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()