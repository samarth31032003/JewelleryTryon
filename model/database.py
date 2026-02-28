# model/database.py
import sqlite3
import json
import os
import hashlib
import secrets
from utils.paths import DB_PATH, resolve_path
from model.models import JewelryItem

class JewelryDB:
    def __init__(self):
        self.conn = None
        self._init_db()
        self._ensure_default_admin()

    def _init_db(self):
        """Creates tables with the final schema (requires deleting the old .db file first)."""
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        cursor = self.conn.cursor()
        
        # 1. Base Table: Jewelry
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jewelry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                model_path TEXT NOT NULL,
                texture_path TEXT,
                thumbnail_path TEXT,
                image_2d_path TEXT,
                settings TEXT,
                details TEXT DEFAULT ''
            )
        ''')
        
        # 2. Auth & License Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                last_online REAL,
                last_seen REAL,
                license_sig TEXT
            )
        """)
        
        # 3. MIGRATION (optional for dev changes.)
        cursor.execute("PRAGMA table_info(jewelry)")
        columns = [info[1] for info in cursor.fetchall()]
        if "details" not in columns:
            print("[DB] Migrating: Adding 'details' column...")
            cursor.execute("ALTER TABLE jewelry ADD COLUMN details TEXT DEFAULT ''")
        
        # Migrations for auth table
        cursor.execute("PRAGMA table_info(auth)")
        auth_columns = [info[1] for info in cursor.fetchall()]
        if "last_online" not in auth_columns:
            print("[DB] Migrating: Adding license tracking columns to auth table...")
            cursor.execute("ALTER TABLE auth ADD COLUMN last_online REAL")
            cursor.execute("ALTER TABLE auth ADD COLUMN last_seen REAL")
            cursor.execute("ALTER TABLE auth ADD COLUMN license_sig TEXT")
        
        # delete the db anyway.
        # if "image_2d_path" not in columns:
        #     print("[DB] Migrating: Adding 'image_2d_path' column...")
        #     cursor.execute("ALTER TABLE jewelry ADD COLUMN image_2d_path TEXT")
            
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
        p_hash = hashlib.sha256((salt + plain_password).encode()).hexdigest()
        cursor = self.conn.cursor()
        # Ensure we only ever have row id=1
        cursor.execute("DELETE FROM auth WHERE id=1")
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

    def _generate_license_sig(self, last_online, last_seen, salt, hwid):
        # We include the HWID to bind the signature to the specific machine
        data = f"{last_online}_{last_seen}_{salt}_{hwid}"
        return hashlib.sha256(data.encode()).hexdigest()

    def save_license_state(self, last_online, last_seen, hwid):
        """Saves current timestamps with an anti-tamper signature."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT salt FROM auth WHERE id=1")
        row = cursor.fetchone()
        if not row:
            return False # Admin password not set up yet
            
        salt = row[0]
        sig = self._generate_license_sig(last_online, last_seen, salt, hwid)
        
        cursor.execute("""
            UPDATE auth 
            SET last_online = ?, last_seen = ?, license_sig = ? 
            WHERE id=1
        """, (last_online, last_seen, sig))
        self.conn.commit()
        return True

    def get_license_state(self, hwid):
        """Retrieves and validates timestamps. Returns (last_online, last_seen) or None if tampered/missing."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT last_online, last_seen, license_sig, salt FROM auth WHERE id=1")
        row = cursor.fetchone()
        
        if not row or row[0] is None or row[1] is None or row[2] is None:
            return None
            
        last_online, last_seen, stored_sig, salt = row
        
        expected_sig = self._generate_license_sig(last_online, last_seen, salt, hwid)
        if expected_sig != stored_sig:
            print("[DB] Warning: License signature mismatch. Tampering detected.")
            return None
            
        return last_online, last_seen

    # --- JEWELRY CRUD ---
    def add_item(self, name, category, model_path, texture_path=None, thumbnail_path=None, image_2d_path=None, details=""):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO jewelry (name, category, model_path, texture_path, thumbnail_path, image_2d_path, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, category, model_path, texture_path, thumbnail_path, image_2d_path, details))
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
        
        cursor.execute('''
            SELECT id, name, category, model_path, texture_path, 
                   thumbnail_path, image_2d_path, settings, details 
            FROM jewelry
        ''')
        rows = cursor.fetchall()
        
        items = []
        for row in rows:
            settings = json.loads(row[7]) if row[7] else {}
            details_text = row[8] if row[8] else ""
            
            abs_model_path = resolve_path(row[3])
            abs_tex_path = resolve_path(row[4]) if row[4] else ""
            abs_thumb_path = resolve_path(row[5]) if row[5] else ""
            abs_img2d_path = resolve_path(row[6]) if row[6] else ""
            
            item = JewelryItem(
                id=row[0], name=row[1], category=row[2], 
                model_path=abs_model_path,    
                texture_path=abs_tex_path,    
                thumbnail_path=abs_thumb_path,
                image_2d_path=abs_img2d_path,
                settings=settings,
                details=details_text
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