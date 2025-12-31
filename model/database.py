# model/databse.py
import sqlite3
import json
import os
from utils.paths import DATA_DIR
from model.models import JewelryItem

DB_PATH = DATA_DIR / "jewelry.db"

class JewelryDB:
    def __init__(self):
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Creates the table and handles schema updates."""
        os.makedirs(DATA_DIR, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Base Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jewelry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                model_path TEXT NOT NULL,
                texture_path TEXT,
                thumbnail_path TEXT,
                settings TEXT  -- New Column for JSON Slider Values
            )
        ''') 
        self.conn.commit()
        # self._seed_data_if_empty() # optional if dont like empty startups

    def _seed_data_if_empty(self):
        """Adds dummy data if the DB is new."""
        if len(self.get_all_items()) == 0:
            print("Database empty. Seeding default items...")
            # Add the bracelet you already have
            self.add_item(
                name="Gold Bangle",
                category="bracelet",
                model_path="data/3d_models/obj/3DModel.obj"
            )
            # Add a placeholder for a second item
            self.add_item(
                name="Silver Cuff",
                category="bracelet",
                model_path="data/3d_models/obj/3DModel.obj"
            )

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
            print(f" [DB] SUCCESS: Updated Item {item_id} with {len(settings_dict)} settings.")
        except Exception as e:
            print(f" [DB] ERROR: Could not save settings for {item_id}: {e}")

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