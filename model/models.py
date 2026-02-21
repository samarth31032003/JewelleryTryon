# model/models.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class JewelryItem:
    id: int
    name: str
    category: str
    model_path: str
    texture_path: Optional[str] = None
    thumbnail_path: Optional[str] = None 
    image_2d_path: Optional[str] = None 
    settings: dict = None
    details: str = ""