# data/models.py
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class JewelryItem:
    id: int
    name: str
    category: str
    model_path: str
    texture_path: str = None
    thumbnail_path: str = None
    # Stores slider values: {'B_Scale': 105, 'B_Rot_X': 90, ...}
    settings: Dict[str, Any] = field(default_factory=dict)