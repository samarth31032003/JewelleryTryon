# main.py
import sys
import os

# Ensure we can find our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from ui.tryon_view import TryOnWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TryOnWindow()
    window.show()
    sys.exit(app.exec_())