# 1. Clone your repo or copy your files over
git clone <your-repo-url>
cd JewelleryTryon

# 2. install python & Create a virtual environment (This is your "Docker" replacement)
Install Python 3.12 for Windows.
python -m venv venv

# 3. Activate it
.\venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. running app directly
python3 main.py

# pyinstallercmds

pip install pyinstaller
pyinstaller --noconfirm --onedir --windowed --name "JewelryTryOn" `
 --add-data "model;model" `
 --add-data "ui;ui" `
 --add-data "graphics;graphics" `
 --add-data "trackers;trackers" `
 --add-data "utils;utils" `
 --hidden-import "mediapipe" `
 --hidden-import "cv2" `
 --hidden-import "numpy" `
 main.py
