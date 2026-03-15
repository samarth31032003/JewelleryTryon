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

# 5. Download MediaPipe Task models (~26MB, required once)
python utils/download_models.py

# 6. running app directly
python main.py

# pyinstallercmds

pip install pyinstaller
pyinstaller --noconfirm --onedir --windowed --name "JewelryTryOn" `
 --add-data "data/assets;data/assets" `
 --hidden-import "mediapipe" `
 --collect-data mediapipe `
 --hidden-import "cv2" `
 --hidden-import "numpy" `
 --icon "data/assets/icon.png" `
 main.py
