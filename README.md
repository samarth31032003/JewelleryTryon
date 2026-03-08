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

antigrav prompt
Implement Secure Password Changing and Cloud-OTP Verification

I need to implement two flows for changing the admin password: an authenticated flow in the Settings menu, and an unauthenticated "Forgot Password" flow utilizing our Cloudflare Worker OTP backend.

Phase 1: Update Database to handle Admin Email

In model/database.py, update _ensure_default_admin to also save a default recovery email (e.g., 'admin@sringar.com') to the app_settings table using your existing save_setting method.

Phase 2: Authenticated Password Change (Settings Menu)

In ui/settings.py, add a new section below the Camera selector.

Add a QLineEdit for "Recovery Email" (populated from db.get_setting('admin_email')).

Add three QLineEdit fields with setEchoMode(QLineEdit.Password): "Current Password", "New Password", and "Confirm New Password".

Add a "Update Credentials" button. When clicked:

Save the Recovery Email.

If the password fields are filled, verify the current password using db.verify_password(). If valid and new passwords match, call db.set_password(). Show appropriate QMessageBox successes or errors.

Phase 3: Forgot Password Flow (Unauthenticated)

In ui/login.py, create a new ForgotPasswordDialog.

Step 1 (Request OTP): Ask the user for their Recovery Email. If it matches db.get_setting('admin_email'), use the requests library to make a POST request to https://license-api.ee-irfansmail.workers.dev/send-otp. Include the header {"Authorization": "YOUR_APP_SECRET"} and body {"email": "their_email"}.

Step 2 (Verify OTP): Change the dialog UI to ask for the 6-digit code. Make a POST request to https://license-api.ee-irfansmail.workers.dev/verify-otp with the email and code.

Step 3 (Reset Password): If the Cloudflare Worker returns {"success": true}, change the dialog UI to ask for a "New Password" and "Confirm Password". Call db.set_password() and close the dialog with a success message.

Add a "Forgot Password?" button or clickable label to the main LoginWindow that triggers this dialog.

Use the existing loguru logger to log these security events appropriately.