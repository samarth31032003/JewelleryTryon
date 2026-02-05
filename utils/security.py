# utils/security.py
import subprocess
import requests
import sys
import platform

class LicenseGuard:
    # Replace this with your raw GitHub/Vercel JSON URL later
    LICENSE_URL = "https://your-static-site.com/allowed_devices.json"
    
    @staticmethod
    def get_hwid():
        """Gets a unique Motherboard/System ID."""
        try:
            if platform.system() == "Windows":
                cmd = 'wmic csproduct get uuid'
                # Run command, decode output, strip headers
                output = subprocess.check_output(cmd, shell=True).decode()
                return output.split('\n')[1].strip()
            else:
                # Fallback for Linux Dev
                return "LINUX_DEV_ID"
        except Exception:
            return "UNKNOWN_ID"

    @staticmethod
    def validate_license():
        """
        Fetches the allowed list from the web.
        Returns True if allowed, False if banned/unlicensed.
        """
        my_id = LicenseGuard.get_hwid()
        print(f"Checking License for HWID: {my_id}")
        
        # DEV BYPASS: If you are developing, you don't want to lock yourself out
        # Remove this block when shipping to client!
        if my_id == "LINUX_DEV_ID": 
            return True

        try:
            # 1. Fetch the JSON list
            # Format expected: {"allowed": ["UUID-1", "UUID-2"], "banned": []}
            response = requests.get(LicenseGuard.LICENSE_URL, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if my_id in data.get("allowed", []):
                    return True
            
            print("License Verification Failed: ID not found in database.")
            return False
            
        except Exception as e:
            print(f"License Server Error: {e}")
            # DECISION: Do you allow offline usage? 
            # If yes, return True here. If strict DRM, return False.
            # For now, let's Fail Open (Allow) if internet is down to be nice? 
            # Or Fail Closed (Block) for security?
            # Let's Fail Closed for safety:
            return False