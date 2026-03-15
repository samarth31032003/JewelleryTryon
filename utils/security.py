# utils/security.py
import subprocess
import urllib.request
import json
import platform
import time
from model.database import JewelryDB
from utils.logger import logger

log = logger.bind(component="utils")
# try-except so it doesn't crash on Linux
try:
    import winreg
except ImportError:
    pass

class LicenseGuard:
    # Your Cloudflare Worker URL
    API_URL = "https://license-api.ee-irfansmail.workers.dev/check?hwid="

    @staticmethod
    def get_hwid():
        """Generates a unique hardware ID locked to the Windows Motherboard."""
        try:
            if platform.system() == "Windows":
                # Safest method for compiled EXEs: Read the OS Cryptography Machine GUID
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                        hwid, _ = winreg.QueryValueEx(key, "MachineGuid")
                        return hwid.upper()
                except Exception as e:
                    print(f"[License] Registry read failed: {e}. Falling back to WMIC.")
                    # Fallback just in case
                    hwid = subprocess.check_output(
                        'wmic csproduct get uuid', 
                        shell=True, 
                        stdin=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    ).decode().split('\n')[1].strip()
                    return hwid
            else:
                # Fallback for Linux Dev
                return "LINUX_DEV_ID"
        except Exception as e:
            # Added a print statement so it doesn't fail silently.
            log.error(f"[License Guard Debug] HWID generation failed: {e}")
            return "UNKNOWN-HWID"

    @staticmethod
    def validate_license():
        """Checks the Cloudflare API to see if this HWID is active."""
        hwid = LicenseGuard.get_hwid()
        log.info(f"Checking License for HWID: {hwid}")
        
        # Bypass for local Linux development
        # if hwid == "LINUX_DEV_ID": 
        #     log.info("[License] Linux Dev Mode: Bypassing network check.")
        #     return True        
        
        db = JewelryDB()
        current_time = time.time()
        
        try:
            url = f"{LicenseGuard.API_URL}{hwid}"
            
            # ---  Fake a Chrome Browser User-Agent for cloudflare---
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            req = urllib.request.Request(url, headers=headers)

            # 5 second timeout so the app doesn't freeze forever if internet is slow
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                
                if data.get("status") == "active":
                    log.info(f"[License] internet! Validated successfully for client: {data.get('client', 'Unknown')}")
                    
                    # Update DB with current time for offline mode later
                    db.save_license_state(current_time, current_time, hwid)
                    db.close()
                    return True
                
                log.warning("[License] internet! Blocked: License is revoked or unregistered.")
                db.close()
                return False
                
        except Exception as e:
            # If there's no internet or the API is unreachable, we check the offline grace period.
            log.warning(f"[License] no internet! Connection failed: {e}. Checking offline grace period...")
            
            state = db.get_license_state(hwid)
            if not state:
                log.error("[License] Offline Check Failed: Missing or tampered license state.")
                db.close()
                return False
                
            last_online, last_seen = state
            
            # Time-Travel Check: Clock should not go backward
            if current_time < last_seen - 300: # 5 min tolerance for minor clock drift
                log.error("[License] Offline Check Failed: System clock rewind detected.")
                db.close()
                return False
                
            # Grace Period Check: Max 3 days (3 * 24 * 3600 seconds)
            three_days_sec = 3 * 24 * 3600
            time_offline = current_time - last_online
            
            days_offline = time_offline / (24 * 3600)

            if time_offline > three_days_sec:
                log.error(f"[License] Offline Check Failed: Grace period expired ({days_offline:.1f} days offline).")
                db.close()
                return False
                
            # Valid offline session: update last_seen, keep last_online
            log.info(f"[License] Offline Grace Period Valid. Updating last_seen timestamp.")
            db.save_license_state(last_online, current_time, hwid)
            db.close()
            return True

