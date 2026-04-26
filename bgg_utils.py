import requests
import time
import random
import re
import os
from pathlib import Path

# --- API Configuration ---
TOKEN = os.environ.get("BGG_API_TOKEN", "")
BASE_URL = "https://boardgamegeek.com/xmlapi2"
HEADERS = {
    "User-Agent": "BGG-Research/1.0",
    "Authorization": f"Bearer {TOKEN}"
}

# --- Shared Utilities ---

def fetch_api(endpoint, params, max_attempts=5, delay_range=(1.0, 2.5)):
    """
    Unified API requester with retry logic and BGG-specific status handling.
    """
    url = f"{BASE_URL}/{endpoint}"
    
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, params=params, headers=HEADERS)
            
            if r.status_code == 200:
                # BGG 202-like behavior (processing message in 200)
                if "Your request for this collection has been accepted and will be processed" in r.text or \
                   ("item" not in r.text and "items" not in r.text and "<errors>" not in r.text):
                    time.sleep(5 + attempt * 2)
                    continue
                return r.text
            
            elif r.status_code == 202:
                time.sleep(5 + attempt * 2)
                continue
                
            elif r.status_code == 429:
                # Rate limit
                time.sleep(10 + attempt * 5)
                continue
            
            else:
                print(f"\n[Warning] BGG API returned status {r.status_code} for {endpoint}")
                break

        except Exception as e:
            print(f"\n[Error] Connection error for {endpoint}: {e}")
            time.sleep(5 + attempt * 2)

    return None

def sanitize_filename(name):
    """Sanitize a string to be used as a valid filename."""
    return "".join([c for c in str(name) if c.isalnum() or c in (' ', '.', '_', '-')]).strip()

def check_xml_errors(content):
    """Checks if XML content indicates a BGG error or empty result."""
    if not content:
        return True
    if "<errors>" in content or "<error>" in content:
        return True
    if "<items" not in content and "<item" not in content:
        return True
    return False

def chunk_list(lst, size):
    """Yield successive chunks from a list."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]
