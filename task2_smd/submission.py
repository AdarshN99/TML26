import os
import sys
import requests
from pathlib import Path
import pandas as pd

# --------------------------------
# SUBMISSION PROCESS
# --------------------------------

BASE_URL = "http://34.63.153.158"
API_KEY = ""  # replace with your actual API key

BASE = Path(__file__).parent

TASK_ID = "19-stolen-model-detection"
FILE_PATH = BASE / "submission_wsim.csv"  # replace with your actual file path

SUBMIT = True  # Set to True to enable submission

def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)

if SUBMIT:
    if not os.path.isfile(FILE_PATH):
        die(f"File not found: {FILE_PATH}")

    try:
        with open(FILE_PATH, "rb") as f:
            files = {
                # (fieldname) -> (filename, fileobj, content_type)
                "file": (os.path.basename(FILE_PATH), f, "csv"),
            }
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120),  # (connect timeout, read timeout)
            )
        # Helpful output even on non-2xx
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

        resp.raise_for_status()

        submission_id = body.get("submission_id")
        print("Successfully submitted.")
        print("Server response:", body)
        if submission_id:
            print(f"Submission ID: {submission_id}")

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)
