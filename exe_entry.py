"""
exe_entry.py — PyInstaller entry point
Runs Flask in a background thread + opens the browser automatically.
"""

import sys
import os
import threading
import webbrowser
import time

# ── Make sure the bundled packages are importable ───────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
    # Add MEIPASS to path so `backend` package is found
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PORT = 5000

def open_browser():
    time.sleep(3.0)          # give Flask time to start
    webbrowser.open(f"http://127.0.0.1:{PORT}")

def main():
    # Print banner to any visible console
    print("=" * 54)
    print("  🧠  AI Decision Intelligence Analyzer")
    print(f"  → Opening http://127.0.0.1:{PORT}")
    print("  Close this window to stop the application.")
    print("=" * 54)

    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    # Import and run Flask (paths patched via sys._MEIPASS in app.py)
    from backend.app import app
    app.run(debug=False, port=PORT, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
