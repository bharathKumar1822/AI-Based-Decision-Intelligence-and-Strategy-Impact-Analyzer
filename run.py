"""
run.py  --  Launcher for the Decision Intelligence Analyzer
Usage:   python run.py
         python run.py --public    (also starts ngrok public tunnel)
"""

import os
import sys
import time
import socket
import threading
import webbrowser
import subprocess
from pathlib import Path

# ── Self-bootstrap: re-launch with the .venv Python if needed ──────────────
ROOT = Path(__file__).parent
_venv_python = ROOT / ".venv" / "Scripts" / "python.exe"

if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    # Not running inside the venv — re-launch with the correct interpreter
    raise SystemExit(subprocess.call([str(_venv_python)] + sys.argv))
# ───────────────────────────────────────────────────────────────────────────

# ROOT already defined above by the bootstrap block
PORT = 5000


def get_local_ip() -> str:
    """Return the LAN IP of this machine (for same-WiFi mobile access)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def try_ngrok(port: int):
    """Start an ngrok tunnel and return the public URL (if pyngrok is installed)."""
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(port, "http")
        return str(public_url)
    except ImportError:
        return None
    except Exception as e:
        print(f"  [!] ngrok failed: {e}")
        return None


def open_browser(url: str, delay: float = 2.5):
    """Wait briefly for Flask to start, then open the browser."""
    time.sleep(delay)
    webbrowser.open(url)


def main():
    use_ngrok = "--public" in sys.argv

    local_url   = f"http://127.0.0.1:{PORT}"
    network_ip  = get_local_ip()
    network_url = f"http://{network_ip}:{PORT}"

    # Optional ngrok public tunnel
    public_url = None
    if use_ngrok:
        print("  Starting ngrok tunnel...")
        public_url = try_ngrok(PORT)

    # Open browser automatically
    t = threading.Thread(target=open_browser, args=(local_url,), daemon=True)
    t.start()

    # Print access URLs
    sep = "=" * 62
    print(sep)
    print("  AI Decision Intelligence Analyzer")
    print(sep)
    print(f"  [PC]     Local    -->  {local_url}")
    print(f"  [Mobile] Network  -->  {network_url}")
    if public_url:
        print(f"  [Any]    Public   -->  {public_url}")
    else:
        print("  [Any]    Public   -->  run:  python run.py --public")
    print("-" * 62)
    print(f"  MOBILE: connect phone to same WiFi, open: {network_url}")
    print("-" * 62)
    print("  Press Ctrl+C to stop the server")
    print(sep)

    # Start Flask
    os.chdir(ROOT)
    backend = ROOT / "backend" / "app.py"

    try:
        subprocess.run([sys.executable, str(backend)], check=True)
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
