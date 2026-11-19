"""Start HarvestGuard:  python run.py   ->  http://127.0.0.1:5000"""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    host = os.environ.get("HG_HOST", "127.0.0.1")
    port = int(os.environ.get("HG_PORT", "5000"))
    try:                                   # production-grade server if installed
        from waitress import serve
        print(f"HarvestGuard running on http://{host}:{port} (waitress)")
        serve(app, host=host, port=port, threads=4)
    except ImportError:
        app.run(host=host, port=port, debug=os.environ.get("HG_DEBUG", "0") == "1")
