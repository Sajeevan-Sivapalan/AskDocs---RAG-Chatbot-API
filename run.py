"""
Run the RAG Chatbot server.
Usage:  python run.py
        python run.py --host 0.0.0.0 --port 8080 --reload
"""

import argparse
import uvicorn
from dotenv import load_dotenv

load_dotenv()   # reads .env if present

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   default=8000, type=int)
    parser.add_argument("--reload", action="store_true",
                        help="Enable hot-reload (development only)")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
        log_level = "info",
    )
