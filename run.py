"""
Script to run the Streamlit app with ngrok for public access.
"""
import subprocess
import sys
from pyngrok import ngrok
from config import NGROK_AUTH_TOKEN


def setup_ngrok():
    """Set up ngrok with auth token."""
    if NGROK_AUTH_TOKEN:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    else:
        print("Warning: NGROK_AUTH_TOKEN not found in environment variables.")
        print("Please set your ngrok auth token in the .env file.")


def run_app():
    """Run the Streamlit app with ngrok."""
    try:
        # Setup ngrok
        setup_ngrok()
        
        # Kill any existing ngrok processes
        ngrok.kill()
        
        # Start ngrok tunnel
        public_url = ngrok.connect(8501)
        print(f"🚀 Access your app here: {public_url}")
        
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py", 
            "--server.port", "8501"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        ngrok.kill()
    except Exception as e:
        print(f"❌ Error running app: {e}")
        ngrok.kill()


if __name__ == "__main__":
    run_app()