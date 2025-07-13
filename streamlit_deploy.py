#!/usr/bin/env python3
"""
Streamlit Deployment Test Script
Run this to test the Streamlit app locally before deploying to Streamlit Cloud.
"""

import os
import subprocess
import sys

def test_streamlit_app():
    """Test the Streamlit app locally."""
    print("🧪 Testing Streamlit app for deployment...")
    
    # Set default backend URL for local testing
    os.environ["BACKEND_URL"] = "http://localhost:8000"
    
    try:
        # Change to frontend directory
        os.chdir("frontend")
        
        # Run Streamlit app
        print("🚀 Starting Streamlit app...")
        print("📱 Frontend will be available at: http://localhost:8501")
        print("⚠️  Note: Backend needs to be running on port 8000 for full functionality")
        print("⏹️  Press Ctrl+C to stop")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501", "--server.address", "0.0.0.0"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")
    finally:
        # Return to original directory
        os.chdir("..")

if __name__ == "__main__":
    test_streamlit_app() 