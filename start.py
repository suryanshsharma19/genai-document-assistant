#!/usr/bin/env python3
"""
GenAI Document Assistant - Startup Script
A simple script to start the application in development mode.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    try:
        import fastapi
        import streamlit
        import uvicorn
        print("✅ All Python dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def check_spacy_model():
    """Check if spaCy model is installed."""
    print("🔍 Checking spaCy model...")
    print("✅ spaCy model check skipped (not required)")
            return True

def check_env_file():
    """Check if .env file exists."""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Creating .env file from template...")
        try:
            if Path("env.example").exists():
                import shutil
                shutil.copy("env.example", ".env")
                print("✅ .env file created from template")
                print("⚠️  Please edit .env file with your API keys")
                return False
            else:
                print("❌ env.example file not found")
                return False
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("✅ .env file exists")
        return True

def start_backend():
    """Start the backend server."""
    print("🚀 Starting backend server...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return None
    
    try:
        # Change to backend directory
        os.chdir(backend_dir)
        
        # Start uvicorn server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app",
            "--reload", "--host", "0.0.0.0", "--port", "8000"
        ])
        
        print("✅ Backend server started on http://localhost:8000")
        return process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the frontend server."""
    print("🚀 Starting frontend server...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None
    
    try:
        # Change to frontend directory
        os.chdir(frontend_dir)
        
        # Start streamlit server
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501", "--server.address", "0.0.0.0"
        ])
        
        print("✅ Frontend server started on http://localhost:8501")
        return process
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    """Main startup function."""
    print("🤖 GenAI Document Assistant - Startup Script")
    print("=" * 50)
    
    # Store original directory
    original_dir = os.getcwd()
    
    try:
        # Check dependencies
        if not check_dependencies():
            return
        
        # Check spaCy model
        if not check_spacy_model():
            return
        
        # Check environment file
        env_ok = check_env_file()
        
        print("\n📋 Startup Summary:")
        print("- Backend: FastAPI server on port 8000")
        print("- Frontend: Streamlit app on port 8501")
        print("- API Docs: http://localhost:8000/docs")
        
        if not env_ok:
            print("\n⚠️  IMPORTANT: Please edit .env file with your API keys before starting")
            print("   Required keys: OPENAI_API_KEY or ANTHROPIC_API_KEY")
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                return
        
        # Start servers
        backend_process = start_backend()
        if not backend_process:
            return
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        frontend_process = start_frontend()
        if not frontend_process:
            backend_process.terminate()
            return
        
        print("\n🎉 Application started successfully!")
        print("📱 Frontend: http://localhost:8501")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        
        # Open browser
        try:
            webbrowser.open("http://localhost:8501")
        except:
            pass
        
        print("\n⏹️  Press Ctrl+C to stop all servers")
        
        try:
            # Wait for processes
            backend_process.wait()
            frontend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
            backend_process.terminate()
            frontend_process.terminate()
            print("✅ Servers stopped")
    
    except Exception as e:
        print(f"❌ Startup failed: {e}")
    finally:
        # Return to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main() 