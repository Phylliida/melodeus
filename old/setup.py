#!/usr/bin/env python3
"""
Setup script for Real-time STT with Live Diarization
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ is required. You have Python {}.{}.{}".format(
            version.major, version.minor, version.micro))
        return False
    print("✅ Python {}.{}.{} is compatible".format(
        version.major, version.minor, version.micro))
    return True

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_pyaudio():
    """Check if PyAudio is working and provide platform-specific help."""
    try:
        import pyaudio
        print("✅ PyAudio is installed and working")
        return True
    except ImportError:
        print("❌ PyAudio is not installed or not working")
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            print("💡 Try: brew install portaudio && pip install pyaudio")
        elif system == "linux":
            print("💡 Try: sudo apt-get install libasound-dev portaudio19-dev && pip install pyaudio")
        elif system == "windows":
            print("💡 Try: pip install pyaudio")
        
        return False

def check_api_key():
    """Check if Deepgram API key is configured."""
    api_key = os.getenv('DEEPGRAM_API_KEY')
    if api_key:
        print("✅ Deepgram API key is configured")
        return True
    else:
        print("❌ Deepgram API key is not configured")
        print("💡 Set it with: export DEEPGRAM_API_KEY='your_api_key_here'")
        print("💡 Or create a .env file with: DEEPGRAM_API_KEY=your_api_key_here")
        print("💡 Get your API key from: https://console.deepgram.com/")
        return False

def test_microphone():
    """Test microphone access."""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        # Check if there are any audio input devices
        input_devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append(device_info['name'])
        
        p.terminate()
        
        if input_devices:
            print(f"✅ Found {len(input_devices)} input device(s)")
            print("   Available microphones:")
            for i, device in enumerate(input_devices[:3]):  # Show first 3
                print(f"   • {device}")
            return True
        else:
            print("❌ No microphone devices found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing microphone: {e}")
        return False

def main():
    """Main setup function."""
    print("🎤 Real-time STT with Live Diarization - Setup")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", install_dependencies),
        ("PyAudio", check_pyaudio),
        ("API Key", check_api_key),
        ("Microphone", test_microphone),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Setup complete! You can now run: python realtime_stt_diarization.py")
    else:
        print("❌ Setup incomplete. Please resolve the issues above.")
        print("💡 Check the README.md for detailed troubleshooting.")

if __name__ == "__main__":
    main() 