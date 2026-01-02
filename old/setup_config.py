#!/usr/bin/env python3
"""
Setup script for Voice AI System Configuration
Helps users create their config.yaml file with API keys.
"""

import os
import shutil
from pathlib import Path

def setup_configuration():
    """Interactive setup for configuration file."""
    print("🎙️ Voice AI System Configuration Setup")
    print("=====================================")
    print()
    
    # Check if config already exists
    config_file = Path("config.yaml")
    if config_file.exists():
        print("⚠️  config.yaml already exists!")
        overwrite = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("✅ Setup cancelled - keeping existing config")
            return
    
    # Copy example file
    example_file = Path("config.yaml.example")
    if not example_file.exists():
        print("❌ config.yaml.example not found!")
        print("💡 Creating a basic example file...")
        create_example_config()
    
    print("📋 Creating config.yaml from example...")
    shutil.copy("config.yaml.example", "config.yaml")
    
    # Interactive key setup
    print("\n🔑 API Key Setup")
    print("================")
    print("Enter your API keys (or press Enter to skip):")
    print()
    
    # Get API keys
    deepgram_key = input("🎤 Deepgram API key: ").strip()
    elevenlabs_key = input("🔊 ElevenLabs API key: ").strip()
    openai_key = input("🤖 OpenAI API key: ").strip()
    
    # Optional voice configuration
    print("\n🎵 Voice Configuration")
    print("======================")
    print("Popular voice options:")
    print("  1. T2KZm9rWPG5TgXTyjt7E (Catalyst - default)")
    print("  2. pNInz6obpgDQGcFmaJgB (Rachel)")  
    print("  3. yjJ45q8TVCrtMhEKurxY (Von Fusion)")
    print("  4. EXAVITQu4vr4xnSDxMaL (Sarah)")
    print()
    
    voice_choice = input("Choose voice (1-4) or enter custom ID: ").strip()
    
    voice_ids = {
        "1": "T2KZm9rWPG5TgXTyjt7E",
        "2": "pNInz6obpgDQGcFmaJgB", 
        "3": "yjJ45q8TVCrtMhEKurxY",
        "4": "EXAVITQu4vr4xnSDxMaL"
    }
    
    if voice_choice in voice_ids:
        voice_id = voice_ids[voice_choice]
    elif voice_choice:
        voice_id = voice_choice
    else:
        voice_id = "T2KZm9rWPG5TgXTyjt7E"  # default
    
    # Update config file with provided keys
    if any([deepgram_key, elevenlabs_key, openai_key]):
        print("\n📝 Updating config.yaml with your API keys...")
        update_config_file(deepgram_key, elevenlabs_key, openai_key, voice_id)
    
    print("\n✅ Configuration setup complete!")
    print("\n🚀 Next steps:")
    print("1. Review and edit config.yaml if needed")
    print("2. Run: python unified_voice_conversation_config.py")
    print("3. Or test with: python config_loader.py")
    print()
    print("💡 Tip: config.yaml is in .gitignore so your API keys stay private")

def create_example_config():
    """Create a basic example config file."""
    from config_loader import ConfigLoader
    ConfigLoader.create_example_config("config.yaml.example")

def update_config_file(deepgram_key, elevenlabs_key, openai_key, voice_id):
    """Update the config file with provided API keys."""
    try:
        import yaml
        
        # Read current config
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Update API keys if provided
        if not config.get('api_keys'):
            config['api_keys'] = {}
            
        if deepgram_key:
            config['api_keys']['deepgram'] = deepgram_key
        if elevenlabs_key:
            config['api_keys']['elevenlabs'] = elevenlabs_key
        if openai_key:
            config['api_keys']['openai'] = openai_key
            
        # Update voice ID
        if not config.get('voice'):
            config['voice'] = {}
        config['voice']['id'] = voice_id
        
        # Write back to file
        with open("config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        print("✅ API keys updated in config.yaml")
        
    except Exception as e:
        print(f"⚠️  Error updating config file: {e}")
        print("💡 You can manually edit config.yaml to add your API keys")

def check_requirements():
    """Check if required packages are installed."""
    try:
        import yaml
        import asyncio
        import pyaudio
        from deepgram import DeepgramClient
        from openai import OpenAI
        import websockets
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def main():
    """Main setup function."""
    print("🔍 Checking requirements...")
    if not check_requirements():
        return
    
    print("✅ All requirements satisfied")
    print()
    
    setup_configuration()

if __name__ == "__main__":
    main() 