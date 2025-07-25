#!/usr/bin/env python3
"""
Quick test script to validate all APIs work together for voice-to-LLM system.
Tests Deepgram STT, OpenAI LLM, and ElevenLabs TTS integration.
"""

import asyncio
import pyaudio
import time
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from openai import OpenAI
import requests
import json

async def test_deepgram_stt(api_key: str) -> bool:
    """Test Deepgram STT connection."""
    print("🔍 Testing Deepgram STT...")
    try:
        client = DeepgramClient(api_key)
        
        # Test with a simple connection
        connection = client.listen.websocket.v("1")
        options = LiveOptions(
            model="nova-3",
            language="en-US",
            smart_format=True
        )
        
        success = False
        
        def on_open(*args, **kwargs):
            nonlocal success
            success = True
            print("✅ Deepgram connection successful")
        
        def on_error(*args, **kwargs):
            error = kwargs.get('error', 'Unknown error')
            print(f"❌ Deepgram error: {error}")
        
        connection.on(LiveTranscriptionEvents.Open, on_open)
        connection.on(LiveTranscriptionEvents.Error, on_error)
        
        if connection.start(options):
            await asyncio.sleep(1)  # Wait for connection
            connection.finish()
            return success
        
        return False
        
    except Exception as e:
        print(f"❌ Deepgram test failed: {e}")
        return False

def test_openai_llm(api_key: str) -> bool:
    """Test OpenAI LLM connection."""
    print("🔍 Testing OpenAI LLM...")
    try:
        client = OpenAI(api_key=api_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API test successful' if you can hear this."}],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"✅ OpenAI response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

def test_elevenlabs_tts(api_key: str, voice_id: str = "pNInz6obpgDQGcFmaJgB") -> bool:
    """Test ElevenLabs TTS connection."""
    print("🔍 Testing ElevenLabs TTS...")
    try:
        # Test with simple HTTP API first
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        data = {
            "text": "API test successful",
            "voice_settings": {
                "speed": 1.0,
                "stability": 0.5,
                "similarity_boost": 0.8
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            print("✅ ElevenLabs TTS connection successful")
            return True
        else:
            print(f"❌ ElevenLabs error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ElevenLabs test failed: {e}")
        return False

def test_audio_devices() -> bool:
    """Test audio input/output devices."""
    print("🔍 Testing audio devices...")
    try:
        p = pyaudio.PyAudio()
        
        # Check for input devices
        input_devices = []
        output_devices = []
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append(info['name'])
            if info['maxOutputChannels'] > 0:
                output_devices.append(info['name'])
        
        p.terminate()
        
        if input_devices and output_devices:
            print(f"✅ Found {len(input_devices)} input and {len(output_devices)} output devices")
            return True
        else:
            print("❌ No suitable audio devices found")
            return False
            
    except Exception as e:
        print(f"❌ Audio device test failed: {e}")
        return False

async def run_integration_test():
    """Run full integration test."""
    print("🚀 Voice-to-LLM Integration Test")
    print("=" * 40)
    
    # Get API keys
    deepgram_key = input("Enter Deepgram API key: ").strip()
    elevenlabs_key = input("Enter ElevenLabs API key: ").strip()
    openai_key = input("Enter OpenAI API key: ").strip()
    
    if not all([deepgram_key, elevenlabs_key, openai_key]):
        print("❌ All API keys are required!")
        return
    
    voice_id = input("Enter ElevenLabs voice ID (or press Enter for default): ").strip()
    if not voice_id:
        voice_id = "pNInz6obpgDQGcFmaJgB"
    
    print("\n" + "=" * 40)
    print("Running tests...")
    print("=" * 40)
    
    # Run tests
    results = []
    
    # Test audio devices
    results.append(("Audio Devices", test_audio_devices()))
    
    # Test Deepgram STT
    results.append(("Deepgram STT", await test_deepgram_stt(deepgram_key)))
    
    # Test OpenAI LLM
    results.append(("OpenAI LLM", test_openai_llm(openai_key)))
    
    # Test ElevenLabs TTS
    results.append(("ElevenLabs TTS", test_elevenlabs_tts(elevenlabs_key, voice_id)))
    
    # Print results
    print("\n" + "=" * 40)
    print("Test Results:")
    print("=" * 40)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! You're ready to use the voice-to-LLM system.")
        print("\nNext steps:")
        print("1. Run 'python test_elevenlabs_streaming.py' to test TTS streaming")
        print("2. Run 'python voice_llm_conversation.py' for full voice conversation")
    else:
        print("⚠️  Some tests failed. Please check your API keys and configuration.")
        print("\nTroubleshooting:")
        print("- Verify all API keys are correct and have sufficient credits")
        print("- Check your internet connection")
        print("- Ensure audio devices are properly configured")
        print("- Review the VOICE_LLM_GUIDE.md for detailed setup instructions")

if __name__ == "__main__":
    asyncio.run(run_integration_test()) 