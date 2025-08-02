#!/usr/bin/env python3
"""
List all available audio devices on the system.
This helps configure the output_device_name in config.yaml
"""

import pyaudio
import sys

def list_all_audio_devices():
    """List all audio devices (both input and output)."""
    p = pyaudio.PyAudio()
    
    print("\n" + "="*60)
    print("🎤 AUDIO DEVICES")
    print("="*60)
    
    # Get default devices
    try:
        default_input = p.get_default_input_device_info()['index']
        default_output = p.get_default_output_device_info()['index']
    except:
        default_input = None
        default_output = None
    
    # List all devices
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            
            # Determine device type
            device_types = []
            if info['maxInputChannels'] > 0:
                device_types.append("INPUT")
            if info['maxOutputChannels'] > 0:
                device_types.append("OUTPUT")
            
            # Check if default
            default_markers = []
            if i == default_input:
                default_markers.append("DEFAULT INPUT")
            if i == default_output:
                default_markers.append("DEFAULT OUTPUT")
            
            # Print device info
            print(f"\n📍 Index {i}: {info['name']}")
            print(f"   Type: {', '.join(device_types)}")
            if default_markers:
                print(f"   Status: {', '.join(default_markers)} ⭐")
            print(f"   Host API: {p.get_host_api_info_by_index(info['hostApi'])['name']}")
            print(f"   Sample Rate: {int(info['defaultSampleRate'])} Hz")
            
            if info['maxInputChannels'] > 0:
                print(f"   Input Channels: {info['maxInputChannels']}")
            if info['maxOutputChannels'] > 0:
                print(f"   Output Channels: {info['maxOutputChannels']}")
                
        except Exception as e:
            print(f"\n❌ Error reading device {i}: {e}")
    
    p.terminate()
    
    print("\n" + "="*60)
    print("💡 HOW TO USE:")
    print("="*60)
    print("1. Find your desired INPUT and OUTPUT devices from the list above")
    print("2. Note part of their names (device names are matched using partial matching)")
    print("3. In config.yaml, configure devices:")
    print("\n📥 For INPUT devices (microphones) - under 'stt:':")
    print("   input_device_name: \"<partial_device_name>\"")
    print("\n📤 For OUTPUT devices (speakers) - under 'tts:':")
    print("   output_device_name: \"<partial_device_name>\"")
    print("\n📋 Example config.yaml:")
    print("   stt:")
    print("     input_device_name: \"Scarlett 18i20\"    # Matches audio interface mic")
    print("     # or")
    print("     input_device_name: \"MacBook Pro\"       # Matches built-in mic")
    print("   tts:")
    print("     output_device_name: \"Loopback Audio\"   # Matches virtual audio device")
    print("     # or")
    print("     output_device_name: \"Speakers\"         # Matches any device with 'Speakers'")
    print("\n💡 Use 'null' for default devices")
    print("\n")

if __name__ == "__main__":
    try:
        list_all_audio_devices()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)