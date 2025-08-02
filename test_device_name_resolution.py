#!/usr/bin/env python3
"""
Test script to verify device name resolution functionality.
"""

from async_tts_module import find_audio_device_by_name, list_audio_output_devices

def test_device_resolution():
    """Test device name resolution."""
    print("🧪 Testing device name resolution...")
    print()
    
    # List available devices first
    print("📋 Available devices:")
    list_audio_output_devices()
    print()
    
    # Test some common device name patterns
    test_names = [
        "Loopback Audio",
        "Speakers", 
        "loopback",  # Case insensitive test
        "SPEAKERS",  # Case insensitive test
        "NonExistentDevice",  # Should fail
        None,  # Should return None
        "",  # Should return None
    ]
    
    print("🎯 Testing device name resolution:")
    print("-" * 50)
    
    for name in test_names:
        if name is None:
            print(f"Testing: None")
        elif name == "":
            print(f"Testing: (empty string)")
        else:
            print(f"Testing: '{name}'")
        
        result = find_audio_device_by_name(name)
        
        if result is not None:
            print(f"  ✅ Found device index: {result}")
        else:
            print(f"  ❌ No device found")
        print()

if __name__ == "__main__":
    test_device_resolution()