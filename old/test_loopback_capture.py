#!/usr/bin/env python3
"""
Test capturing system audio output on macOS using BlackHole or similar virtual audio device.

Requirements:
1. Install BlackHole (free): https://github.com/ExistentialAudio/BlackHole
2. In Audio MIDI Setup, create a Multi-Output Device that includes both:
   - Your normal output (speakers/headphones)  
   - BlackHole 2ch
3. Set this Multi-Output Device as your system output
4. This script will capture from BlackHole
"""

import pyaudio
import numpy as np
import time

def find_blackhole_device():
    """Find the BlackHole virtual audio input device."""
    p = pyaudio.PyAudio()
    
    blackhole_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']} - Inputs: {info['maxInputChannels']}")
        
        if 'BlackHole' in info['name'] and info['maxInputChannels'] > 0:
            blackhole_index = i
            print(f"Found BlackHole device at index {i}")
            break
    
    p.terminate()
    return blackhole_index

def test_loopback_capture():
    """Test capturing system audio output."""
    device_index = find_blackhole_device()
    if device_index is None:
        print("BlackHole device not found. Please install BlackHole.")
        return
        
    p = pyaudio.PyAudio()
    
    # Open stream from BlackHole
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=256
    )
    
    print("Capturing system audio... Play some audio to test.")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            data = stream.read(256, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            
            # Simple volume meter
            volume = np.abs(audio_array).mean()
            bar = '=' * int(volume / 100)
            print(f"\rVolume: {bar:<50}", end='', flush=True)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    test_loopback_capture()
 