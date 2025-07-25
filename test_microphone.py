#!/usr/bin/env python3
"""
Simple microphone test to diagnose audio input issues
"""

import pyaudio
import sys
import time

def test_microphone():
    """Test microphone setup and permissions."""
    
    print("🎤 Testing microphone setup...")
    
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # List all audio devices
        print("\n📋 Available audio devices:")
        device_count = p.get_device_count()
        
        input_devices = []
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append((i, device_info['name']))
                print(f"  {i}: {device_info['name']} (Input channels: {device_info['maxInputChannels']})")
        
        if not input_devices:
            print("❌ No input devices found!")
            return False
        
        # Test default input device
        try:
            default_input = p.get_default_input_device_info()
            print(f"\n🎯 Default input device: {default_input['name']}")
        except Exception as e:
            print(f"⚠️ Could not get default input device: {e}")
            return False
        
        # Test recording from microphone
        print("\n🔊 Testing audio recording...")
        try:
            # Open stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
            
            print("🎤 Recording for 3 seconds... (speak into your microphone)")
            
            # Record for 3 seconds
            audio_data = []
            for i in range(0, int(16000 / 1024 * 3)):
                data = stream.read(1024)
                audio_data.append(data)
            
            # Stop recording
            stream.stop_stream()
            stream.close()
            
            # Check if we got audio data
            if audio_data:
                print("✅ Audio recording successful!")
                print(f"📊 Recorded {len(audio_data)} audio chunks")
                
                # Check if there's actual audio (not just silence)
                import struct
                sample_sum = 0
                for chunk in audio_data:
                    samples = struct.unpack(f"{len(chunk)//2}h", chunk)
                    sample_sum += sum(abs(s) for s in samples)
                
                if sample_sum > 1000:  # Threshold for detecting actual audio
                    print("✅ Audio input detected (not just silence)")
                else:
                    print("⚠️ Only silence detected - check if microphone is working")
                    
                return True
            else:
                print("❌ No audio data recorded")
                return False
                
        except Exception as e:
            print(f"❌ Audio recording failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ PyAudio initialization failed: {e}")
        return False
    
    finally:
        try:
            p.terminate()
        except:
            pass

if __name__ == "__main__":
    test_microphone() 