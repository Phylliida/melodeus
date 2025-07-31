#!/usr/bin/env python3
"""
Reference Audio Creation Tool
Creates reference audio files for voice fingerprinting.

Usage:
    python create_reference_audio.py --name Alice --output speaker_profiles/alice_reference.wav
    
This will record 30+ seconds of audio to use as a reference for speaker identification.
"""

import argparse
import pyaudio
import wave
import time
import os
from pathlib import Path

def record_reference_audio(name: str, output_path: str, duration: int = 35):
    """Record reference audio for a speaker."""
    
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    p = pyaudio.PyAudio()
    
    print(f"🎤 Recording reference audio for {name}")
    print(f"📁 Output: {output_path}")
    print(f"⏱️  Duration: {duration} seconds")
    print()
    print("📋 INSTRUCTIONS:")
    print("   • Speak clearly and naturally")
    print("   • Say a variety of phrases (numbers, names, common words)")
    print("   • Keep consistent distance from microphone")
    print("   • Avoid background noise")
    print()
    print("💡 EXAMPLE SCRIPT (feel free to improvise):")
    print("   'Hello, my name is [your name]. I am recording a voice sample for")
    print("   the AI voice system. Today is [current date]. Let me count from")
    print("   one to twenty. One, two, three... [continue counting]. I also like")
    print("   to talk about technology, artificial intelligence, and voice")
    print("   recognition systems. This should be enough audio for the system")
    print("   to learn my voice patterns.'")
    print()
    
    input("Press Enter when ready to start recording...")
    
    try:
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        print(f"🔴 RECORDING... ({duration} seconds)")
        
        frames = []
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Show progress
            elapsed = (i * CHUNK) / RATE
            remaining = duration - elapsed
            if int(elapsed) % 5 == 0 and elapsed > 0:
                print(f"⏱️  {remaining:.0f} seconds remaining...")
        
        print("✅ Recording complete!")
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save the audio file
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"💾 Audio saved to: {output_path}")
        print(f"🎯 Ready to use for voice fingerprinting!")
        
        # Get file size for verification
        file_size = os.path.getsize(output_path)
        print(f"📊 File size: {file_size / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ Error during recording: {e}")
        p.terminate()

def main():
    parser = argparse.ArgumentParser(description="Create reference audio for voice fingerprinting")
    parser.add_argument("--name", required=True, help="Name of the speaker")
    parser.add_argument("--output", required=True, help="Output WAV file path")
    parser.add_argument("--duration", type=int, default=35, help="Recording duration in seconds (default: 35)")
    
    args = parser.parse_args()
    
    # Validate output path
    if not args.output.endswith('.wav'):
        print("❌ Output file must have .wav extension")
        return
    
    # Check if file already exists
    if os.path.exists(args.output):
        overwrite = input(f"⚠️  File {args.output} already exists. Overwrite? (y/N): ")
        if overwrite.lower() != 'y':
            print("Cancelled.")
            return
    
    record_reference_audio(args.name, args.output, args.duration)

if __name__ == "__main__":
    main()