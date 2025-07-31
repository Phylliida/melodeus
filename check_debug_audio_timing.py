#!/usr/bin/env python3
"""
Check the actual timing of debug audio segments vs Deepgram's reported timing.
"""

import soundfile as sf
import os
from pathlib import Path

def analyze_debug_audio():
    """Analyze the timing of debug audio segments."""
    debug_dir = Path("debug_audio_segments")
    
    if not debug_dir.exists():
        print("❌ No debug audio directory found")
        return
    
    print("🔍 Debug Audio Timing Analysis")
    print("=" * 50)
    
    audio_files = sorted(debug_dir.glob("*.wav"))
    
    for audio_file in audio_files:
        try:
            # Load audio file
            audio_data, sample_rate = sf.read(audio_file)
            
            # Calculate actual duration
            actual_duration = len(audio_data) / sample_rate
            
            # Extract info from filename
            filename = audio_file.name
            # Format: segment_001_speaker_0_23468samples.wav
            parts = filename.replace('.wav', '').split('_')
            if len(parts) >= 4:
                segment_num = parts[1]
                speaker_id = parts[3]
                expected_samples = int(parts[4].replace('samples', ''))
                
                print(f"📁 {filename}")
                print(f"   Expected samples: {expected_samples}")
                print(f"   Actual samples: {len(audio_data)}")
                print(f"   Sample rate: {sample_rate}Hz")
                print(f"   Actual duration: {actual_duration:.3f}s")
                print(f"   Expected duration: {expected_samples/16000:.3f}s")
                
                if abs(actual_duration - expected_samples/16000) > 0.01:
                    print(f"   ⚠️  Duration mismatch!")
                else:
                    print(f"   ✅ Duration matches")
                print()
            
        except Exception as e:
            print(f"❌ Error analyzing {audio_file}: {e}")

if __name__ == "__main__":
    analyze_debug_audio()