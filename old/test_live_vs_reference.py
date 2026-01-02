#!/usr/bin/env python3
"""
Test to compare live microphone audio vs reference audio characteristics.
"""

import numpy as np
import sounddevice as sd
from titanet_voice_fingerprinting import TitaNetVoiceFingerprinter, VoiceSegment
from config_loader import load_config
import time

def test_live_vs_reference():
    """Test similarities between live recording and reference audio."""
    print("🧪 Live vs Reference Audio Test")
    print("=" * 40)
    
    # Load config and fingerprinter
    config = load_config('config.yaml')
    fingerprinter = TitaNetVoiceFingerprinter(config.speakers)
    
    print(f"✅ TitaNet fingerprinter loaded")
    print(f"📋 Confidence threshold: {config.speakers.recognition.confidence_threshold}")
    
    # Record multiple samples
    duration = 4.0  # Same as reference chunks
    sample_rate = 16000
    
    print(f"\n🎤 Recording {duration}s samples (same length as reference chunks)")
    
    for i in range(3):
        print(f"\n📝 Sample {i+1}/3")
        print("⏱️  Starting in 3...")
        time.sleep(1)
        print("⏱️  2...")
        time.sleep(1)
        print("⏱️  1...")
        time.sleep(1)
        print("🔴 SPEAK NOW! (4 seconds)")
        
        # Record audio
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()
        
        print("✅ Recording complete!")
        
        # Create VoiceSegment
        audio_data = recording.flatten()
        voice_segment = VoiceSegment(
            audio_data=audio_data,
            speaker_id=0,
            start_time=0.0,
            end_time=duration,
            words=["test", "sample"],
            sample_rate=sample_rate
        )
        
        # Create fingerprint
        fingerprint = fingerprinter._create_titanet_fingerprint(voice_segment)
        
        if fingerprint is None:
            print("❌ Failed to create fingerprint")
            continue
        
        print(f"✅ Created fingerprint (confidence: {fingerprint.confidence:.3f})")
        
        # Test against each reference profile
        print(f"🔍 Testing against reference fingerprints:")
        
        for speaker_id, ref_fingerprints in fingerprinter.reference_fingerprints.items():
            if not ref_fingerprints:
                continue
            
            # Get speaker name
            speaker_name = speaker_id
            for profile_id, profile in config.speakers.profiles.items():
                if profile_id == speaker_id:
                    speaker_name = profile.name
                    break
            
            # Calculate similarities against all reference fingerprints
            similarities = []
            for ref_fp in ref_fingerprints:
                similarity = fingerprinter._cosine_similarity(fingerprint.embedding, ref_fp.embedding)
                similarities.append(similarity)
            
            best_similarity = max(similarities)
            avg_similarity = np.mean(similarities)
            min_similarity = min(similarities)
            
            threshold = config.speakers.recognition.confidence_threshold
            status = "✅" if best_similarity >= threshold else "❌"
            
            print(f"  {status} {speaker_name:15} | Best: {best_similarity:.3f} | Avg: {avg_similarity:.3f} | Min: {min_similarity:.3f}")
        
        print(f"  ⚡ Audio RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")
        print(f"  📊 Audio range: [{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
        
        time.sleep(1)  # Brief pause between samples

if __name__ == "__main__":
    test_live_vs_reference()