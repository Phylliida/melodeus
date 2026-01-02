#!/usr/bin/env python3
"""
Quick voice recognition test - record a sample and see how it matches.
"""

import numpy as np
import sounddevice as sd
import time
from titanet_voice_fingerprinting import TitaNetVoiceFingerprinter
from config_loader import load_config

def quick_test():
    """Quick test - record 3 seconds and see recognition results."""
    print("🔊 Quick Voice Recognition Test")
    print("=" * 40)
    
    # Load config and fingerprinter
    config = load_config('config.yaml')
    fingerprinter = TitaNetVoiceFingerprinter(config.speakers)
    
    print(f"📋 Loaded {len(fingerprinter.reference_fingerprints)} speaker profiles")
    for speaker_id, fingerprints in fingerprinter.reference_fingerprints.items():
        speaker_name = "Unknown"
        for profile_id, profile in config.speakers.profiles.items():
            if profile_id == speaker_id:
                speaker_name = profile.name
                break
        print(f"  - {speaker_name}: {len(fingerprints)} reference fingerprints")
    
    print(f"\n🎯 Confidence threshold: {config.speakers.recognition.confidence_threshold}")
    
    # Record audio
    duration = 3.0
    sample_rate = 16000
    
    print(f"\n🎤 Recording {duration} seconds...")
    print("⏱️  Starting in 3...")
    time.sleep(1)
    print("⏱️  2...")
    time.sleep(1) 
    print("⏱️  1...")
    time.sleep(1)
    print("🔴 SPEAK NOW!")
    
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    
    print("✅ Recording complete!")
    
    # Create fingerprint using TitaNet API
    audio_data = recording.flatten()
    
    # Create VoiceSegment for TitaNet
    from titanet_voice_fingerprinting import VoiceSegment
    voice_segment = VoiceSegment(
        audio_data=audio_data,
        speaker_id=0,
        start_time=0.0,
        end_time=duration,
        words=["test"],
        sample_rate=sample_rate
    )
    
    fingerprint = fingerprinter._create_titanet_fingerprint(voice_segment)
    
    if fingerprint is None:
        print("❌ Failed to create fingerprint from recording")
        return
    
    print(f"🔬 Created fingerprint with {len(fingerprint.embedding)} features")
    
    # Test against all speakers
    print(f"\n🧪 Testing against all speakers:")
    
    results = {}
    for speaker_id, ref_fingerprints in fingerprinter.reference_fingerprints.items():
        if not ref_fingerprints:
            continue
            
        # Calculate similarities using TitaNet's cosine similarity method
        similarities = []
        for ref_fp in ref_fingerprints:
            similarity = fingerprinter._cosine_similarity(fingerprint.embedding, ref_fp.embedding)
            similarities.append(similarity)
        
        best_similarity = max(similarities)
        avg_similarity = np.mean(similarities)
        
        results[speaker_id] = {
            'best': best_similarity,
            'avg': avg_similarity,
            'count': len(similarities)
        }
        
        # Get speaker name
        speaker_name = "Unknown"
        for profile_id, profile in config.speakers.profiles.items():
            if profile_id == speaker_id:
                speaker_name = profile.name
                break
        
        threshold = config.speakers.recognition.confidence_threshold
        status = "✅" if best_similarity >= threshold else "❌"
        print(f"  {status} {speaker_name:15} | Best: {best_similarity:.3f} | Avg: {avg_similarity:.3f} | Refs: {len(similarities)}")
    
    # Find best match
    if results:
        best_speaker = max(results.keys(), key=lambda x: results[x]['best'])
        best_score = results[best_speaker]['best']
        
        # Get best speaker name
        best_name = "Unknown"
        for profile_id, profile in config.speakers.profiles.items():
            if profile_id == best_speaker:
                best_name = profile.name
                break
        
        print(f"\n🏆 BEST MATCH: {best_name} (similarity: {best_score:.3f})")
        
        threshold = config.speakers.recognition.confidence_threshold
        if best_score >= threshold:
            print(f"✅ ACCEPTED - Above threshold ({threshold:.3f})")
        else:
            print(f"❌ REJECTED - Below threshold ({threshold:.3f})")
    
    print(f"\n💡 To improve accuracy:")
    print(f"   - Make sure you're speaking clearly")
    print(f"   - Check that reference audio quality is good")
    print(f"   - Consider adjusting the confidence threshold")
    print(f"   - Record longer reference samples if needed")

if __name__ == "__main__":
    quick_test()