#!/usr/bin/env python3
"""
Test TitaNet API compatibility without user interaction.
"""

import numpy as np
from titanet_voice_fingerprinting import TitaNetVoiceFingerprinter, VoiceSegment
from config_loader import load_config

def test_titanet_api():
    """Test TitaNet API methods."""
    print("🧪 Testing TitaNet API Compatibility")
    print("=" * 40)
    
    # Load config and fingerprinter
    config = load_config('config.yaml')
    fingerprinter = TitaNetVoiceFingerprinter(config.speakers)
    
    print(f"✅ TitaNet fingerprinter loaded")
    print(f"📋 Loaded {len(fingerprinter.reference_fingerprints)} speaker profiles")
    
    # Test with dummy audio
    duration = 3.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    
    # Generate speech-like audio
    t = np.linspace(0, duration, samples)
    fundamental = 150  # Hz
    audio = (np.sin(2 * np.pi * fundamental * t) * 0.3 +
            np.sin(2 * np.pi * fundamental * 2 * t) * 0.2)
    
    # Add modulation and noise
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
    audio = audio * envelope + np.random.normal(0, 0.05, samples)
    audio = audio.astype(np.float32)
    
    print(f"🎧 Generated {duration}s test audio")
    
    # Create VoiceSegment
    voice_segment = VoiceSegment(
        audio_data=audio,
        speaker_id=0,
        start_time=0.0,
        end_time=duration,
        words=["test", "audio"],
        sample_rate=sample_rate
    )
    
    print(f"📦 Created VoiceSegment")
    
    # Create fingerprint
    fingerprint = fingerprinter._create_titanet_fingerprint(voice_segment)
    
    if fingerprint is None:
        print("❌ Failed to create fingerprint")
        return False
    
    print(f"✅ Created TitaNet fingerprint")
    print(f"🔢 Embedding shape: {fingerprint.embedding.shape}")
    print(f"⏱️  Duration: {fingerprint.duration:.1f}s")
    
    # Test similarity calculation
    self_similarity = fingerprinter._cosine_similarity(fingerprint.embedding, fingerprint.embedding)
    print(f"🔄 Self-similarity: {self_similarity:.6f} (should be 1.0)")
    
    if abs(self_similarity - 1.0) < 0.001:
        print("✅ Cosine similarity working correctly")
    else:
        print("❌ Cosine similarity issue")
        return False
    
    # Test against reference fingerprints
    print(f"\n🧪 Testing against reference fingerprints:")
    
    for speaker_id, ref_fingerprints in fingerprinter.reference_fingerprints.items():
        if not ref_fingerprints:
            continue
            
        # Get speaker name
        speaker_name = speaker_id
        for profile_id, profile in config.speakers.profiles.items():
            if profile_id == speaker_id:
                speaker_name = profile.name
                break
        
        # Calculate similarities
        similarities = []
        for ref_fp in ref_fingerprints:
            similarity = fingerprinter._cosine_similarity(fingerprint.embedding, ref_fp.embedding)
            similarities.append(similarity)
        
        best_similarity = max(similarities)
        avg_similarity = np.mean(similarities)
        
        threshold = config.speakers.recognition.confidence_threshold
        status = "✅" if best_similarity >= threshold else "❌"
        
        print(f"  {status} {speaker_name:15} | Best: {best_similarity:.3f} | Avg: {avg_similarity:.3f}")
    
    print(f"\n✅ All TitaNet API tests passed!")
    return True

if __name__ == "__main__":
    success = test_titanet_api()
    if success:
        print("\n🎉 TitaNet integration is working correctly!")
        print("💡 You can now use quick_voice_test.py or your main voice system")
    else:
        print("\n❌ TitaNet integration has issues")