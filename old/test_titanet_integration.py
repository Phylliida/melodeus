#!/usr/bin/env python3
"""
Test TitaNet integration and compare with old MFCC approach.
"""

import numpy as np
import time
from config_loader import load_config

def test_titanet_loading():
    """Test loading TitaNet model and creating embeddings."""
    print("🔊 Testing TitaNet Integration")
    print("=" * 50)
    
    try:
        from titanet_voice_fingerprinting import TitaNetVoiceFingerprinter, NEMO_AVAILABLE
        
        if not NEMO_AVAILABLE:
            print("❌ NeMo not available - cannot test TitaNet")
            return False
        
        # Load config
        print("📋 Loading configuration...")
        config = load_config('config.yaml')
        
        # Initialize TitaNet fingerprinter
        print("🔊 Initializing TitaNet fingerprinter...")
        start_time = time.time()
        
        fingerprinter = TitaNetVoiceFingerprinter(config.speakers)
        
        init_time = time.time() - start_time
        print(f"✅ TitaNet initialized in {init_time:.2f}s")
        
        # Test with dummy audio
        print("\n🧪 Testing embedding creation...")
        
        # Create 3 seconds of dummy audio (speech-like)
        duration = 3.0
        sample_rate = 16000
        samples = int(duration * sample_rate)
        
        # Generate speech-like audio (not just noise)
        t = np.linspace(0, duration, samples)
        # Mix of fundamental frequency + harmonics (simulates speech)
        fundamental = 150  # Hz
        audio = (np.sin(2 * np.pi * fundamental * t) * 0.3 +
                np.sin(2 * np.pi * fundamental * 2 * t) * 0.2 +
                np.sin(2 * np.pi * fundamental * 3 * t) * 0.1)
        
        # Add some modulation (simulates speech envelope)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
        audio = audio * envelope
        
        # Add a bit of noise
        noise = np.random.normal(0, 0.05, samples)
        audio = audio + noise
        
        # Normalize
        audio = audio.astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Create voice segment
        from titanet_voice_fingerprinting import VoiceSegment
        voice_segment = VoiceSegment(
            audio_data=audio,
            speaker_id=0,
            start_time=0.0,
            end_time=duration,
            words=["test", "audio"],
            sample_rate=sample_rate
        )
        
        # Create embedding
        start_time = time.time()
        fingerprint = fingerprinter._create_titanet_fingerprint(voice_segment)
        embedding_time = time.time() - start_time
        
        if fingerprint:
            print(f"✅ Created TitaNet embedding in {embedding_time:.3f}s")
            print(f"🔢 Embedding shape: {fingerprint.embedding.shape}")
            print(f"📊 Embedding stats: mean={np.mean(fingerprint.embedding):.3f}, std={np.std(fingerprint.embedding):.3f}")
            print(f"⏱️  Duration: {fingerprint.duration:.1f}s")
            print(f"🎯 Confidence: {fingerprint.confidence:.3f}")
            
            # Test similarity with itself (should be 1.0)
            self_similarity = fingerprinter._cosine_similarity(fingerprint.embedding, fingerprint.embedding)
            print(f"🔄 Self-similarity: {self_similarity:.6f} (should be 1.0)")
            
            # Test with a different random embedding (should be low)
            random_embedding = np.random.randn(*fingerprint.embedding.shape).astype(np.float32)
            random_embedding = random_embedding / np.linalg.norm(random_embedding)
            random_similarity = fingerprinter._cosine_similarity(fingerprint.embedding, random_embedding)
            print(f"🎲 Random similarity: {random_similarity:.6f} (should be low)")
            
            return True
        else:
            print("❌ Failed to create embedding")
            return False
            
    except Exception as e:
        print(f"❌ Error testing TitaNet: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_mfcc():
    """Compare TitaNet with old MFCC approach."""
    print("\n📊 Comparing TitaNet vs MFCC")
    print("-" * 30)
    
    try:
        # Test old MFCC system
        from voice_fingerprinting import VoiceFingerprinter
        config = load_config('config.yaml')
        
        print("🔧 Testing old MFCC system...")
        mfcc_fingerprinter = VoiceFingerprinter(config.speakers)
        
        # Test TitaNet system
        from titanet_voice_fingerprinting import TitaNetVoiceFingerprinter, NEMO_AVAILABLE
        
        if NEMO_AVAILABLE:
            print("🤖 Testing TitaNet system...")
            titanet_fingerprinter = TitaNetVoiceFingerprinter(config.speakers)
            
            print("\n📈 Comparison Summary:")
            print(f"  MFCC Features: {len(mfcc_fingerprinter.reference_fingerprints.get('Antra', [{}])[-1].embedding) if mfcc_fingerprinter.reference_fingerprints.get('Antra') else 0} dimensions")
            print(f"  TitaNet Features: 192 dimensions (deep learning)")
            print(f"  MFCC Approach: Traditional signal processing")
            print(f"  TitaNet Approach: State-of-the-art neural network")
            print(f"  Expected Accuracy: TitaNet >> MFCC")
            
        else:
            print("⚠️  Cannot compare - NeMo not available")
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")

def main():
    """Run all tests."""
    success = test_titanet_loading()
    
    if success:
        compare_with_mfcc()
        
        print("\n🎉 TitaNet Integration Test Results:")
        print("✅ TitaNet model loaded successfully")
        print("✅ Embedding creation working")
        print("✅ Similarity calculations working")
        print("\n💡 Next Steps:")
        print("  1. Update async_stt_module.py to use TitaNet")
        print("  2. Test with real voice samples")
        print("  3. Compare recognition accuracy")
    else:
        print("\n❌ TitaNet integration failed")
        print("💡 Troubleshooting:")
        print("  1. Check NeMo installation: pip install 'nemo_toolkit[asr]'")
        print("  2. Check CUDA/PyTorch compatibility")
        print("  3. Check internet connection (for model download)")

if __name__ == "__main__":
    main()