#!/usr/bin/env python3
"""
Test the discriminative power of TitaNet vs MFCC embeddings.
"""

import numpy as np
from config_loader import load_config

def test_cross_speaker_discrimination():
    """Test how well TitaNet distinguishes between speakers vs MFCC."""
    print("🔍 Testing Speaker Discrimination")
    print("=" * 50)
    
    config = load_config('config.yaml')
    
    print("🤖 Testing TitaNet discrimination...")
    try:
        from titanet_voice_fingerprinting import TitaNetVoiceFingerprinter
        titanet_fp = TitaNetVoiceFingerprinter(config.speakers)
        
        # Get fingerprints for each speaker
        titanet_results = {}
        for speaker_id, fingerprints in titanet_fp.reference_fingerprints.items():
            if fingerprints:
                # Get speaker name
                speaker_name = speaker_id
                for profile_id, profile in config.speakers.profiles.items():
                    if profile_id == speaker_id:
                        speaker_name = profile.name
                        break
                
                embeddings = [fp.embedding for fp in fingerprints]
                titanet_results[speaker_name] = embeddings
        
        print(f"📊 TitaNet: Loaded {len(titanet_results)} speakers")
        for speaker, embs in titanet_results.items():
            print(f"  - {speaker}: {len(embs)} embeddings")
        
        # Calculate cross-speaker similarities (TitaNet)
        titanet_cross_sims = []
        titanet_self_sims = []
        
        speakers = list(titanet_results.keys())
        for i, speaker1 in enumerate(speakers):
            for j, speaker2 in enumerate(speakers):
                embs1 = titanet_results[speaker1]
                embs2 = titanet_results[speaker2]
                
                # Calculate all pairwise similarities
                for emb1 in embs1:
                    for emb2 in embs2:
                        similarity = titanet_fp._cosine_similarity(emb1, emb2)
                        
                        if i == j:  # Same speaker
                            titanet_self_sims.append(similarity)
                        else:  # Different speakers
                            titanet_cross_sims.append(similarity)
        
        print(f"\n🎯 TitaNet Results:")
        print(f"  Same Speaker Similarity: {np.mean(titanet_self_sims):.3f} ± {np.std(titanet_self_sims):.3f}")
        print(f"  Cross Speaker Similarity: {np.mean(titanet_cross_sims):.3f} ± {np.std(titanet_cross_sims):.3f}")
        print(f"  Separation: {np.mean(titanet_self_sims) - np.mean(titanet_cross_sims):.3f}")
        
    except Exception as e:
        print(f"❌ TitaNet test failed: {e}")
        return
    
    print("\n🔧 Testing MFCC discrimination...")
    try:
        from voice_fingerprinting import VoiceFingerprinter
        mfcc_fp = VoiceFingerprinter(config.speakers)
        
        # Get MFCC fingerprints
        mfcc_results = {}
        for speaker_id, fingerprints in mfcc_fp.reference_fingerprints.items():
            if fingerprints:
                # Get speaker name
                speaker_name = speaker_id
                for profile_id, profile in config.speakers.profiles.items():
                    if profile_id == speaker_id:
                        speaker_name = profile.name
                        break
                
                embeddings = [fp.embedding for fp in fingerprints]
                mfcc_results[speaker_name] = embeddings
        
        print(f"📊 MFCC: Loaded {len(mfcc_results)} speakers")
        for speaker, embs in mfcc_results.items():
            print(f"  - {speaker}: {len(embs)} embeddings")
        
        # Calculate cross-speaker similarities (MFCC)
        mfcc_cross_sims = []
        mfcc_self_sims = []
        
        speakers = list(mfcc_results.keys())
        for i, speaker1 in enumerate(speakers):
            for j, speaker2 in enumerate(speakers):
                embs1 = mfcc_results[speaker1]
                embs2 = mfcc_results[speaker2]
                
                for emb1 in embs1:
                    for emb2 in embs2:
                        # Cosine similarity
                        norm1 = np.linalg.norm(emb1)
                        norm2 = np.linalg.norm(emb2)
                        if norm1 > 0 and norm2 > 0:
                            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                        else:
                            similarity = 0.0
                        
                        if i == j:  # Same speaker
                            mfcc_self_sims.append(similarity)
                        else:  # Different speakers
                            mfcc_cross_sims.append(similarity)
        
        print(f"\n🎯 MFCC Results:")
        print(f"  Same Speaker Similarity: {np.mean(mfcc_self_sims):.3f} ± {np.std(mfcc_self_sims):.3f}")
        print(f"  Cross Speaker Similarity: {np.mean(mfcc_cross_sims):.3f} ± {np.std(mfcc_cross_sims):.3f}")
        print(f"  Separation: {np.mean(mfcc_self_sims) - np.mean(mfcc_cross_sims):.3f}")
        
    except Exception as e:
        print(f"❌ MFCC test failed: {e}")
        return
    
    # Compare results
    print(f"\n📊 COMPARISON:")
    print(f"{'Method':<10} {'Same Speaker':<15} {'Cross Speaker':<15} {'Separation':<12} {'Winner'}")
    print("-" * 70)
    
    titanet_sep = np.mean(titanet_self_sims) - np.mean(titanet_cross_sims)
    mfcc_sep = np.mean(mfcc_self_sims) - np.mean(mfcc_cross_sims)
    
    print(f"{'TitaNet':<10} {np.mean(titanet_self_sims):<15.3f} {np.mean(titanet_cross_sims):<15.3f} {titanet_sep:<12.3f} {'🏆' if titanet_sep > mfcc_sep else ''}")
    print(f"{'MFCC':<10} {np.mean(mfcc_self_sims):<15.3f} {np.mean(mfcc_cross_sims):<15.3f} {mfcc_sep:<12.3f} {'🏆' if mfcc_sep > titanet_sep else ''}")
    
    print(f"\n💡 Analysis:")
    if titanet_sep > mfcc_sep:
        improvement = (titanet_sep / mfcc_sep - 1) * 100
        print(f"✅ TitaNet shows {improvement:.1f}% better speaker separation than MFCC")
        print(f"🎯 TitaNet should provide much better speaker recognition accuracy")
    else:
        print(f"❌ MFCC performed better (unexpected - check implementation)")
    
    # Check if separation is good enough
    if titanet_sep > 0.3:
        print(f"✅ TitaNet separation ({titanet_sep:.3f}) is excellent for speaker recognition")
    elif titanet_sep > 0.1:
        print(f"⚠️  TitaNet separation ({titanet_sep:.3f}) is moderate - may need threshold tuning")
    else:
        print(f"❌ TitaNet separation ({titanet_sep:.3f}) is poor - check reference audio quality")

if __name__ == "__main__":
    test_cross_speaker_discrimination()