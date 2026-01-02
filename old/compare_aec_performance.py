#!/usr/bin/env python3
"""
Compare performance between old and new AEC implementations.
"""

import numpy as np
import time
import random
from advanced_echo_cancellation import AdaptiveEchoCancellationProcessor
from improved_aec_processor import ImprovedEchoCancellationProcessor

def simulate_realistic_scenario(processor, name="Processor"):
    """Simulate a realistic voice conversation scenario."""
    print(f"\n🎯 Testing {name}")
    print("=" * 60)
    
    # Track start time
    start_time = time.time()
    
    # Phase 1: Thinking sound (30 frames)
    print("🎵 Phase 1: Thinking sound")
    for i in range(30):
        ref_audio = np.random.randint(-1000, 1000, 256, dtype=np.int16).tobytes()
        processor.add_reference_audio(ref_audio)
        
        if i > 5:
            mic_audio = np.random.randint(-100, 100, 256, dtype=np.int16).tobytes()
            processor.process(mic_audio)
    
    # Phase 2: TTS speech with realistic bursts
    print("🗣️ Phase 2: TTS speech (bursty)")
    tts_chunks = [
        1723,  # "nods slowly, wincing at the movement"
        830,   # "yeah."
        1532,  # "Yeah, that's that's probably a good idea."
        511,   # short pause
        1723,  # another long utterance
    ]
    
    for chunk_ms in tts_chunks:
        # Send TTS burst
        samples = int(chunk_ms * 16)
        ref_audio = np.random.randint(-5000, 5000, samples, dtype=np.int16).tobytes()
        processor.add_reference_audio(ref_audio)
        
        # Microphone processes during TTS playback
        frames_during_tts = int(chunk_ms / 16)
        for i in range(frames_during_tts):
            mic_audio = np.random.randint(-200, 200, 256, dtype=np.int16).tobytes()
            processor.process(mic_audio)
            time.sleep(0.001)
        
        # Small gap between TTS chunks
        for i in range(5):
            mic_audio = np.random.randint(-100, 100, 256, dtype=np.int16).tobytes()
            processor.process(mic_audio)
    
    # Phase 3: Silence
    print("🔇 Phase 3: Silence")
    for i in range(30):
        mic_audio = np.random.randint(-50, 50, 256, dtype=np.int16).tobytes()
        processor.process(mic_audio)
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    # Get final statistics
    stats = processor.get_stats()
    #processor.print_stats()
    
    return stats, elapsed

def main():
    """Compare the two AEC implementations."""
    print("🔬 AEC Performance Comparison")
    print("=" * 60)
    
    # Test parameters
    frame_size = 256
    filter_length = 2048
    sample_rate = 16000
    
    # Test old implementation
    print("\n📦 1. ORIGINAL AdaptiveEchoCancellationProcessor")
    old_processor = AdaptiveEchoCancellationProcessor(
        frame_size=frame_size,
        filter_length=filter_length,
        sample_rate=sample_rate,
        initial_delay_ms=100,  # Original default
        debug_level=1
    )
    old_stats, old_time = simulate_realistic_scenario(old_processor, "Original AEC")
    
    print("\n" + "=" * 60)
    
    # Test new implementation
    print("\n📦 2. IMPROVED EchoCancellationProcessor")
    new_processor = ImprovedEchoCancellationProcessor(
        frame_size=frame_size,
        filter_length=filter_length,
        sample_rate=sample_rate,
        initial_delay_ms=200,  # Improved default
        debug_level=1
    )
    new_stats, new_time = simulate_realistic_scenario(new_processor, "Improved AEC")
    
    # Compare results
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Underrun Rate:")
    print(f"   Original: {old_stats['underrun_rate']*100:6.1f}%")
    print(f"   Improved: {new_stats['underrun_rate']*100:6.1f}%")
    improvement = (old_stats['underrun_rate'] - new_stats['underrun_rate']) / old_stats['underrun_rate'] * 100
    print(f"   Improvement: {improvement:.1f}%")
    
    print(f"\n📈 Buffer Performance:")
    print(f"   Original underruns: {old_stats['buffer_underruns']}")
    print(f"   Improved underruns: {new_stats['buffer_underruns']}")
    print(f"   Original max fill: {old_stats.get('max_fill_level', 'N/A')} frames")
    print(f"   Improved max fill: {new_stats.get('max_fill_level', 'N/A')} frames")
    
    print(f"\n⏱️ Delay Management:")
    print(f"   Original delay: {old_stats['current_delay_ms']:.0f}ms ({old_stats['current_delay_ms']/1000*old_stats['frames_processed']/new_time:.1f} frames avg)")
    print(f"   Improved delay: {new_stats['current_delay_ms']:.0f}ms ({new_stats['current_delay_ms']/1000*new_stats['frames_processed']/new_time:.1f} frames avg)")
    print(f"   Original adjustments: {old_stats['delay_adjustments']}")
    print(f"   Improved adjustments: {new_stats['delay_adjustments']}")
    
    print(f"\n⚡ Processing Time:")
    print(f"   Original: {old_time:.2f}s")
    print(f"   Improved: {new_time:.2f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("📝 SUMMARY")
    print("=" * 60)
    
    if new_stats['underrun_rate'] < old_stats['underrun_rate'] * 0.5:
        print("✅ Improved AEC shows SIGNIFICANT improvement in handling bursty TTS!")
        print("   - Much lower underrun rate")
        print("   - Better buffer management")
        print("   - More stable performance")
    elif new_stats['underrun_rate'] < old_stats['underrun_rate']:
        print("✅ Improved AEC shows moderate improvement")
        print("   - Lower underrun rate")
        print("   - Better burst handling")
    else:
        print("⚠️ Results are similar - check configuration")
    
    print(f"\n💡 Key improvements in new implementation:")
    print("   1. Larger initial delay (200ms vs 100ms) for TTS bursts")
    print("   2. Prebuffering phase to avoid initial underruns")
    print("   3. Better burst detection and handling")
    print("   4. Smoother adaptation algorithm")
    print("   5. Improved frame alignment handling")

if __name__ == "__main__":
    main()