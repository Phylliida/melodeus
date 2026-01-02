#!/usr/bin/env python3
"""
Test script to verify improved echo cancellation performance.
"""

import numpy as np
import time
from improved_aec_processor import ImprovedEchoCancellationProcessor

def test_basic_functionality():
    """Test basic echo cancellation functionality."""
    print("🧪 Test 1: Basic Functionality")
    print("=" * 50)
    
    processor = ImprovedEchoCancellationProcessor(
        frame_size=256,
        filter_length=2048,
        sample_rate=16000,
        initial_delay_ms=200,
        debug_level=1
    )
    
    # Add some reference audio
    for i in range(20):
        ref_audio = np.random.randint(-1000, 1000, 256, dtype=np.int16).tobytes()
        processor.add_reference_audio(ref_audio)
    
    # Process near-end audio
    for i in range(20):
        near_audio = np.random.randint(-500, 500, 256, dtype=np.int16).tobytes()
        processed = processor.process(near_audio)
        assert len(processed) == len(near_audio), "Output size mismatch"
    
    processor.print_stats()
    stats = processor.get_stats()
    assert stats['underrun_rate'] < 0.1, "Too many underruns in basic test"
    print("✅ Basic functionality test passed\n")

def test_bursty_tts():
    """Test handling of bursty TTS output."""
    print("🧪 Test 2: Bursty TTS Handling")
    print("=" * 50)
    
    processor = ImprovedEchoCancellationProcessor(
        frame_size=256,
        filter_length=2048,
        sample_rate=16000,
        initial_delay_ms=200,
        debug_level=1
    )
    
    # Simulate prebuffering phase
    print("📦 Prebuffering phase...")
    for i in range(10):
        ref_audio = np.random.randint(-1000, 1000, 256, dtype=np.int16).tobytes()
        processor.add_reference_audio(ref_audio)
    
    # Simulate TTS bursts like in real usage
    print("💥 Testing TTS bursts...")
    burst_sizes_ms = [1723, 830, 1532, 511, 1723]  # From your log
    
    for burst_ms in burst_sizes_ms:
        # Send burst
        samples = int(burst_ms * 16)  # 16 samples per ms at 16kHz
        ref_audio = np.random.randint(-5000, 5000, samples, dtype=np.int16).tobytes()
        processor.add_reference_audio(ref_audio)
        
        # Process microphone audio during and after burst
        frames_to_process = int(burst_ms / 16) + 10  # 16ms per frame
        for i in range(frames_to_process):
            near_audio = np.random.randint(-500, 500, 256, dtype=np.int16).tobytes()
            processed = processor.process(near_audio)
            time.sleep(0.001)  # Small delay to simulate real-time
    
    processor.print_stats()
    stats = processor.get_stats()
    print(f"\n📊 Burst test results:")
    print(f"   Underrun rate: {stats['underrun_rate']*100:.1f}%")
    print(f"   Max buffer fill: {stats['max_fill_level']} frames")
    print(f"   Bursts handled: {stats['burst_count']}")
    
    assert stats['underrun_rate'] < 0.15, "Too many underruns with bursts"
    print("✅ Bursty TTS test passed\n")

def test_underrun_recovery():
    """Test recovery from buffer underruns."""
    print("🧪 Test 3: Underrun Recovery")
    print("=" * 50)
    
    processor = ImprovedEchoCancellationProcessor(
        frame_size=256,
        filter_length=2048,
        sample_rate=16000,
        initial_delay_ms=100,  # Lower delay to force underruns
        debug_level=1
    )
    
    # Disable adaptation for controlled test
    processor.enable_adaptation(False)
    
    # Process without enough reference audio to cause underruns
    print("🔴 Forcing underruns...")
    for i in range(50):
        near_audio = np.random.randint(-500, 500, 256, dtype=np.int16).tobytes()
        processed = processor.process(near_audio)
    
    initial_stats = processor.get_stats()
    print(f"   Underruns: {initial_stats['buffer_underruns']}")
    
    # Now add reference audio and verify recovery
    print("🟢 Adding reference audio for recovery...")
    for i in range(30):
        ref_audio = np.random.randint(-1000, 1000, 256, dtype=np.int16).tobytes()
        processor.add_reference_audio(ref_audio)
    
    # Process with available reference
    underruns_before = initial_stats['buffer_underruns']
    for i in range(30):
        near_audio = np.random.randint(-500, 500, 256, dtype=np.int16).tobytes()
        processed = processor.process(near_audio)
    
    final_stats = processor.get_stats()
    new_underruns = final_stats['buffer_underruns'] - underruns_before
    
    print(f"   New underruns after recovery: {new_underruns}")
    assert new_underruns < 5, "Failed to recover from underruns"
    print("✅ Underrun recovery test passed\n")

def test_adaptation():
    """Test automatic delay adaptation."""
    print("🧪 Test 4: Automatic Delay Adaptation")
    print("=" * 50)
    
    processor = ImprovedEchoCancellationProcessor(
        frame_size=256,
        filter_length=2048,
        sample_rate=16000,
        initial_delay_ms=100,  # Start with low delay
        debug_level=1
    )
    
    initial_delay = processor.current_delay_frames
    print(f"📏 Initial delay: {initial_delay} frames")
    
    # Simulate conditions that should trigger adaptation
    print("🔄 Simulating variable conditions...")
    
    # Phase 1: Cause underruns to trigger increase
    for cycle in range(150):  # ~2.4 seconds
        if cycle % 10 < 3:  # 30% reference data
            ref_audio = np.random.randint(-1000, 1000, 256, dtype=np.int16).tobytes()
            processor.add_reference_audio(ref_audio)
        
        near_audio = np.random.randint(-500, 500, 256, dtype=np.int16).tobytes()
        processed = processor.process(near_audio)
        time.sleep(0.001)
    
    mid_stats = processor.get_stats()
    mid_delay = processor.current_delay_frames
    print(f"📏 Delay after underruns: {mid_delay} frames")
    assert mid_delay > initial_delay, "Delay should increase with underruns"
    
    # Phase 2: Provide excess reference to potentially trigger decrease
    print("📦 Providing excess reference data...")
    for cycle in range(200):  # ~3.2 seconds
        # Always add reference
        ref_audio = np.random.randint(-1000, 1000, 256, dtype=np.int16).tobytes()
        processor.add_reference_audio(ref_audio)
        
        # Process less frequently
        if cycle % 2 == 0:
            near_audio = np.random.randint(-500, 500, 256, dtype=np.int16).tobytes()
            processed = processor.process(near_audio)
        time.sleep(0.001)
    
    final_stats = processor.get_stats()
    final_delay = processor.current_delay_frames
    
    processor.print_stats()
    print(f"\n📊 Adaptation results:")
    print(f"   Initial delay: {initial_delay} frames")
    print(f"   After underruns: {mid_delay} frames") 
    print(f"   Final delay: {final_delay} frames")
    print(f"   Total adjustments: {final_stats['delay_adjustments']}")
    
    assert final_stats['delay_adjustments'] > 0, "No delay adaptations occurred"
    print("✅ Adaptation test passed\n")

def test_frame_alignment():
    """Test proper frame alignment with various chunk sizes."""
    print("🧪 Test 5: Frame Alignment")
    print("=" * 50)
    
    processor = ImprovedEchoCancellationProcessor(
        frame_size=256,
        filter_length=2048,
        sample_rate=16000,
        initial_delay_ms=200,
        debug_level=2  # Verbose to see alignment warnings
    )
    
    # Test various non-aligned chunk sizes
    test_sizes = [
        (512, "Aligned (2 frames)"),
        (513, "Misaligned by 1 sample"),
        (640, "2.5 frames"),
        (1000, "3.9 frames"),
        (256*5 + 128, "5.5 frames")
    ]
    
    print("🔧 Testing various chunk sizes...")
    for size, description in test_sizes:
        print(f"\n   Testing {size} samples ({description}):")
        
        # Reset processor
        processor.reset()
        
        # Add reference audio of this size
        ref_audio = np.random.randint(-1000, 1000, size, dtype=np.int16).tobytes()
        processor.add_reference_audio(ref_audio)
        
        # Process some near-end audio
        for i in range(10):
            near_audio = np.random.randint(-500, 500, 256, dtype=np.int16).tobytes()
            processed = processor.process(near_audio)
        
        stats = processor.get_stats()
        print(f"   Realignments: {stats['realignment_count']}")
    
    print("\n✅ Frame alignment test completed\n")

def run_all_tests():
    """Run all tests."""
    print("🚀 Running Improved AEC Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_functionality,
        test_bursty_tts,
        test_underrun_recovery,
        test_adaptation,
        test_frame_alignment
    ]
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"❌ Test failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Test error: {e}")
            return False
    
    print("✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)