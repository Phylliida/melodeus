#!/usr/bin/env python3
"""
Diagnose why echo cancellation isn't working.
"""

import numpy as np
from speexdsp import EchoCanceller
import struct

def test_basic_echo_cancellation():
    """Test if echo cancellation works at all."""
    
    print("🔍 Diagnosing Echo Cancellation Issues")
    print("=" * 50)
    
    # Parameters matching your system
    sample_rate = 16000
    frame_size = 256
    filter_length = 2048
    
    # Create echo canceller
    print(f"\n📊 Creating echo canceller:")
    print(f"  Frame size: {frame_size} samples")
    print(f"  Filter length: {filter_length} samples")
    print(f"  Sample rate: {sample_rate} Hz")
    
    ec = EchoCanceller.create(frame_size, filter_length, sample_rate)
    
    # Test 1: Check what happens with identical signals
    print("\n🧪 Test 1: Identical reference and near-end signals")
    
    # Generate a test signal
    t = np.arange(frame_size) / sample_rate
    signal = np.sin(2 * np.pi * 440 * t) * 0.5
    signal_int16 = (signal * 32767).astype(np.int16)
    signal_bytes = signal_int16.tobytes()
    
    # Process identical signals
    result_bytes = ec.process(signal_bytes, signal_bytes)
    result_int16 = np.frombuffer(result_bytes, dtype=np.int16)
    result = result_int16.astype(float) / 32767
    
    # Compare
    input_power = np.mean(signal ** 2)
    output_power = np.mean(result ** 2)
    
    print(f"  Input power: {input_power:.6f}")
    print(f"  Output power: {output_power:.6f}")
    print(f"  Power ratio: {output_power/input_power:.3f}")
    
    # Test 2: Check with multiple iterations
    print("\n🧪 Test 2: Processing multiple frames")
    
    for i in range(5):
        result_bytes = ec.process(signal_bytes, signal_bytes)
        result = np.frombuffer(result_bytes, dtype=np.int16).astype(float) / 32767
        output_power = np.mean(result ** 2)
        print(f"  Frame {i+1}: output power = {output_power:.6f}")
    
    # Test 3: Test with actual echo scenario
    print("\n🧪 Test 3: Realistic echo scenario")
    
    # Reference signal (speaker output)
    reference = np.sin(t * 2 * np.pi * 440) * 0.3
    
    # Near-end signal (microphone = echo + local speech)
    echo = reference * 0.7  # 70% of speaker signal
    local_speech = np.sin(t * 2 * np.pi * 880) * 0.1  # Different frequency
    near_end = echo + local_speech
    
    # Convert to bytes
    ref_bytes = (reference * 32767).astype(np.int16).tobytes()
    near_bytes = (near_end * 32767).astype(np.int16).tobytes()
    
    # Process
    result_bytes = ec.process(near_bytes, ref_bytes)
    result = np.frombuffer(result_bytes, dtype=np.int16).astype(float) / 32767
    
    echo_power = np.mean(echo ** 2)
    local_power = np.mean(local_speech ** 2)
    output_power = np.mean(result ** 2)
    
    print(f"  Echo power: {echo_power:.6f}")
    print(f"  Local speech power: {local_power:.6f}")
    print(f"  Output power: {output_power:.6f}")
    print(f"  Expected remaining: ~{local_power:.6f}")
    
    # Test 4: Check the data format
    print("\n🧪 Test 4: Data format verification")
    
    # Check if the library expects different data format
    test_samples = np.array([0, 16383, -16384, 32767, -32768], dtype=np.int16)
    test_bytes = test_samples.tobytes()
    
    print(f"  Test samples: {test_samples}")
    print(f"  As bytes: {[b for b in test_bytes[:10]]}")
    
    # Verify round-trip
    recovered = np.frombuffer(test_bytes, dtype=np.int16)
    print(f"  Recovered: {recovered}")
    print(f"  Match: {np.array_equal(test_samples, recovered)}")
    
    # Test 5: Try with the C API expectations
    print("\n🧪 Test 5: Testing different delay compensations")
    
    # The echo canceller might expect the reference to be delayed
    for delay_frames in [0, 1, 2, 4]:
        # Create new canceller for each test
        ec_test = EchoCanceller.create(frame_size, filter_length, sample_rate)
        
        # Generate signals
        reference = np.sin(t * 2 * np.pi * 440) * 0.5
        echo = reference * 0.8
        
        # Process multiple frames to train
        for i in range(10):
            ref_bytes = (reference * 32767).astype(np.int16).tobytes()
            near_bytes = (echo * 32767).astype(np.int16).tobytes()
            
            if i >= delay_frames:
                result_bytes = ec_test.process(near_bytes, ref_bytes)
                if i == 9:  # Last frame
                    result = np.frombuffer(result_bytes, dtype=np.int16).astype(float) / 32767
                    residual = np.mean(result ** 2)
                    print(f"  Delay {delay_frames} frames: residual = {residual:.6f}")

if __name__ == "__main__":
    test_basic_echo_cancellation()
 