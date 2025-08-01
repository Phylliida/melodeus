#!/usr/bin/env python3
"""
Test the actual effectiveness of echo cancellation, not just buffer metrics.
"""

import numpy as np
from speexdsp import EchoCanceller
import matplotlib.pyplot as plt

def test_echo_cancellation_effectiveness():
    """Test if echo cancellation is actually working."""
    
    print("🧪 Testing Echo Cancellation Effectiveness")
    print("=" * 50)
    
    # Parameters
    sample_rate = 16000
    frame_size = 256
    filter_length = 2048
    
    # Create echo canceller
    ec = EchoCanceller.create(frame_size, filter_length, sample_rate)
    
    # Test 1: Simple sine wave echo
    print("\n📊 Test 1: Sine wave echo cancellation")
    
    # Generate reference (speaker) signal
    t = np.arange(frame_size) / sample_rate
    reference = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Generate near-end signal (mic = echo + noise)
    echo = reference * 0.8  # 80% echo
    noise = np.random.randn(frame_size) * 0.01  # Small noise
    near_end = echo + noise
    
    # Process
    ref_bytes = (reference * 32767).astype(np.int16).tobytes()
    near_bytes = (near_end * 32767).astype(np.int16).tobytes()
    
    result_bytes = ec.process(near_bytes, ref_bytes)
    result = np.frombuffer(result_bytes, dtype=np.int16).astype(float) / 32767
    
    # Calculate echo reduction
    echo_power_before = np.mean(echo ** 2)
    echo_power_after = np.mean(result ** 2)
    reduction_db = 10 * np.log10(echo_power_before / max(echo_power_after, 1e-10))
    
    print(f"  Echo power before: {echo_power_before:.6f}")
    print(f"  Echo power after: {echo_power_after:.6f}")
    print(f"  Echo reduction: {reduction_db:.1f} dB")
    
    # Test 2: With proper training
    print("\n📊 Test 2: Echo cancellation with adaptation")
    
    # Reset
    ec.reset()
    
    # Train the echo canceller with several frames
    print("  Training echo canceller...")
    for i in range(100):
        # Generate consistent echo scenario
        phase = i * frame_size / sample_rate * 2 * np.pi * 440
        reference = np.sin(phase + t * 2 * np.pi * 440) * 0.5
        echo = reference * 0.8
        near_end = echo + np.random.randn(frame_size) * 0.01
        
        ref_bytes = (reference * 32767).astype(np.int16).tobytes()
        near_bytes = (near_end * 32767).astype(np.int16).tobytes()
        
        result_bytes = ec.process(near_bytes, ref_bytes)
        
        if i % 20 == 0:
            result = np.frombuffer(result_bytes, dtype=np.int16).astype(float) / 32767
            residual_power = np.mean(result ** 2)
            print(f"    Frame {i}: residual power = {residual_power:.6f}")
    
    # Test final performance
    print("\n  Testing after adaptation...")
    reference = np.sin(t * 2 * np.pi * 440) * 0.5
    echo = reference * 0.8
    near_end = echo + np.random.randn(frame_size) * 0.01
    
    ref_bytes = (reference * 32767).astype(np.int16).tobytes()
    near_bytes = (near_end * 32767).astype(np.int16).tobytes()
    
    result_bytes = ec.process(near_bytes, ref_bytes)
    result = np.frombuffer(result_bytes, dtype=np.int16).astype(float) / 32767
    
    echo_power_before = np.mean(echo ** 2)
    echo_power_after = np.mean(result ** 2)
    reduction_db = 10 * np.log10(echo_power_before / max(echo_power_after, 1e-10))
    
    print(f"  Echo power before: {echo_power_before:.6f}")
    print(f"  Echo power after: {echo_power_after:.6f}")
    print(f"  Echo reduction after training: {reduction_db:.1f} dB")
    
    # Test 3: Delay mismatch
    print("\n📊 Test 3: Effect of delay mismatch")
    
    for delay_samples in [0, 64, 128, 256]:
        ec.reset()
        
        # Create delayed echo
        reference = np.sin(t * 2 * np.pi * 440) * 0.5
        echo = np.zeros(frame_size)
        if delay_samples < frame_size:
            echo[delay_samples:] = reference[:-delay_samples] * 0.8
        
        near_end = echo + np.random.randn(frame_size) * 0.01
        
        ref_bytes = (reference * 32767).astype(np.int16).tobytes()
        near_bytes = (near_end * 32767).astype(np.int16).tobytes()
        
        # Process multiple frames to let it adapt
        for _ in range(10):
            result_bytes = ec.process(near_bytes, ref_bytes)
        
        result = np.frombuffer(result_bytes, dtype=np.int16).astype(float) / 32767
        
        if delay_samples < frame_size:
            echo_power = np.mean(echo[delay_samples:] ** 2)
            result_power = np.mean(result[delay_samples:] ** 2)
            if echo_power > 0:
                reduction_db = 10 * np.log10(echo_power / max(result_power, 1e-10))
            else:
                reduction_db = 0
        else:
            reduction_db = 0
            
        print(f"  Delay {delay_samples} samples ({delay_samples/sample_rate*1000:.1f}ms): "
              f"reduction = {reduction_db:.1f} dB")

if __name__ == "__main__":
    test_echo_cancellation_effectiveness()
 