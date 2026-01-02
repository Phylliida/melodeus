#!/usr/bin/env python3
"""
Test raw speexdsp to ensure it's working correctly.
"""

import numpy as np
import speexdsp

# Create echo canceller
frame_size = 256
filter_length = 2048
sample_rate = 16000

ec = speexdsp.EchoCanceller(frame_size, filter_length, sample_rate)

# Generate test signal
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration))
reference = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)

# Create echo (delayed reference)
echo_delay = int(0.1 * sample_rate)  # 100ms
echo = np.zeros(len(reference) + echo_delay, dtype=np.int16)
echo[echo_delay:] = (reference * 0.7).astype(np.int16)

# Add some "speech"
speech = (np.sin(2 * np.pi * 880 * t) * 8192).astype(np.int16)
mic_signal = echo.copy()
mic_signal[:len(speech)] += speech

# Process in real-time fashion
output = []
ref_padded = np.zeros_like(mic_signal)
ref_padded[:len(reference)] = reference

# Process frame by frame with correct timing
for i in range(0, len(mic_signal) - frame_size, frame_size):
    near_frame = mic_signal[i:i+frame_size]
    far_frame = ref_padded[i:i+frame_size]
    
    # Process expects bytes
    processed = ec.process(near_frame.tobytes(), far_frame.tobytes())
    output.extend(np.frombuffer(processed, dtype=np.int16))

output_array = np.array(output)

# Measure echo reduction
echo_region = slice(echo_delay, echo_delay + len(reference) - frame_size)
original_power = np.mean(mic_signal[echo_region] ** 2)
processed_power = np.mean(output_array[echo_region] ** 2)

if original_power > 0:
    reduction_db = 10 * np.log10(processed_power / original_power)
    print(f"Echo reduction: {reduction_db:.1f} dB")
    
    if reduction_db < -6:
        print("✅ Speex AEC is working correctly!")
    else:
        print("❌ Speex AEC not effective")
else:
    print("❌ No echo in test signal")