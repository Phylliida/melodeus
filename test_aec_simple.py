#!/usr/bin/env python3
"""
Simple test to verify AEC is actually reducing echo.
"""

import numpy as np
import time
from advanced_echo_cancellation import AdaptiveEchoCancellationProcessor

def generate_tone(frequency, duration, sample_rate=16000):
    """Generate a sine wave tone."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return (np.sin(2 * np.pi * frequency * t) * 32767 * 0.5).astype(np.int16)

def test_aec():
    # Initialize AEC
    aec = AdaptiveEchoCancellationProcessor(
        frame_size=256,
        filter_length=2048,
        sample_rate=16000,
        initial_delay_ms=100,
        debug_level=2
    )
    
    # Generate test signal (1 second of 440Hz tone)
    reference_signal = generate_tone(440, 1.0)
    
    # Simulate echo with some attenuation and delay
    echo_delay_samples = 1600  # 100ms delay  
    echo_gain = 0.5  # Reduced to avoid clipping
    
    # Create "microphone" signal with echo
    mic_signal = np.zeros(len(reference_signal) + echo_delay_samples, dtype=np.int16)
    mic_signal[echo_delay_samples:] = (reference_signal * echo_gain).astype(np.int16)
    
    # Add some "real" speech (different frequency)
    real_speech = generate_tone(880, 1.0)  # Higher frequency
    mic_signal[:len(real_speech)] += (real_speech * 0.3).astype(np.int16)
    
    # Process in chunks
    chunk_size = 256 * 2  # bytes
    processed_signal = bytearray()
    
    # Process in real-time fashion - feed reference and process mic together
    print("Processing in real-time mode...")
    
    # Pad reference to match mic signal length
    padded_reference = np.zeros_like(mic_signal)
    padded_reference[:len(reference_signal)] = reference_signal
    
    for i in range(0, len(mic_signal) * 2, chunk_size):
        # Feed reference chunk
        ref_chunk = padded_reference.tobytes()[i:i+chunk_size]
        if ref_chunk:
            aec.add_reference_audio(ref_chunk)
        
        # Process mic chunk
        mic_chunk = mic_signal.tobytes()[i:i+chunk_size]
        if mic_chunk:
            processed = aec.process(mic_chunk)
            processed_signal.extend(processed)
    
    # Convert back to numpy for analysis
    processed_array = np.frombuffer(processed_signal, dtype=np.int16)
    
    print(f"\nProcessed signal length: {len(processed_array)} samples")
    print(f"Original signal length: {len(mic_signal)} samples")
    
    # Calculate echo reduction
    echo_start = echo_delay_samples
    echo_end = min(echo_start + len(reference_signal), len(processed_array))
    
    if echo_end > echo_start and echo_start < len(processed_array):
        original_echo_power = np.mean(mic_signal[echo_start:echo_end] ** 2)
        processed_echo_power = np.mean(processed_array[echo_start:echo_end] ** 2)
        
        print(f"\nOriginal echo power: {original_echo_power:.0f}")
        print(f"Processed echo power: {processed_echo_power:.0f}")
        
        if original_echo_power > 0 and processed_echo_power > 0:
            echo_reduction_db = 10 * np.log10(processed_echo_power / original_echo_power)
            print(f"Echo reduction: {echo_reduction_db:.1f} dB")
            
            if echo_reduction_db < -6:
                print("✅ AEC is working! Significant echo reduction achieved.")
            else:
                print("❌ AEC not effective. Echo not significantly reduced.")
        elif processed_echo_power == 0:
            print("⚠️ AEC completely suppressed the signal (might be over-suppressing)")
        else:
            print("❌ No echo detected in original signal")
    
    # Print final stats
    aec.print_stats()

if __name__ == "__main__":
    test_aec()