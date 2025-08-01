#!/usr/bin/env python3
"""
Simple test for echo cancellation functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from speexdsp import EchoCanceller
import wave
import struct

def generate_test_signal(duration=2.0, frequency=440.0, sample_rate=16000):
    """Generate a test sine wave signal."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    signal = np.sin(2 * np.pi * frequency * t)
    return signal

def add_echo(signal, delay_samples=800, attenuation=0.5):
    """Add a simple echo to the signal."""
    echo_signal = np.zeros_like(signal)
    echo_signal[delay_samples:] = signal[:-delay_samples] * attenuation
    return signal + echo_signal

def test_echo_cancellation():
    """Test basic echo cancellation."""
    print("🔊 Testing Echo Cancellation...")
    
    # Parameters
    sample_rate = 16000
    frame_size = 256
    filter_length = 2048
    
    try:
        # Create echo canceller
        echo_canceller = EchoCanceller.create(frame_size, filter_length, sample_rate)
        print("✅ Echo canceller created successfully!")
        print(f"   Frame size: {frame_size}")
        print(f"   Filter length: {filter_length}")
        print(f"   Sample rate: {sample_rate}")
        
        # Generate test signals
        duration = 2.0
        
        # Far-end signal (speaker output)
        far_signal = generate_test_signal(duration, 440.0, sample_rate)
        
        # Near-end signal (microphone input with echo)
        near_signal = generate_test_signal(duration, 880.0, sample_rate) * 0.3  # User's voice
        echo = add_echo(far_signal, delay_samples=800, attenuation=0.5)
        near_with_echo = near_signal + echo
        
        print("\n📊 Processing audio frames...")
        
        # Process in frames
        num_frames = len(far_signal) // frame_size
        processed_signal = np.zeros(num_frames * frame_size)
        
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            
            # Convert to bytes (16-bit PCM)
            far_frame = (far_signal[start:end] * 32767).astype(np.int16).tobytes()
            near_frame = (near_with_echo[start:end] * 32767).astype(np.int16).tobytes()
            
            # Process with echo canceller
            processed_frame = echo_canceller.process(near_frame, far_frame)
            
            # Convert back to float
            processed_samples = np.frombuffer(processed_frame, dtype=np.int16).astype(np.float32) / 32767
            processed_signal[start:end] = processed_samples
        
        print(f"✅ Processed {num_frames} frames")
        
        # Calculate echo reduction
        echo_power_before = np.mean(echo[:num_frames * frame_size] ** 2)
        echo_power_after = np.mean((processed_signal - near_signal[:num_frames * frame_size]) ** 2)
        echo_reduction_db = 10 * np.log10(echo_power_before / max(echo_power_after, 1e-10))
        
        print(f"\n📈 Results:")
        print(f"   Echo power before: {echo_power_before:.6f}")
        print(f"   Echo power after: {echo_power_after:.6f}")
        print(f"   Echo reduction: {echo_reduction_db:.1f} dB")
        
        # Save results
        save_results = input("\n💾 Save audio files for comparison? (y/n): ")
        if save_results.lower() == 'y':
            # Save original with echo
            with wave.open('test_with_echo.wav', 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((near_with_echo[:num_frames * frame_size] * 32767).astype(np.int16).tobytes())
            
            # Save processed
            with wave.open('test_processed.wav', 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((processed_signal * 32767).astype(np.int16).tobytes())
            
            print("✅ Saved test_with_echo.wav and test_processed.wav")
        
        # Plot results
        plot_results = input("\n📊 Plot results? (y/n): ")
        if plot_results.lower() == 'y':
            time = np.arange(num_frames * frame_size) / sample_rate
            
            plt.figure(figsize=(12, 8))
            
            # Plot original near signal
            plt.subplot(3, 1, 1)
            plt.plot(time, near_signal[:num_frames * frame_size])
            plt.title('Original Near Signal (User Voice)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            # Plot with echo
            plt.subplot(3, 1, 2)
            plt.plot(time, near_with_echo[:num_frames * frame_size])
            plt.title('Near Signal with Echo')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            # Plot processed
            plt.subplot(3, 1, 3)
            plt.plot(time, processed_signal)
            plt.title('Processed Signal (Echo Removed)')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        print("\n✅ Echo cancellation test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_realtime_simulation():
    """Simulate real-time echo cancellation."""
    print("\n🎤 Testing Real-time Simulation...")
    
    sample_rate = 16000
    frame_size = 256
    filter_length = 2048
    
    try:
        # Create echo canceller
        echo_canceller = EchoCanceller.create(frame_size, filter_length, sample_rate)
        
        # Simulate 100 frames of real-time processing
        print("Processing 100 frames in real-time simulation...")
        
        for frame_num in range(100):
            # Generate random audio (simulating microphone and speaker)
            far_frame = np.random.randn(frame_size) * 0.1
            near_frame = far_frame * 0.5 + np.random.randn(frame_size) * 0.05  # Echo + noise
            
            # Convert to bytes
            far_bytes = (far_frame * 32767).astype(np.int16).tobytes()
            near_bytes = (near_frame * 32767).astype(np.int16).tobytes()
            
            # Process
            processed = echo_canceller.process(near_bytes, far_bytes)
            
            if frame_num % 20 == 0:
                print(f"   Processed frame {frame_num}")
        
        print("✅ Real-time simulation completed!")
        
    except Exception as e:
        print(f"❌ Real-time test error: {e}")

if __name__ == "__main__":
    print("🔊 Echo Cancellation Test Suite")
    print("================================\n")
    
    # Test basic functionality
    test_echo_cancellation()
    
    # Test real-time simulation
    test_realtime_simulation()
 