#!/usr/bin/env python3
"""
Acoustic Echo Cancellation module using speexdsp
Removes echo from microphone input using reference audio from TTS output
"""

import numpy as np
from typing import Optional, Tuple
from speexdsp import EchoCanceller
import struct
import threading
from collections import deque
import time


class EchoCancellationProcessor:
    """Processes audio to remove acoustic echo using speexdsp."""
    
    def __init__(self, 
                 frame_size: int = 256,
                 filter_length: int = 2048, 
                 sample_rate: int = 16000,
                 reference_delay_ms: int = 100):
        """
        Initialize echo cancellation processor.
        
        Args:
            frame_size: Number of samples per frame (must be power of 2)
            filter_length: Length of the echo cancellation filter
            sample_rate: Audio sample rate in Hz
            reference_delay_ms: Delay to account for audio output latency
        """
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        self.reference_delay_ms = reference_delay_ms
        
        # Calculate delay in samples
        self.delay_samples = int((reference_delay_ms / 1000.0) * sample_rate)
        
        # Create echo canceller
        self.echo_canceller = EchoCanceller.create(frame_size, filter_length, sample_rate)
        
        # Reference audio buffer (ring buffer for delay compensation)
        self.reference_buffer_size = self.delay_samples + frame_size * 10  # Extra buffer
        self.reference_buffer = deque(maxlen=self.reference_buffer_size)
        
        # Initialize with silence
        for _ in range(self.reference_buffer_size):
            self.reference_buffer.append(0)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Stats
        self.frames_processed = 0
        self.enabled = True
        
        print(f"🔊 Echo cancellation initialized:")
        print(f"   Frame size: {frame_size} samples")
        print(f"   Filter length: {filter_length} samples")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Reference delay: {reference_delay_ms} ms ({self.delay_samples} samples)")
    
    def add_reference_audio(self, audio_data: bytes):
        """
        Add reference audio (TTS output) to the buffer.
        
        Args:
            audio_data: Raw audio data (16-bit PCM)
        """
        if not self.enabled:
            return
            
        with self.lock:
            # Convert bytes to samples
            samples = struct.unpack(f'{len(audio_data)//2}h', audio_data)
            
            # Add to reference buffer
            self.reference_buffer.extend(samples)
    
    def process(self, mic_audio: bytes) -> bytes:
        """
        Process microphone audio to remove echo.
        
        Args:
            mic_audio: Raw microphone audio data (16-bit PCM)
            
        Returns:
            Processed audio with echo removed
        """
        if not self.enabled or len(mic_audio) != self.frame_size * 2:
            return mic_audio
        
        with self.lock:
            try:
                # Get delayed reference audio
                ref_start = len(self.reference_buffer) - self.delay_samples - self.frame_size
                if ref_start < 0:
                    # Not enough reference data yet, return original
                    return mic_audio
                
                # Extract reference frame
                ref_samples = []
                for i in range(self.frame_size):
                    ref_samples.append(self.reference_buffer[ref_start + i])
                
                # Convert to bytes
                ref_audio = struct.pack(f'{len(ref_samples)}h', *ref_samples)
                
                # Process with echo canceller
                # mic_audio is "near" (with echo), ref_audio is "far" (reference)
                processed = self.echo_canceller.process(mic_audio, ref_audio)
                
                self.frames_processed += 1
                
                return processed
                
            except Exception as e:
                print(f"❌ Echo cancellation error: {e}")
                return mic_audio
    
    def reset(self):
        """Reset the echo canceller state."""
        with self.lock:
            # Recreate echo canceller
            self.echo_canceller = EchoCanceller.create(
                self.frame_size, 
                self.filter_length, 
                self.sample_rate
            )
            
            # Clear reference buffer
            self.reference_buffer.clear()
            for _ in range(self.reference_buffer_size):
                self.reference_buffer.append(0)
            
            self.frames_processed = 0
            print("🔄 Echo cancellation reset")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable echo cancellation."""
        self.enabled = enabled
        if enabled:
            self.reset()
        print(f"🔊 Echo cancellation {'enabled' if enabled else 'disabled'}")
    
    def get_stats(self) -> dict:
        """Get echo cancellation statistics."""
        return {
            'enabled': self.enabled,
            'frames_processed': self.frames_processed,
            'reference_buffer_fill': len(self.reference_buffer),
            'delay_ms': self.reference_delay_ms
        }


class AdaptiveEchoCancellation:
    """Adaptive echo cancellation with automatic parameter tuning."""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize adaptive echo cancellation."""
        self.sample_rate = sample_rate
        
        # Try different frame sizes for best performance
        self.frame_sizes = [128, 256, 512]
        self.current_frame_size = 256
        
        # Create processor
        self.processor = EchoCancellationProcessor(
            frame_size=self.current_frame_size,
            filter_length=2048,
            sample_rate=sample_rate,
            reference_delay_ms=100  # Start with 100ms delay
        )
        
        # Adaptation parameters
        self.adaptation_enabled = True
        self.min_delay_ms = 50
        self.max_delay_ms = 300
        self.delay_step_ms = 10
        
    def adapt_delay(self, correlation_score: float):
        """
        Adapt reference delay based on correlation between input and output.
        
        Args:
            correlation_score: Measure of echo presence (0-1, higher = more echo)
        """
        if not self.adaptation_enabled:
            return
            
        current_delay = self.processor.reference_delay_ms
        
        # If high correlation (echo present), try adjusting delay
        if correlation_score > 0.7:
            # Try increasing delay
            new_delay = min(current_delay + self.delay_step_ms, self.max_delay_ms)
            if new_delay != current_delay:
                print(f"📊 Adapting echo delay: {current_delay}ms -> {new_delay}ms")
                self.processor.reference_delay_ms = new_delay
                self.processor.delay_samples = int((new_delay / 1000.0) * self.sample_rate)
        elif correlation_score < 0.3 and current_delay > self.min_delay_ms:
            # Low correlation, can try reducing delay for lower latency
            new_delay = max(current_delay - self.delay_step_ms, self.min_delay_ms)
            if new_delay != current_delay:
                print(f"📊 Adapting echo delay: {current_delay}ms -> {new_delay}ms")
                self.processor.reference_delay_ms = new_delay
                self.processor.delay_samples = int((new_delay / 1000.0) * self.sample_rate)


# Example usage
if __name__ == "__main__":
    # Create echo canceller
    aec = EchoCancellationProcessor(
        frame_size=256,
        filter_length=2048,
        sample_rate=16000,
        reference_delay_ms=100
    )
    
    print("Echo cancellation module ready")
    
    # In practice:
    # 1. Feed TTS output to add_reference_audio()
    # 2. Process microphone input with process()
    # 3. Send processed audio to STT