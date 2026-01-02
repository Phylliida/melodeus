#!/usr/bin/env python3
"""
Improved Acoustic Echo Cancellation (AEC) module with proper synchronization.
"""

import numpy as np
from speexdsp import EchoCanceller
import collections
import threading
import time
from typing import Optional, Tuple

class ImprovedEchoCancellationProcessor:
    """
    Processes audio streams to remove echo using speexdsp with proper timing synchronization.
    """
    def __init__(self, frame_size: int, filter_length: int, sample_rate: int, 
                 reference_delay_ms: int = 100, buffer_duration_ms: int = 5000):
        """
        Initializes the ImprovedEchoCancellationProcessor.

        Args:
            frame_size (int): The number of samples per frame for processing.
            filter_length (int): The length of the echo cancellation filter.
            sample_rate (int): The audio sample rate (e.g., 16000 Hz).
            reference_delay_ms (int): The estimated acoustic delay in milliseconds.
            buffer_duration_ms (int): Duration of reference buffer in milliseconds.
        """
        if not (frame_size > 0 and (frame_size & (frame_size - 1) == 0)):
            raise ValueError("frame_size must be a power of 2")
        if not (filter_length > frame_size and (filter_length & (filter_length - 1) == 0)):
            raise ValueError("filter_length must be a power of 2 and greater than frame_size")

        self.frame_size = frame_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        self.reference_delay_samples = int(sample_rate * (reference_delay_ms / 1000.0))
        
        # Calculate buffer size based on duration
        self.buffer_size_samples = int(sample_rate * (buffer_duration_ms / 1000.0))
        
        # Create echo canceller
        self.echo_canceller = EchoCanceller.create(frame_size, filter_length, sample_rate)
        
        # Circular buffer for reference audio with timestamps
        self.reference_buffer = collections.deque(maxlen=self.buffer_size_samples)
        self.reference_timestamps = collections.deque(maxlen=self.buffer_size_samples)
        
        # Buffer for accumulating samples to frame_size
        self.near_buffer = np.array([], dtype=np.int16)
        self.far_buffer = np.array([], dtype=np.int16)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.frames_processed = 0
        self.frames_skipped = 0
        
        print(f"🔊 ImprovedEchoCancellationProcessor initialized:")
        print(f"   Frame size: {frame_size} samples")
        print(f"   Filter length: {filter_length} samples")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Acoustic delay: {reference_delay_ms} ms ({self.reference_delay_samples} samples)")
        print(f"   Buffer duration: {buffer_duration_ms} ms ({self.buffer_size_samples} samples)")

    def add_reference_audio(self, audio_data: bytes, timestamp: Optional[float] = None):
        """
        Adds far-end (reference) audio data with timestamp.
        
        Args:
            audio_data (bytes): Raw audio bytes (16-bit PCM).
            timestamp (float): When this audio is/was played. If None, uses current time.
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Convert bytes to int16 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        with self.lock:
            # Add each sample with its timestamp
            for sample in audio_np:
                self.reference_buffer.append(sample)
                self.reference_timestamps.append(timestamp)
                # Increment timestamp for each sample
                timestamp += 1.0 / self.sample_rate

    def process(self, near_end_audio: bytes, capture_timestamp: Optional[float] = None) -> bytes:
        """
        Processes near-end (microphone) audio to remove echo.
        
        Args:
            near_end_audio (bytes): Raw audio bytes from the microphone.
            capture_timestamp (float): When this audio was captured. If None, uses current time.
            
        Returns:
            bytes: Echo-cancelled audio bytes.
        """
        if capture_timestamp is None:
            capture_timestamp = time.time()
            
        near_end_np = np.frombuffer(near_end_audio, dtype=np.int16)
        
        with self.lock:
            # Add to near buffer
            self.near_buffer = np.concatenate([self.near_buffer, near_end_np])
            
            # Process complete frames
            output_buffer = np.array([], dtype=np.int16)
            
            while len(self.near_buffer) >= self.frame_size:
                # Extract one frame
                near_frame = self.near_buffer[:self.frame_size]
                self.near_buffer = self.near_buffer[self.frame_size:]
                
                # Find corresponding reference audio based on timestamp
                reference_timestamp = capture_timestamp - (self.reference_delay_samples / self.sample_rate)
                
                # Find the reference samples that match this timestamp
                far_frame = self._get_reference_frame_at_time(reference_timestamp)
                
                if far_frame is not None:
                    # Process with echo canceller
                    processed_frame = self.echo_canceller.process(near_frame, far_frame)
                    self.frames_processed += 1
                else:
                    # No reference data available, pass through
                    processed_frame = near_frame
                    self.frames_skipped += 1
                    
                output_buffer = np.concatenate([output_buffer, processed_frame])
                
                # Update timestamp for next frame
                capture_timestamp += self.frame_size / self.sample_rate
            
            return output_buffer.tobytes()

    def _get_reference_frame_at_time(self, target_timestamp: float) -> Optional[np.ndarray]:
        """
        Retrieves reference audio frame that was playing at the given timestamp.
        """
        if not self.reference_timestamps:
            return None
            
        # Find the closest timestamp in our buffer
        min_time = self.reference_timestamps[0]
        max_time = self.reference_timestamps[-1]
        
        if target_timestamp < min_time or target_timestamp > max_time:
            return None
            
        # Find starting index for this timestamp
        # Simple linear search for now (could optimize with binary search)
        start_idx = None
        for i, ts in enumerate(self.reference_timestamps):
            if ts >= target_timestamp:
                start_idx = i
                break
                
        if start_idx is None or start_idx + self.frame_size > len(self.reference_buffer):
            return None
            
        # Extract frame
        frame_data = []
        for i in range(self.frame_size):
            if start_idx + i < len(self.reference_buffer):
                frame_data.append(self.reference_buffer[start_idx + i])
            else:
                frame_data.append(0)  # Pad with zeros if needed
                
        return np.array(frame_data, dtype=np.int16)

    def get_stats(self) -> dict:
        """Returns processing statistics."""
        with self.lock:
            total = self.frames_processed + self.frames_skipped
            return {
                'frames_processed': self.frames_processed,
                'frames_skipped': self.frames_skipped,
                'success_rate': self.frames_processed / total if total > 0 else 0,
                'buffer_fill': len(self.reference_buffer) / self.buffer_size_samples
            }

    def reset(self):
        """Resets the echo canceller state."""
        with self.lock:
            self.echo_canceller.reset()
            self.reference_buffer.clear()
            self.reference_timestamps.clear()
            self.near_buffer = np.array([], dtype=np.int16)
            self.far_buffer = np.array([], dtype=np.int16)
            self.frames_processed = 0
            self.frames_skipped = 0
            print("🔄 ImprovedEchoCancellationProcessor reset.")


class SimpleEchoCancellationProcessor:
    """
    A simpler echo cancellation processor that uses a fixed delay buffer.
    This is more suitable for real-time streaming where timing is critical.
    """
    def __init__(self, frame_size: int, filter_length: int, sample_rate: int, 
                 reference_delay_ms: int = 100):
        """
        Initializes the SimpleEchoCancellationProcessor.
        """
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        self.delay_frames = int((reference_delay_ms / 1000.0) * sample_rate / frame_size)
        
        # Create echo canceller
        self.echo_canceller = EchoCanceller.create(frame_size, filter_length, sample_rate)
        
        # Circular buffer for reference frames
        self.reference_frames = collections.deque(maxlen=self.delay_frames + 10)
        
        # Buffers for accumulating samples
        self.near_buffer = bytearray()
        self.far_buffer = bytearray()
        
        # Thread safety
        self.lock = threading.Lock()
        
        print(f"🔊 SimpleEchoCancellationProcessor initialized:")
        print(f"   Frame size: {frame_size} samples ({frame_size * 1000 / sample_rate:.1f} ms)")
        print(f"   Delay: {self.delay_frames} frames ({reference_delay_ms} ms)")

    def add_reference_audio(self, audio_data: bytes):
        """
        Adds reference audio data.
        """
        with self.lock:
            self.far_buffer.extend(audio_data)
            
            # Process complete frames
            bytes_per_frame = self.frame_size * 2  # 16-bit = 2 bytes
            while len(self.far_buffer) >= bytes_per_frame:
                frame_bytes = bytes(self.far_buffer[:bytes_per_frame])
                self.far_buffer = self.far_buffer[bytes_per_frame:]
                
                # Convert to numpy array and store
                frame_np = np.frombuffer(frame_bytes, dtype=np.int16)
                self.reference_frames.append(frame_np)

    def process(self, near_end_audio: bytes) -> bytes:
        """
        Processes near-end audio to remove echo.
        """
        with self.lock:
            self.near_buffer.extend(near_end_audio)
            
            output = bytearray()
            bytes_per_frame = self.frame_size * 2
            
            while len(self.near_buffer) >= bytes_per_frame:
                # Extract near-end frame
                near_bytes = bytes(self.near_buffer[:bytes_per_frame])
                self.near_buffer = self.near_buffer[bytes_per_frame:]
                near_frame = np.frombuffer(near_bytes, dtype=np.int16)
                
                # Get reference frame from delay buffer
                if len(self.reference_frames) >= self.delay_frames:
                    # Use the delayed reference frame
                    far_frame = self.reference_frames[0]
                    self.reference_frames.popleft()
                else:
                    # Not enough reference data yet, use zeros
                    far_frame = np.zeros(self.frame_size, dtype=np.int16)
                
                # Process with echo canceller
                # speexdsp expects bytes, not numpy arrays
                processed_bytes = self.echo_canceller.process(near_frame.tobytes(), far_frame.tobytes())
                output.extend(processed_bytes)
            
            return bytes(output)

    def reset(self):
        """Resets the processor."""
        with self.lock:
            self.echo_canceller.reset()
            self.reference_frames.clear()
            self.near_buffer.clear()
            self.far_buffer.clear()
            print("🔄 SimpleEchoCancellationProcessor reset.")
 