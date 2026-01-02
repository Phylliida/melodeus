import collections
import numpy as np
from typing import Optional
import time

class DelayedEchoCancellationProcessor:
    """
    Echo cancellation with dynamic delay based on queue depth tracking.
    """
    def __init__(self, frame_size: int, filter_length: int, sample_rate: int):
        # Import here to avoid issues if not installed
        import speexdsp
        
        self.echo_canceller = speexdsp.EchoCanceller(
            frame_size=frame_size,
            filter_length=filter_length
        )
        
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.bytes_per_frame = frame_size * 2  # 16-bit audio
        
        # Circular buffer to delay reference audio
        max_delay_seconds = 5  # Support up to 5 seconds of delay
        buffer_size = int(sample_rate * max_delay_seconds * 2)  # 2 bytes per sample
        self.reference_buffer = collections.deque(maxlen=buffer_size)
        
        # Track queue depth when reference audio is added
        self.queue_depth_tracker = collections.deque()
        self.last_playback_time = time.time()
        
    def add_reference_audio(self, audio_data: bytes, queue_depth: int = 0):
        """
        Add reference audio with queue depth information.
        
        Args:
            audio_data: The audio bytes being sent to playback
            queue_depth: Current number of chunks in the playback queue
        """
        # Calculate expected playback time based on queue depth
        # Assume each chunk is ~100ms of audio (typical for TTS)
        chunk_duration = 0.1  # seconds
        expected_delay = queue_depth * chunk_duration
        
        # Store audio with timing information
        timestamp = time.time() + expected_delay
        self.queue_depth_tracker.append((timestamp, len(self.reference_buffer)))
        
        # Add to circular buffer
        self.reference_buffer.extend(audio_data)
        
    def process(self, near_data: bytes) -> bytes:
        """
        Process microphone input with delayed reference audio.
        """
        current_time = time.time()
        
        # Find the right position in reference buffer based on timing
        reference_position = 0
        while self.queue_depth_tracker and self.queue_depth_tracker[0][0] <= current_time:
            _, reference_position = self.queue_depth_tracker.popleft()
        
        # Extract appropriate reference audio
        output = bytearray()
        near_array = np.frombuffer(near_data, dtype=np.int16)
        
        for i in range(0, len(near_array), self.frame_size):
            near_frame = near_array[i:i+self.frame_size]
            if len(near_frame) < self.frame_size:
                break
                
            # Get corresponding reference frame
            ref_start = reference_position + i * 2
            ref_end = ref_start + self.frame_size * 2
            
            if ref_end <= len(self.reference_buffer):
                # Extract reference data from buffer
                ref_bytes = bytes([self.reference_buffer[j] for j in range(ref_start, ref_end)])
                ref_frame = np.frombuffer(ref_bytes, dtype=np.int16)
            else:
                # No reference audio available yet (silence)
                ref_frame = np.zeros(self.frame_size, dtype=np.int16)
            
            # Process with echo canceller
            processed_bytes = self.echo_canceller.process(
                near_frame.tobytes(),
                ref_frame.tobytes()
            )
            output.extend(processed_bytes)
            
        return bytes(output)
 