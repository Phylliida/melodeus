#!/usr/bin/env python3
"""
Advanced echo cancellation with adaptive delay and better synchronization.
"""

import numpy as np
from speexdsp import EchoCanceller
import collections
import threading
import time
from typing import Optional, Deque
import statistics

class AdaptiveEchoCancellationProcessor:
    """
    Echo cancellation processor with adaptive delay estimation.
    """
    def __init__(self, frame_size: int, filter_length: int, sample_rate: int, 
                 initial_delay_ms: int = 100, debug_level: int = 1):
        """
        Initialize with adaptive delay capabilities.
        """
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        self.debug_level = debug_level
        
        # Adaptive delay parameters
        self.initial_delay_ms = initial_delay_ms
        self.current_delay_frames = int((initial_delay_ms / 1000.0) * sample_rate / frame_size)
        self.min_delay_frames = max(3, int((50 / 1000.0) * sample_rate / frame_size))  # Min 50ms
        self.max_delay_frames = int((500 / 1000.0) * sample_rate / frame_size)  # Max 500ms
        
        # Create echo canceller
        self.echo_canceller = EchoCanceller.create(frame_size, filter_length, sample_rate)
        
        # Ring buffer for reference frames with timestamps
        self.reference_buffer_size = self.max_delay_frames + 50
        self.reference_frames = collections.deque(maxlen=self.reference_buffer_size)
        self.reference_timestamps = collections.deque(maxlen=self.reference_buffer_size)
        
        # Buffers for accumulating samples
        self.near_buffer = bytearray()
        self.far_buffer = bytearray()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance tracking
        self.stats = {
            'reference_calls': 0,
            'process_calls': 0,
            'frames_processed': 0,
            'buffer_underruns': 0,
            'buffer_overruns': 0,
            'delay_adjustments': 0,
            'current_fill_level': 0,
            'last_underrun_time': 0,
            'underrun_streak': 0
        }
        
        # Delay adaptation
        self.adaptation_enabled = True
        self.fill_levels = collections.deque(maxlen=100)
        self.last_adaptation_time = time.time()
        
        print(f"🔊 AdaptiveEchoCancellationProcessor initialized:")
        print(f"   Frame size: {frame_size} samples ({frame_size * 1000 / sample_rate:.1f} ms)")
        print(f"   Initial delay: {self.current_delay_frames} frames ({initial_delay_ms} ms)")
        print(f"   Adaptive delay: {self.min_delay_frames}-{self.max_delay_frames} frames")
        print(f"   Debug level: {debug_level}")

    def add_reference_audio(self, audio_data: bytes):
        """
        Adds reference audio data with timestamp.
        """
        with self.lock:
            self.stats['reference_calls'] += 1
            current_time = time.time()
            
            # Debug: Check for bursts and alignment
            samples = len(audio_data) // 2
            frames = samples / self.frame_size
            if frames > 4:  # More than 4 frames at once
                print(f"⚠️ BURST: Reference audio burst of {samples} samples ({frames:.1f} frames)")
            if samples % self.frame_size != 0:
                print(f"⚠️ MISALIGNED: Reference audio not frame-aligned: {samples} samples ({frames:.2f} frames)")
            
            # Add to buffer
            self.far_buffer.extend(audio_data)
            
            # Process complete frames
            bytes_per_frame = self.frame_size * 2
            
            while len(self.far_buffer) >= bytes_per_frame:
                frame_bytes = bytes(self.far_buffer[:bytes_per_frame])
                self.far_buffer = self.far_buffer[bytes_per_frame:]
                
                # Convert to numpy array and store with timestamp
                frame_np = np.frombuffer(frame_bytes, dtype=np.int16)
                self.reference_frames.append(frame_np)
                self.reference_timestamps.append(current_time)
                
            # Track buffer fill level
            self.stats['current_fill_level'] = len(self.reference_frames)
            self.fill_levels.append(self.stats['current_fill_level'])

    def process(self, near_end_audio: bytes) -> bytes:
        """
        Processes near-end audio with adaptive delay.
        """
        with self.lock:
            self.stats['process_calls'] += 1
            current_time = time.time()
            
            # Add to buffer
            self.near_buffer.extend(near_end_audio)
            
            output = bytearray()
            bytes_per_frame = self.frame_size * 2
            
            while len(self.near_buffer) >= bytes_per_frame:
                # Extract near-end frame
                near_bytes = bytes(self.near_buffer[:bytes_per_frame])
                self.near_buffer = self.near_buffer[bytes_per_frame:]
                
                # Get reference frame with adaptive delay
                far_frame, had_underrun = self._get_reference_frame_adaptive()
                
                if had_underrun:
                    self.stats['buffer_underruns'] += 1
                    self.stats['underrun_streak'] += 1
                    self.stats['last_underrun_time'] = current_time
                else:
                    self.stats['underrun_streak'] = 0
                
                # Process with echo canceller
                try:
                    processed_bytes = self.echo_canceller.process(near_bytes, far_frame.tobytes())
                    output.extend(processed_bytes)
                    self.stats['frames_processed'] += 1
                except Exception as e:
                    if self.debug_level >= 1:
                        print(f"❌ Echo cancellation error: {e}")
                    output.extend(near_bytes)
                    
            # Adapt delay if needed
            self._adapt_delay()
            
            return bytes(output)

    def _get_reference_frame_adaptive(self) -> tuple[np.ndarray, bool]:
        """
        Get reference frame with adaptive delay.
        Returns (frame, had_underrun).
        """
        buffer_len = len(self.reference_frames)
        
        # Check if we have enough frames
        if buffer_len > self.current_delay_frames:
            # Normal operation - use delayed frame
            frame = self.reference_frames.popleft()
            return frame, False
        elif buffer_len > 0:
            # Not enough delay, but we have some frames - use oldest
            frame = self.reference_frames.popleft()
            if self.debug_level >= 2:
                print(f"⚠️  Buffer low: {buffer_len} frames (need {self.current_delay_frames})")
            return frame, False
        else:
            # Complete underrun - no reference data
            if self.debug_level >= 2:
                print(f"❌ Buffer underrun! Delay: {self.current_delay_frames} frames")
            return np.zeros(self.frame_size, dtype=np.int16), True

    def _adapt_delay(self):
        """
        Adapt delay based on buffer performance.
        """
        if not self.adaptation_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_adaptation_time < 1.0:  # Adapt at most once per second
            return
            
        if len(self.fill_levels) < 20:  # Need enough data
            return
            
        # Calculate average fill level
        avg_fill = statistics.mean(self.fill_levels)
        
        # Adapt based on performance
        adapted = False
        
        # If we're having frequent underruns, increase delay
        if self.stats['underrun_streak'] > 5:
            if self.current_delay_frames < self.max_delay_frames:
                self.current_delay_frames += 1
                adapted = True
                if self.debug_level >= 1:
                    delay_ms = self.current_delay_frames * self.frame_size * 1000 / self.sample_rate
                    print(f"📈 Increased delay to {self.current_delay_frames} frames ({delay_ms:.0f}ms) due to underruns")
        
        # If buffer is consistently full AND we're well above minimum, we might reduce delay
        # But be conservative - only reduce if we have a lot of headroom
        elif avg_fill > self.current_delay_frames + 20 and self.stats['underrun_streak'] == 0:
            if self.current_delay_frames > self.min_delay_frames + 2:  # Keep some margin
                self.current_delay_frames -= 1
                adapted = True
                if self.debug_level >= 1:
                    delay_ms = self.current_delay_frames * self.frame_size * 1000 / self.sample_rate
                    print(f"📉 Reduced delay to {self.current_delay_frames} frames ({delay_ms:.0f}ms) - buffer very healthy")
        
        if adapted:
            self.stats['delay_adjustments'] += 1
            self.last_adaptation_time = current_time
            self.fill_levels.clear()  # Reset for next period

    def get_stats(self):
        """Get current statistics."""
        with self.lock:
            stats = self.stats.copy()
            if stats['frames_processed'] > 0:
                stats['underrun_rate'] = stats['buffer_underruns'] / stats['frames_processed']
            else:
                stats['underrun_rate'] = 0
            stats['current_delay_ms'] = self.current_delay_frames * self.frame_size * 1000 / self.sample_rate
            return stats

    def print_stats(self):
        """Print current statistics."""
        stats = self.get_stats()
        print("\n📊 ECHO CANCELLATION STATISTICS:")
        print(f"   Reference calls: {stats['reference_calls']}")
        print(f"   Process calls: {stats['process_calls']}")
        print(f"   Frames processed: {stats['frames_processed']}")
        print(f"   Buffer underruns: {stats['buffer_underruns']}")
        print(f"   Current delay: {stats['current_delay_ms']:.0f}ms ({self.current_delay_frames} frames)")
        print(f"   Buffer fill: {stats['current_fill_level']} frames")
        print(f"   Delay adjustments: {stats['delay_adjustments']}")
        
        if stats['frames_processed'] > 0:
            print(f"   Underrun rate: {stats['underrun_rate']*100:.1f}%")

    def reset(self):
        """Reset the processor."""
        with self.lock:
            self.echo_canceller.reset()
            self.reference_frames.clear()
            self.reference_timestamps.clear()
            self.near_buffer.clear()
            self.far_buffer.clear()
            self.fill_levels.clear()
            self.current_delay_frames = int((self.initial_delay_ms / 1000.0) * self.sample_rate / self.frame_size)
            
            # Reset stats
            for key in self.stats:
                self.stats[key] = 0
                
            print("🔄 AdaptiveEchoCancellationProcessor reset.")


if __name__ == "__main__":
    print("🧪 Testing Adaptive Echo Cancellation")
    print("=====================================")
    
    # Create processor
    processor = AdaptiveEchoCancellationProcessor(
        frame_size=256,
        filter_length=2048,
        sample_rate=16000,
        initial_delay_ms=100,
        debug_level=2
    )
    
    # Simulate realistic audio flow
    import random
    
    print("\n📊 Simulating audio flow with variable timing...")
    
    # Simulate thinking sound followed by speech
    for phase in range(3):
        print(f"\n--- Phase {phase + 1} ---")
        
        if phase == 0:
            # Thinking sound phase - regular pulses
            print("🎵 Thinking sound phase")
            for i in range(50):
                ref_audio = np.random.randint(-1000, 1000, 256, dtype=np.int16).tobytes()
                processor.add_reference_audio(ref_audio)
                
                # Microphone processes with slight delay
                if i > 5:
                    mic_audio = np.random.randint(-100, 100, 256, dtype=np.int16).tobytes()
                    result = processor.process(mic_audio)
                
                time.sleep(0.016)  # 16ms per frame
                
        elif phase == 1:
            # Speech phase - variable chunk sizes
            print("🗣️ Speech phase")
            for i in range(100):
                # TTS sends variable chunks
                if random.random() < 0.7:
                    chunk_size = random.choice([1024, 2048, 4096])
                    ref_audio = np.random.randint(-5000, 5000, chunk_size//2, dtype=np.int16).tobytes()
                    processor.add_reference_audio(ref_audio)
                
                # Microphone regular processing
                mic_audio = np.random.randint(-100, 100, 256, dtype=np.int16).tobytes()
                result = processor.process(mic_audio)
                
                time.sleep(0.016 + random.uniform(-0.005, 0.005))  # Some jitter
                
        else:
            # Silence phase
            print("🔇 Silence phase")
            for i in range(30):
                # Only microphone, no reference
                mic_audio = np.random.randint(-50, 50, 256, dtype=np.int16).tobytes()
                result = processor.process(mic_audio)
                time.sleep(0.016)
        
        processor.print_stats()
    
    print("\n✅ Test completed!")
 
