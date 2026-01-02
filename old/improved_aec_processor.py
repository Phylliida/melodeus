#!/usr/bin/env python3
"""
Improved echo cancellation processor with better handling of bursty TTS output.
Addresses timing mismatches and buffer underruns.
"""

import numpy as np
from speexdsp import EchoCanceller
import collections
import threading
import time
from typing import Optional, Deque
import statistics

class ImprovedEchoCancellationProcessor:
    """
    Echo cancellation processor optimized for real-time voice conversations
    with bursty TTS output and regular microphone input.
    """
    def __init__(self, frame_size: int, filter_length: int, sample_rate: int, 
                 initial_delay_ms: int = 200, debug_level: int = 1):
        """
        Initialize with improved buffering and timing.
        
        Args:
            frame_size: Samples per frame (e.g., 256 for 16ms at 16kHz)
            filter_length: Echo filter length in samples (e.g., 2048)
            sample_rate: Audio sample rate in Hz (e.g., 16000)
            initial_delay_ms: Initial delay in milliseconds (default 200ms for TTS)
            debug_level: Debug output level (0=none, 1=basic, 2=verbose)
        """
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        self.debug_level = debug_level
        
        # Increased delay parameters for TTS bursts
        self.initial_delay_ms = initial_delay_ms
        self.current_delay_frames = int((initial_delay_ms / 1000.0) * sample_rate / frame_size)
        self.min_delay_frames = int((150 / 1000.0) * sample_rate / frame_size)  # Min 150ms
        self.max_delay_frames = int((1000 / 1000.0) * sample_rate / frame_size)  # Max 1s
        
        # Create echo canceller
        self.echo_canceller = EchoCanceller.create(frame_size, filter_length, sample_rate)
        
        # Larger ring buffer for bursty TTS
        self.reference_buffer_size = self.max_delay_frames + 200  # Extra headroom
        self.reference_frames = collections.deque(maxlen=self.reference_buffer_size)
        
        # Frame accumulation buffers
        self.near_buffer = bytearray()
        self.far_buffer = bytearray()
        
        # Pre-buffer for TTS bursts
        self.prebuffer_frames = int((100 / 1000.0) * sample_rate / frame_size)  # 100ms prebuffer
        self.prebuffering = True
        self.prebuffer_count = 0
        
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
            'consecutive_underruns': 0,
            'max_fill_level': 0,
            'burst_count': 0,
            'realignment_count': 0
        }
        
        # Delay adaptation with smoother adjustments
        self.adaptation_enabled = True
        self.fill_levels = collections.deque(maxlen=200)  # More samples
        self.last_adaptation_time = time.time()
        self.adaptation_interval = 2.0  # Slower adaptation
        
        # Timing tracking
        self.last_reference_time = 0
        self.reference_rate_samples = collections.deque(maxlen=50)
        
        print(f"🔊 ImprovedEchoCancellationProcessor initialized:")
        print(f"   Frame size: {frame_size} samples ({frame_size * 1000 / sample_rate:.1f} ms)")
        print(f"   Initial delay: {self.current_delay_frames} frames ({initial_delay_ms} ms)")
        print(f"   Adaptive delay range: {self.min_delay_frames}-{self.max_delay_frames} frames")
        print(f"   Prebuffer: {self.prebuffer_frames} frames")
        print(f"   Debug level: {debug_level}")

    def add_reference_audio(self, audio_data: bytes):
        """
        Adds reference audio data with improved burst handling.
        """
        with self.lock:
            self.stats['reference_calls'] += 1
            current_time = time.time()
            
            # Track timing between reference calls
            if self.last_reference_time > 0:
                time_diff = current_time - self.last_reference_time
                samples_diff = len(audio_data) // 2
                self.reference_rate_samples.append((time_diff, samples_diff))
            self.last_reference_time = current_time
            
            # Detect bursts
            samples = len(audio_data) // 2
            frames = samples / self.frame_size
            
            if frames > 10:  # Large burst detected
                self.stats['burst_count'] += 1
                if self.debug_level >= 1:
                    print(f"💥 TTS burst: {samples} samples ({frames:.1f} frames, {samples*1000/self.sample_rate:.1f}ms)")
            
            # Add to buffer for frame alignment
            self.far_buffer.extend(audio_data)
            
            # Process complete frames with proper alignment
            bytes_per_frame = self.frame_size * 2
            frames_added = 0
            
            while len(self.far_buffer) >= bytes_per_frame:
                # Extract exactly one frame
                frame_bytes = bytes(self.far_buffer[:bytes_per_frame])
                self.far_buffer = self.far_buffer[bytes_per_frame:]
                
                # Convert to numpy array
                frame_np = np.frombuffer(frame_bytes, dtype=np.int16)
                self.reference_frames.append(frame_np)
                frames_added += 1
                
                # Track statistics
                self.stats['current_fill_level'] = len(self.reference_frames)
                if self.stats['current_fill_level'] > self.stats['max_fill_level']:
                    self.stats['max_fill_level'] = self.stats['current_fill_level']
            
            # Handle prebuffering phase
            if self.prebuffering:
                self.prebuffer_count += frames_added
                if self.prebuffer_count >= self.prebuffer_frames:
                    self.prebuffering = False
                    if self.debug_level >= 1:
                        print(f"✅ Prebuffering complete: {self.prebuffer_count} frames ready")
            
            # Track fill level for adaptation
            self.fill_levels.append(self.stats['current_fill_level'])
            
            # Realign if we have partial frame data
            if len(self.far_buffer) > 0 and self.debug_level >= 2:
                print(f"⚠️ Partial frame: {len(self.far_buffer)} bytes remaining")
                self.stats['realignment_count'] += 1

    def process(self, near_end_audio: bytes) -> bytes:
        """
        Processes near-end audio with improved underrun handling.
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
                
                # Skip echo cancellation during prebuffering
                if self.prebuffering:
                    output.extend(near_bytes)
                    continue
                
                # Get reference frame with improved handling
                far_frame, status = self._get_reference_frame_improved()
                
                if status == 'underrun':
                    self.stats['buffer_underruns'] += 1
                    self.stats['consecutive_underruns'] += 1
                elif status == 'ok':
                    self.stats['consecutive_underruns'] = 0
                
                # Process with echo canceller
                try:
                    if status != 'underrun':
                        processed_bytes = self.echo_canceller.process(near_bytes, far_frame.tobytes())
                        output.extend(processed_bytes)
                    else:
                        # On underrun, pass through original audio
                        output.extend(near_bytes)
                    
                    self.stats['frames_processed'] += 1
                    
                except Exception as e:
                    if self.debug_level >= 1:
                        print(f"❌ Echo cancellation error: {e}")
                    output.extend(near_bytes)
            
            # Adapt delay if needed
            self._adapt_delay_improved()
            
            return bytes(output)

    def _get_reference_frame_improved(self) -> tuple[np.ndarray, str]:
        """
        Get reference frame with improved underrun handling.
        Returns (frame, status) where status is 'ok', 'low', or 'underrun'.
        """
        buffer_len = len(self.reference_frames)
        
        # Check buffer status
        if buffer_len > self.current_delay_frames:
            # Normal operation
            frame = self.reference_frames.popleft()
            return frame, 'ok'
        elif buffer_len > self.min_delay_frames // 2:
            # Low buffer but usable
            frame = self.reference_frames.popleft()
            if self.debug_level >= 2:
                print(f"⚠️ Buffer low: {buffer_len} frames (target: {self.current_delay_frames})")
            return frame, 'low'
        elif buffer_len > 0:
            # Very low, use what we have
            frame = self.reference_frames.popleft()
            return frame, 'low'
        else:
            # Complete underrun
            # if self.debug_level >= 1 and self.stats['consecutive_underruns'] % 10 == 0:
            #     print(f"❌ Buffer underrun streak: {self.stats['consecutive_underruns']}")
            return np.zeros(self.frame_size, dtype=np.int16), 'underrun'

    def _adapt_delay_improved(self):
        """
        Improved delay adaptation based on buffer performance and TTS patterns.
        """
        if not self.adaptation_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return
            
        if len(self.fill_levels) < 50:  # Need more data
            return
            
        # Calculate statistics
        avg_fill = statistics.mean(self.fill_levels)
        min_fill = min(self.fill_levels)
        max_fill = max(self.fill_levels)
        fill_variance = statistics.variance(self.fill_levels) if len(self.fill_levels) > 1 else 0
        
        # Calculate underrun rate
        if self.stats['frames_processed'] > 0:
            underrun_rate = self.stats['buffer_underruns'] / self.stats['frames_processed']
        else:
            underrun_rate = 0
        
        adapted = False
        
        # Aggressive increase on high underrun rate
        if underrun_rate > 0.1 or self.stats['consecutive_underruns'] > 20:
            increase = min(5, (self.max_delay_frames - self.current_delay_frames) // 2)
            if increase > 0:
                self.current_delay_frames += increase
                adapted = True
                if self.debug_level >= 1:
                    delay_ms = self.current_delay_frames * self.frame_size * 1000 / self.sample_rate
                    print(f"📈 Increased delay by {increase} to {self.current_delay_frames} frames ({delay_ms:.0f}ms) - high underrun rate: {underrun_rate:.1%}")
        
        # Moderate increase on consistent low buffer
        elif min_fill < self.current_delay_frames * 0.5 and underrun_rate > 0.05:
            if self.current_delay_frames < self.max_delay_frames - 2:
                self.current_delay_frames += 2
                adapted = True
                if self.debug_level >= 1:
                    delay_ms = self.current_delay_frames * self.frame_size * 1000 / self.sample_rate
                    print(f"📈 Increased delay to {self.current_delay_frames} frames ({delay_ms:.0f}ms) - buffer frequently low")
        
        # Conservative decrease only with very stable buffer
        elif (avg_fill > self.current_delay_frames * 2 and 
              min_fill > self.current_delay_frames and 
              underrun_rate < 0.01 and
              fill_variance < (self.current_delay_frames ** 2) * 0.25):
            if self.current_delay_frames > self.min_delay_frames + 5:
                self.current_delay_frames -= 1
                adapted = True
                if self.debug_level >= 1:
                    delay_ms = self.current_delay_frames * self.frame_size * 1000 / self.sample_rate
                    print(f"📉 Reduced delay to {self.current_delay_frames} frames ({delay_ms:.0f}ms) - buffer very stable")
        
        if adapted:
            self.stats['delay_adjustments'] += 1
            self.last_adaptation_time = current_time
            # Don't clear fill levels, just trim old data
            while len(self.fill_levels) > 100:
                self.fill_levels.popleft()
            # Reset consecutive underrun count after adaptation
            self.stats['consecutive_underruns'] = 0

    def get_stats(self):
        """Get current statistics with additional metrics."""
        with self.lock:
            stats = self.stats.copy()
            if stats['frames_processed'] > 0:
                stats['underrun_rate'] = stats['buffer_underruns'] / stats['frames_processed']
            else:
                stats['underrun_rate'] = 0
            stats['current_delay_ms'] = self.current_delay_frames * self.frame_size * 1000 / self.sample_rate
            stats['prebuffering'] = self.prebuffering
            
            # Calculate average reference data rate if available
            if len(self.reference_rate_samples) > 10:
                total_time = sum(t for t, s in self.reference_rate_samples)
                total_samples = sum(s for t, s in self.reference_rate_samples)
                if total_time > 0:
                    stats['avg_reference_rate'] = total_samples / total_time
                else:
                    stats['avg_reference_rate'] = 0
            else:
                stats['avg_reference_rate'] = 0
                
            return stats

    def print_stats(self):
        """Print current statistics with enhanced information."""
        stats = self.get_stats()
        print("\n📊 IMPROVED ECHO CANCELLATION STATISTICS:")
        print(f"   Reference calls: {stats['reference_calls']}")
        print(f"   Process calls: {stats['process_calls']}")
        print(f"   Frames processed: {stats['frames_processed']}")
        print(f"   Buffer underruns: {stats['buffer_underruns']}")
        print(f"   Current delay: {stats['current_delay_ms']:.0f}ms ({self.current_delay_frames} frames)")
        print(f"   Buffer fill: {stats['current_fill_level']} frames (max: {stats['max_fill_level']})")
        print(f"   Delay adjustments: {stats['delay_adjustments']}")
        print(f"   TTS bursts detected: {stats['burst_count']}")
        print(f"   Realignments: {stats['realignment_count']}")
        
        if stats['frames_processed'] > 0:
            print(f"   Underrun rate: {stats['underrun_rate']*100:.1f}%")
        
        if stats['avg_reference_rate'] > 0:
            print(f"   Avg reference rate: {stats['avg_reference_rate']:.0f} samples/sec")
        
        if stats['prebuffering']:
            print(f"   Status: PREBUFFERING ({self.prebuffer_count}/{self.prebuffer_frames} frames)")

    def reset(self):
        """Reset the processor to initial state."""
        with self.lock:
            self.echo_canceller.reset()
            self.reference_frames.clear()
            self.near_buffer.clear()
            self.far_buffer.clear()
            self.fill_levels.clear()
            self.reference_rate_samples.clear()
            
            # Reset to initial delay
            self.current_delay_frames = int((self.initial_delay_ms / 1000.0) * self.sample_rate / self.frame_size)
            
            # Reset prebuffering
            self.prebuffering = True
            self.prebuffer_count = 0
            
            # Reset timing
            self.last_reference_time = 0
            
            # Reset stats but keep some for debugging
            for key in self.stats:
                self.stats[key] = 0
                
            print("🔄 ImprovedEchoCancellationProcessor reset.")

    def enable_adaptation(self, enabled: bool):
        """Enable or disable automatic delay adaptation."""
        self.adaptation_enabled = enabled
        print(f"🔧 Delay adaptation {'enabled' if enabled else 'disabled'}")

    def set_delay_ms(self, delay_ms: int):
        """Manually set the delay in milliseconds."""
        with self.lock:
            new_delay_frames = int((delay_ms / 1000.0) * self.sample_rate / self.frame_size)
            new_delay_frames = max(self.min_delay_frames, min(self.max_delay_frames, new_delay_frames))
            self.current_delay_frames = new_delay_frames
            print(f"🔧 Delay manually set to {self.current_delay_frames} frames ({delay_ms}ms)")


if __name__ == "__main__":
    print("🧪 Testing Improved Echo Cancellation")
    print("=====================================")
    
    # Create processor
    processor = ImprovedEchoCancellationProcessor(
        frame_size=256,
        filter_length=2048,
        sample_rate=16000,
        initial_delay_ms=200,  # Higher initial delay
        debug_level=10
    )
    
    # Simulate realistic TTS bursts
    import random
    
    print("\n📊 Simulating realistic TTS/microphone interaction...")
    
    # Phase 1: Initial thinking sound (regular)
    print("\n--- Phase 1: Thinking Sound ---")
    for i in range(30):
        # Regular thinking sound pulses
        ref_audio = np.random.randint(-1000, 1000, 256, dtype=np.int16).tobytes()
        processor.add_reference_audio(ref_audio)
        
        # Microphone processes regularly
        if i > 5:  # Give some initial buffer
            mic_audio = np.random.randint(-100, 100, 256, dtype=np.int16).tobytes()
            result = processor.process(mic_audio)
        
        time.sleep(0.016)
    
    #processor.print_stats()
    
    # Phase 2: TTS speech (bursty)
    print("\n--- Phase 2: TTS Speech (Bursty) ---")
    for i in range(50):
        # TTS sends large chunks irregularly
        if random.random() < 0.3:  # 30% chance of TTS data
            # Simulate realistic TTS chunk sizes from your log
            chunk_size = random.choice([1723, 830, 1532, 511, 1723])  # ms
            samples = int(chunk_size * 16)  # 16 samples per ms at 16kHz
            ref_audio = np.random.randint(-5000, 5000, samples, dtype=np.int16).tobytes()
            processor.add_reference_audio(ref_audio)
        
        # Microphone always processes
        mic_audio = np.random.randint(-200, 200, 256, dtype=np.int16).tobytes()
        result = processor.process(mic_audio)
        
        time.sleep(0.016)
    
    #processor.print_stats()
    
    # Phase 3: Silence
    print("\n--- Phase 3: Silence ---")
    for i in range(30):
        # Only microphone, no reference
        mic_audio = np.random.randint(-50, 50, 256, dtype=np.int16).tobytes()
        result = processor.process(mic_audio)
        time.sleep(0.016)
    
    #processor.print_stats()
    
    print("\n✅ Test completed!")