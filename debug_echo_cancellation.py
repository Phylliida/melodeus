#!/usr/bin/env python3
"""
Debugging version of echo cancellation with detailed logging
"""

import numpy as np
from speexdsp import EchoCanceller
import collections
import threading
import time
from typing import Optional
import os

class DebugEchoCancellationProcessor:
    """
    Echo cancellation processor with extensive debugging
    """
    def __init__(self, frame_size: int, filter_length: int, sample_rate: int, 
                 reference_delay_ms: int = 100, debug_level: int = 1):
        """
        Initialize with debugging capabilities.
        
        Args:
            debug_level: 0=off, 1=basic, 2=detailed, 3=verbose
        """
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        self.delay_frames = int((reference_delay_ms / 1000.0) * sample_rate / frame_size)
        self.debug_level = debug_level
        
        # Create echo canceller
        self.echo_canceller = EchoCanceller.create(frame_size, filter_length, sample_rate)
        
        # Circular buffer for reference frames
        self.reference_frames = collections.deque(maxlen=self.delay_frames + 10)
        
        # Buffers for accumulating samples
        self.near_buffer = bytearray()
        self.far_buffer = bytearray()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Debug statistics
        self.stats = {
            'reference_calls': 0,
            'process_calls': 0,
            'frames_processed': 0,
            'buffer_underruns': 0,
            'last_reference_time': 0,
            'last_process_time': 0,
            'total_reference_bytes': 0,
            'total_near_bytes': 0
        }
        
        # Debug log file
        if debug_level >= 2:
            self.debug_file = open(f'echo_debug_{int(time.time())}.log', 'w')
        else:
            self.debug_file = None
            
        print(f"🔊 DebugEchoCancellationProcessor initialized:")
        print(f"   Frame size: {frame_size} samples ({frame_size * 1000 / sample_rate:.1f} ms)")
        print(f"   Delay: {self.delay_frames} frames ({reference_delay_ms} ms)")
        print(f"   Debug level: {debug_level}")

    def add_reference_audio(self, audio_data: bytes):
        """
        Adds reference audio data with detailed logging.
        """
        with self.lock:
            self.stats['reference_calls'] += 1
            self.stats['total_reference_bytes'] += len(audio_data)
            current_time = time.time()
            
            if self.debug_level >= 2:
                time_since_last = current_time - self.stats['last_reference_time']
                self._log(f"REF: {len(audio_data)} bytes, {time_since_last*1000:.1f}ms since last")
                
            self.stats['last_reference_time'] = current_time
            
            # Add to buffer
            self.far_buffer.extend(audio_data)
            
            # Process complete frames
            bytes_per_frame = self.frame_size * 2  # 16-bit = 2 bytes
            frames_added = 0
            
            while len(self.far_buffer) >= bytes_per_frame:
                frame_bytes = bytes(self.far_buffer[:bytes_per_frame])
                self.far_buffer = self.far_buffer[bytes_per_frame:]
                
                # Convert to numpy array for storage
                frame_np = np.frombuffer(frame_bytes, dtype=np.int16)
                self.reference_frames.append(frame_np)
                frames_added += 1
                
            if self.debug_level >= 3:
                self._log(f"REF: Added {frames_added} frames, buffer has {len(self.reference_frames)} frames")

    def process(self, near_end_audio: bytes) -> bytes:
        """
        Processes near-end audio with detailed logging.
        """
        with self.lock:
            self.stats['process_calls'] += 1
            self.stats['total_near_bytes'] += len(near_end_audio)
            current_time = time.time()
            
            if self.debug_level >= 2:
                time_since_last = current_time - self.stats['last_process_time']
                self._log(f"PROC: {len(near_end_audio)} bytes, {time_since_last*1000:.1f}ms since last")
                
            self.stats['last_process_time'] = current_time
            
            # Add to buffer
            self.near_buffer.extend(near_end_audio)
            
            output = bytearray()
            bytes_per_frame = self.frame_size * 2
            frames_processed = 0
            
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
                    
                    if self.debug_level >= 3:
                        self._log(f"PROC: Using reference frame, {len(self.reference_frames)} frames remaining")
                else:
                    # Not enough reference data yet
                    far_frame = np.zeros(self.frame_size, dtype=np.int16)
                    self.stats['buffer_underruns'] += 1
                    
                    if self.debug_level >= 1:
                        self._log(f"PROC: Buffer underrun! Only {len(self.reference_frames)} frames available")
                
                # Process with echo canceller
                try:
                    # speexdsp expects bytes
                    processed_bytes = self.echo_canceller.process(near_bytes, far_frame.tobytes())
                    output.extend(processed_bytes)
                    frames_processed += 1
                    self.stats['frames_processed'] += 1
                except Exception as e:
                    self._log(f"ERROR: Echo cancellation failed: {e}")
                    output.extend(near_bytes)  # Fallback to original
                    
            if self.debug_level >= 2:
                self._log(f"PROC: Processed {frames_processed} frames")
                
            return bytes(output)

    def get_stats(self):
        """Get current statistics"""
        with self.lock:
            return self.stats.copy()
            
    def print_stats(self):
        """Print current statistics"""
        stats = self.get_stats()
        print("\n📊 ECHO CANCELLATION STATISTICS:")
        print(f"   Reference calls: {stats['reference_calls']}")
        print(f"   Process calls: {stats['process_calls']}")
        print(f"   Frames processed: {stats['frames_processed']}")
        print(f"   Buffer underruns: {stats['buffer_underruns']}")
        print(f"   Total reference: {stats['total_reference_bytes']/1024:.1f} KB")
        print(f"   Total near-end: {stats['total_near_bytes']/1024:.1f} KB")
        
        if stats['reference_calls'] > 0:
            avg_ref_size = stats['total_reference_bytes'] / stats['reference_calls']
            print(f"   Avg reference chunk: {avg_ref_size:.0f} bytes")
            
        if stats['frames_processed'] > 0:
            underrun_rate = stats['buffer_underruns'] / stats['frames_processed'] * 100
            print(f"   Underrun rate: {underrun_rate:.1f}%")

    def _log(self, message: str):
        """Log a debug message"""
        timestamp = time.time()
        log_msg = f"[{timestamp:.3f}] {message}"
        
        if self.debug_file:
            self.debug_file.write(log_msg + '\n')
            self.debug_file.flush()
            
        if self.debug_level >= 3:
            print(f"🔍 {log_msg}")

    def reset(self):
        """Reset the processor"""
        with self.lock:
            self.echo_canceller.reset()
            self.reference_frames.clear()
            self.near_buffer.clear()
            self.far_buffer.clear()
            print("🔄 DebugEchoCancellationProcessor reset.")
            
    def __del__(self):
        """Cleanup"""
        if self.debug_file:
            self.debug_file.close()
            
            
# Test timing analysis
if __name__ == "__main__":
    print("🧪 Testing Echo Cancellation Timing")
    print("===================================")
    
    # Create processor with verbose debugging
    processor = DebugEchoCancellationProcessor(
        frame_size=256,
        filter_length=2048,
        sample_rate=16000,
        reference_delay_ms=100,
        debug_level=2
    )
    
    # Simulate audio flow
    import random
    
    print("\n📊 Simulating audio flow...")
    
    # Simulate TTS sending reference audio in chunks
    for i in range(10):
        # TTS chunks can vary in size
        chunk_size = random.choice([512, 1024, 2048])
        ref_audio = np.random.randint(-1000, 1000, chunk_size//2, dtype=np.int16).tobytes()
        processor.add_reference_audio(ref_audio)
        time.sleep(0.05)  # 50ms between chunks
        
    # Simulate microphone audio
    for i in range(20):
        # Microphone usually has consistent chunk size
        mic_audio = np.random.randint(-1000, 1000, 256, dtype=np.int16).tobytes()
        result = processor.process(mic_audio)
        time.sleep(0.016)  # ~16ms for 256 samples at 16kHz
        
    # Print statistics
    #processor.print_stats()
 