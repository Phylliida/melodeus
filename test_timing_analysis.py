#!/usr/bin/env python3
"""
Analyze the timing relationship between reference audio and microphone input.
"""

import time
import threading
import numpy as np
from collections import deque

class TimingAnalyzer:
    def __init__(self):
        self.reference_times = deque(maxlen=1000)
        self.process_times = deque(maxlen=1000)
        self.reference_chunks = deque(maxlen=1000)
        self.lock = threading.Lock()
        
    def add_reference(self, audio_data: bytes):
        """Called when reference audio is sent to AEC."""
        with self.lock:
            current_time = time.time()
            self.reference_times.append(current_time)
            self.reference_chunks.append(len(audio_data))
            print(f"📊 REF: {len(audio_data)} bytes at {current_time:.3f}")
            
    def process(self, near_data: bytes) -> bytes:
        """Called when microphone audio is processed."""
        with self.lock:
            current_time = time.time()
            self.process_times.append(current_time)
            
            # Find matching reference
            if self.reference_times:
                ref_time = self.reference_times[0]
                delay = (current_time - ref_time) * 1000  # ms
                print(f"📊 PROC: {len(near_data)} bytes at {current_time:.3f}, delay from ref: {delay:.1f}ms")
            
        return near_data
    
    def print_summary(self):
        """Print timing analysis."""
        with self.lock:
            if not self.reference_times or not self.process_times:
                print("No data collected")
                return
                
            print("\n📊 TIMING ANALYSIS:")
            print(f"Reference calls: {len(self.reference_times)}")
            print(f"Process calls: {len(self.process_times)}")
            
            # Calculate average chunk size
            if self.reference_chunks:
                avg_chunk = sum(self.reference_chunks) / len(self.reference_chunks)
                print(f"Average reference chunk: {avg_chunk:.0f} bytes ({avg_chunk/512:.1f} frames)")
            
            # Calculate timing gaps
            delays = []
            for proc_time in self.process_times:
                # Find closest reference time before this process time
                best_delay = None
                for ref_time in self.reference_times:
                    if ref_time <= proc_time:
                        delay = (proc_time - ref_time) * 1000
                        if best_delay is None or delay < best_delay:
                            best_delay = delay
                
                if best_delay is not None:
                    delays.append(best_delay)
            
            if delays:
                print(f"Average delay: {np.mean(delays):.1f}ms")
                print(f"Min delay: {np.min(delays):.1f}ms")
                print(f"Max delay: {np.max(delays):.1f}ms")

# Test it
if __name__ == "__main__":
    analyzer = TimingAnalyzer()
    
    # Simulate reference audio being sent
    print("Simulating reference audio...")
    for i in range(10):
        analyzer.add_reference(b'x' * 512)  # 256 samples
        time.sleep(0.016)  # 16ms
    
    # Simulate processing with delay
    print("\nSimulating processing...")
    time.sleep(0.1)  # 100ms delay
    for i in range(10):
        analyzer.process(b'y' * 512)
        time.sleep(0.016)
    
    analyzer.print_summary()