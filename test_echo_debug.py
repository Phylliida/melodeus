#!/usr/bin/env python3
"""
Debug tool for echo cancellation timing and data flow
"""

import time
import numpy as np
import threading
from collections import deque
from datetime import datetime

class EchoDebugger:
    def __init__(self):
        self.reference_log = deque(maxlen=1000)
        self.process_log = deque(maxlen=1000)
        self.lock = threading.Lock()
        
    def log_reference(self, audio_data, source="TTS"):
        """Log reference audio data"""
        with self.lock:
            entry = {
                'timestamp': time.time(),
                'source': source,
                'data_type': type(audio_data).__name__,
                'data_len': len(audio_data) if hasattr(audio_data, '__len__') else 'N/A',
                'sample': audio_data[:20] if hasattr(audio_data, '__getitem__') else str(audio_data)[:50]
            }
            self.reference_log.append(entry)
            
    def log_process(self, near_data, far_data, result):
        """Log echo cancellation processing"""
        with self.lock:
            entry = {
                'timestamp': time.time(),
                'near_type': type(near_data).__name__,
                'far_type': type(far_data).__name__,
                'result_type': type(result).__name__ if result is not None else 'None',
                'near_len': len(near_data) if hasattr(near_data, '__len__') else 'N/A',
                'far_len': len(far_data) if hasattr(far_data, '__len__') else 'N/A'
            }
            self.process_log.append(entry)
            
    def print_summary(self):
        """Print debug summary"""
        with self.lock:
            print("\n" + "="*60)
            print("ECHO CANCELLATION DEBUG SUMMARY")
            print("="*60)
            
            # Reference audio log
            print("\n📥 REFERENCE AUDIO (Last 5 entries):")
            for entry in list(self.reference_log)[-5:]:
                dt = datetime.fromtimestamp(entry['timestamp']).strftime('%H:%M:%S.%f')[:-3]
                print(f"  [{dt}] {entry['source']}: {entry['data_type']} len={entry['data_len']}")
                print(f"    Sample: {entry['sample']}")
                
            # Process log
            print("\n🔧 PROCESSING (Last 5 entries):")
            for entry in list(self.process_log)[-5:]:
                dt = datetime.fromtimestamp(entry['timestamp']).strftime('%H:%M:%S.%f')[:-3]
                print(f"  [{dt}] Near: {entry['near_type']} ({entry['near_len']}) "
                      f"Far: {entry['far_type']} ({entry['far_len']}) "
                      f"Result: {entry['result_type']}")
                      
            # Timing analysis
            if len(self.reference_log) > 1 and len(self.process_log) > 1:
                print("\n⏱️  TIMING ANALYSIS:")
                ref_times = [e['timestamp'] for e in self.reference_log]
                proc_times = [e['timestamp'] for e in self.process_log]
                
                # Average time between reference audio
                ref_deltas = [ref_times[i+1] - ref_times[i] for i in range(len(ref_times)-1)]
                if ref_deltas:
                    avg_ref_delta = sum(ref_deltas) / len(ref_deltas)
                    print(f"  Avg time between reference audio: {avg_ref_delta*1000:.1f}ms")
                
                # Check delay between reference and processing
                if ref_times and proc_times:
                    latest_ref = ref_times[-1]
                    latest_proc = proc_times[-1]
                    delay = latest_proc - latest_ref
                    print(f"  Delay between last reference and process: {delay*1000:.1f}ms")
                    
            print("="*60 + "\n")

# Global debugger instance
echo_debugger = EchoDebugger()

# Test the data flow
if __name__ == "__main__":
    print("🔍 Testing Echo Cancellation Data Flow")
    print("=====================================")
    
    # Test 1: Check what type of data the echo canceller expects
    try:
        from speexdsp import EchoCanceller
        ec = EchoCanceller.create(256, 2048, 16000)
        
        # Test with different data types
        test_data = {
            'bytes': b'\x00' * 512,
            'numpy': np.zeros(256, dtype=np.int16),
            'string': 'test' * 128,
            'list': [0] * 256
        }
        
        print("\n📊 Testing data types with EchoCanceller:")
        for name, data in test_data.items():
            try:
                if name == 'numpy':
                    result = ec.process(data, data)
                    print(f"  ✅ {name}: Success!")
                else:
                    # Try converting to numpy first
                    if name == 'bytes':
                        np_data = np.frombuffer(data, dtype=np.int16)
                    elif name == 'list':
                        np_data = np.array(data, dtype=np.int16)
                    else:
                        np_data = data  # This will fail
                    
                    result = ec.process(np_data, np_data)
                    print(f"  ✅ {name} (converted): Success!")
            except Exception as e:
                print(f"  ❌ {name}: {type(e).__name__}: {str(e)[:60]}")
                
    except ImportError:
        print("❌ speexdsp not available")
    
    # Test 2: Check the actual data flow in our system
    print("\n📊 Checking our implementation:")
    
    # Check what type of data is being passed
    test_audio = b'\x00\x01' * 256  # 512 bytes = 256 int16 samples
    
    print(f"  Input type: {type(test_audio).__name__}")
    print(f"  Input length: {len(test_audio)} bytes")
    
    # Convert to numpy
    np_audio = np.frombuffer(test_audio, dtype=np.int16)
    print(f"  Numpy type: {np_audio.dtype}")
    print(f"  Numpy shape: {np_audio.shape}")
    print(f"  Numpy length: {len(np_audio)} samples")