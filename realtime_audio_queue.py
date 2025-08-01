"""
Real-time audio queue with precise playback timing control.
Keeps PyAudio buffer minimal and tracks exact playback timing.
"""

import queue
import threading
import time
import pyaudio
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

@dataclass
class AudioChunk:
    data: bytes
    timestamp: float
    duration: float

class RealtimeAudioQueue:
    """
    Maintains minimal PyAudio buffering for precise timing control.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 256,  # Very small chunks
        device_index: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        
        # Main queue for audio chunks
        self.audio_queue = queue.Queue(maxsize=1000)  # Our buffering
        
        # Callback for echo cancellation
        self.echo_callback: Optional[Callable[[bytes, float], None]] = None
        
        # Timing tracking
        self.playback_started = False
        self.last_chunk_time = 0.0
        
        # PyAudio setup
        self.pa = pyaudio.PyAudio()
        self.stream = None
        
        # Control
        self.running = False
        self.thread = None
        
    def set_echo_callback(self, callback: Callable[[bytes, float], None]):
        """Set callback that receives audio data with precise timing."""
        self.echo_callback = callback
        
    def start(self):
        """Start the real-time playback thread."""
        if self.running:
            return
            
        self.running = True
        
        # Open stream with MINIMAL buffering
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.stream.start_stream()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback - called when audio device needs data.
        This is called in real-time, so we know EXACTLY when audio plays.
        """
        if not self.running:
            return (b'\x00' * frame_count * 2, pyaudio.paComplete)
            
        try:
            # Get next chunk from our queue
            chunk = self.audio_queue.get_nowait()
            
            # This is the KEY moment - audio is about to play RIGHT NOW
            if self.echo_callback:
                # Send to AEC with current timestamp
                current_time = time.time()
                self.echo_callback(chunk.data, current_time)
            
            # Pad or trim to exact frame count
            data = chunk.data
            needed_bytes = frame_count * 2 * self.channels
            
            if len(data) < needed_bytes:
                # Pad with silence
                data += b'\x00' * (needed_bytes - len(data))
            elif len(data) > needed_bytes:
                # Trim excess
                data = data[:needed_bytes]
                
            return (data, pyaudio.paContinue)
            
        except queue.Empty:
            # No audio available - return silence
            silence = b'\x00' * frame_count * 2 * self.channels
            return (silence, pyaudio.paContinue)
            
    def add_audio(self, audio_data: bytes):
        """Add audio to the queue."""
        # Split into small chunks for fine-grained control
        chunk_bytes = self.chunk_size * 2 * self.channels
        
        for i in range(0, len(audio_data), chunk_bytes):
            chunk_data = audio_data[i:i + chunk_bytes]
            chunk = AudioChunk(
                data=chunk_data,
                timestamp=time.time(),
                duration=len(chunk_data) / (self.sample_rate * 2 * self.channels)
            )
            
            # This will block if queue is full, providing backpressure
            self.audio_queue.put(chunk)
            
    def get_latency(self) -> float:
        """Get current playback latency in seconds."""
        return self.audio_queue.qsize() * (self.chunk_size / self.sample_rate)
        
    def stop(self):
        """Stop playback."""
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
    def __del__(self):
        """Cleanup."""
        self.stop()
        if hasattr(self, 'pa'):
            self.pa.terminate()


# Test script
if __name__ == "__main__":
    import math
    
    print("Testing real-time audio queue...")
    
    # Create queue with very small chunks
    rt_queue = RealtimeAudioQueue(
        sample_rate=16000,
        chunk_size=160  # 10ms chunks
    )
    
    # Echo callback for testing
    def echo_callback(data: bytes, timestamp: float):
        latency = rt_queue.get_latency()
        print(f"Playing audio at {timestamp:.3f}, queue latency: {latency:.3f}s")
    
    rt_queue.set_echo_callback(echo_callback)
    rt_queue.start()
    
    # Generate test tone
    duration = 2.0
    frequency = 440
    t = np.linspace(0, duration, int(16000 * duration))
    tone = (np.sin(2 * np.pi * frequency * t) * 32767 * 0.3).astype(np.int16)
    
    print("Adding audio to queue...")
    rt_queue.add_audio(tone.tobytes())
    
    print(f"Initial latency: {rt_queue.get_latency():.3f}s")
    
    # Wait for playback
    time.sleep(duration + 0.5)
    
    rt_queue.stop()
    print("Done!")
 