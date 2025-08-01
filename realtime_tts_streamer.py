"""
Modified AsyncTTSStreamer with real-time playback control for accurate AEC timing.
"""

import asyncio
import queue
import threading
import time
import pyaudio
import numpy as np
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass

@dataclass
class AudioChunk:
    data: bytes
    text: str
    chunk_index: int
    is_final: bool

class RealtimeTTSStreamer:
    """
    TTS streamer with minimal buffering and precise playback timing.
    """
    
    def __init__(
        self,
        model_id: str,
        api_key: str,
        voice_id: str = "alloy",
        sample_rate: int = 16000,
        output_device_index: Optional[int] = None,
        speed: float = 1.0,
        chunk_size: int = 256  # Small chunks for low latency
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self.output_device_index = output_device_index
        self.speed = speed
        self.chunk_size = chunk_size
        
        # Audio queue with limited size to prevent excessive buffering
        self.audio_queue = queue.Queue(maxsize=50)  # ~500ms at 10ms chunks
        
        # Echo cancellation callback
        self.echo_callback: Optional[Callable[[bytes], None]] = None
        
        # State tracking
        self.is_playing = False
        self.current_text = ""
        self.spoken_text = ""
        
        # PyAudio setup
        self.pa = pyaudio.PyAudio()
        self.stream = None
        
        # Control
        self.running = False
        self._stop_requested = False
        
    def set_echo_cancellation_callback(self, callback: Callable[[bytes], None]):
        """Set callback for echo cancellation."""
        self.echo_callback = callback
        
    def start(self):
        """Start the real-time playback stream."""
        if self.running:
            return
            
        self.running = True
        self._stop_requested = False
        
        # Open stream with callback
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.output_device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.stream.start_stream()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback - called when audio device needs data.
        This gives us precise timing for AEC.
        """
        if not self.running or self._stop_requested:
            return (b'\x00' * frame_count * 2, pyaudio.paComplete)
            
        try:
            # Try to get audio chunk
            chunk = self.audio_queue.get_nowait()
            
            # Update spoken text tracking
            if chunk.text:
                self.spoken_text += chunk.text
                
            # Send to AEC at the exact moment of playback
            if self.echo_callback and chunk.data:
                self.echo_callback(chunk.data)
                
            # Ensure we return exactly frame_count samples
            data = chunk.data
            needed_bytes = frame_count * 2
            
            if len(data) < needed_bytes:
                data += b'\x00' * (needed_bytes - len(data))
            elif len(data) > needed_bytes:
                # Put excess back in queue
                excess = data[needed_bytes:]
                remaining_chunk = AudioChunk(
                    data=excess,
                    text="",
                    chunk_index=chunk.chunk_index,
                    is_final=chunk.is_final
                )
                try:
                    self.audio_queue.put_nowait(remaining_chunk)
                except queue.Full:
                    pass  # Drop if queue is full
                data = data[:needed_bytes]
                
            self.is_playing = True
            return (data, pyaudio.paContinue)
            
        except queue.Empty:
            # No audio available
            self.is_playing = False
            return (b'\x00' * frame_count * 2, pyaudio.paContinue)
            
    async def stream_tts(self, text: str) -> None:
        """
        Stream TTS with minimal buffering.
        """
        self.current_text = text
        self.spoken_text = ""
        
        # Here you would call your TTS API (OpenAI, etc.)
        # For now, simulate with chunks
        
        # Split text into words for simulation
        words = text.split()
        
        for i, word in enumerate(words):
            if self._stop_requested:
                break
                
            # Simulate TTS API delay
            await asyncio.sleep(0.05)
            
            # Generate fake audio (replace with real TTS)
            duration = len(word) * 0.1  # 100ms per character
            samples = int(self.sample_rate * duration)
            
            # Create small chunks for low latency
            chunk_samples = self.chunk_size
            
            for j in range(0, samples, chunk_samples):
                if self._stop_requested:
                    break
                    
                # Generate chunk (replace with real audio)
                chunk_size = min(chunk_samples, samples - j)
                audio_data = np.zeros(chunk_size, dtype=np.int16)
                
                # Determine if this is the last chunk of this word
                is_last_chunk = (j + chunk_samples >= samples)
                
                chunk = AudioChunk(
                    data=audio_data.tobytes(),
                    text=word + " " if is_last_chunk else "",
                    chunk_index=i,
                    is_final=(i == len(words) - 1 and is_last_chunk)
                )
                
                # This will block if queue is full, providing natural backpressure
                try:
                    self.audio_queue.put(chunk, timeout=0.1)
                except queue.Full:
                    if self._stop_requested:
                        break
                    # Try again
                    await asyncio.sleep(0.01)
                    
    def stop(self):
        """Stop playback immediately."""
        self._stop_requested = True
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
    def get_queue_latency(self) -> float:
        """Get current queue latency in seconds."""
        return self.audio_queue.qsize() * (self.chunk_size / self.sample_rate)
        
    def is_currently_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self.is_playing and not self.audio_queue.empty()
        
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        self.pa.terminate()
        
    def get_spoken_text_heuristic(self) -> str:
        """Get the text that has been spoken so far."""
        return self.spoken_text.strip()
 