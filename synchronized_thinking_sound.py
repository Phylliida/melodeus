#!/usr/bin/env python3
"""
Synchronized thinking sound player that streams audio in real-time.
"""
import asyncio
import threading
import time
from typing import Optional, Callable

import numpy as np

from mel_aec_audio import ensure_stream_started, write_playback_pcm

class SynchronizedThinkingSoundPlayer:
    """Plays thinking sound with proper timing for echo cancellation."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.stream = None  # Retained for backwards compatibility with previous PyAudio usage
        self.is_playing = False
        self.play_thread = None
        self._stop_event = threading.Event()
        self._current_generation = 0
        self._lock = threading.Lock()
        
        # Echo cancellation callback
        self.echo_cancellation_callback = None
        
        # Frame size for echo cancellation (must match STT)
        self.frame_size = 256  # samples
        self.frame_duration = self.frame_size / self.sample_rate  # seconds
        
        # Generate thinking sound frames
        self._generate_thinking_frames()
        
    def set_echo_cancellation_callback(self, callback: Callable[[bytes], None]):
        """Set callback for echo cancellation."""
        self.echo_cancellation_callback = callback
        
    def _generate_thinking_frames(self):
        """Generate thinking sound as individual frames."""
        # Parameters for thinking sound
        pulse_duration = 0.15  # 150ms
        silence_duration = 0.35  # 350ms
        frequency = 440  # A4
        
        # Calculate frames needed
        pulse_frames = int(pulse_duration / self.frame_duration)
        silence_frames = int(silence_duration / self.frame_duration)
        
        # Generate pulse
        pulse_samples = pulse_frames * self.frame_size
        t = np.linspace(0, pulse_duration, pulse_samples)
        envelope = np.sin(np.pi * t / pulse_duration) ** 2
        sound = envelope * np.sin(2 * np.pi * frequency * t) * 0.1
        sound += envelope * np.sin(2 * np.pi * frequency/2 * t) * 0.05
        
        # Convert to int16 and split into frames
        pulse_int16 = (sound * 32767).astype(np.int16)
        pulse_frames_list = [
            pulse_int16[i*self.frame_size:(i+1)*self.frame_size]
            for i in range(pulse_frames)
        ]
        
        # Generate silence frames
        silence_frame = np.zeros(self.frame_size, dtype=np.int16)
        silence_frames_list = [silence_frame] * silence_frames
        
        # Combine into full cycle
        self.thinking_frames = pulse_frames_list + silence_frames_list
        self.total_frames = len(self.thinking_frames)
        
        print(f"🎵 Generated thinking sound: {self.total_frames} frames "
              f"({self.total_frames * self.frame_duration:.2f}s cycle)")
    
    def _play_worker(self):
        """Worker thread that plays thinking sound with proper timing."""
        try:
            ensure_stream_started()
            next_frame_time = time.perf_counter()
            frame_index = 0
            
            while not self._stop_event.is_set():
                current_time = time.perf_counter()
                
                # Check if it's time for the next frame
                if current_time >= next_frame_time:
                    # Get current frame
                    frame = self.thinking_frames[frame_index]
                    frame_bytes = frame.tobytes()
                    
                    try:
                        # Play the frame through mel-aec
                        write_playback_pcm(frame_bytes, self.sample_rate)
                        
                        # Send to echo cancellation at the same time
                        if self.echo_cancellation_callback:
                            try:
                                self.echo_cancellation_callback(frame_bytes)
                            except Exception as ec_error:
                                if frame_index == 0:  # Only log once per cycle
                                    print(f"⚠️ Thinking sound echo cancellation error: {ec_error}")
                    
                    except Exception as e:
                        if "Stream closed" not in str(e):
                            print(f"Error playing thinking sound: {e}")
                        break
                    
                    # Move to next frame
                    frame_index = (frame_index + 1) % self.total_frames
                    
                    # Schedule next frame
                    next_frame_time += self.frame_duration
                    
                    # If we're running behind, catch up without drifting indefinitely
                    if next_frame_time < current_time:
                        next_frame_time = current_time + self.frame_duration
                else:
                    # Sleep until next frame time
                    sleep_time = next_frame_time - current_time
                    if sleep_time > 0:
                        time.sleep(sleep_time * 0.9)  # Sleep 90% to avoid overshooting
                        
        except Exception as e:
            print(f"Error in thinking sound setup: {e}")
        finally:
            with self._lock:
                self.stream = None
                self.is_playing = False
    
    async def start(self, generation: int = 0):
        """Start playing the thinking sound."""
        with self._lock:
            if self.is_playing and self._current_generation == generation:
                return
                
            if self.is_playing:
                print(f"🎵 Thinking sound already playing, updated to generation {generation}")
                self._current_generation = generation
                return
                
            self._current_generation = generation
            self.is_playing = True
            self._stop_event.clear()
            
            # Start playback thread
            self.play_thread = threading.Thread(
                target=self._play_worker,
                daemon=True,
                name=f"ThinkingSound-{generation}"
            )
            self.play_thread.start()
            
            print(f"🎵 Started synchronized thinking sound for generation {generation}")
    
    async def stop(self, generation: Optional[int] = None):
        """Stop playing the thinking sound."""
        with self._lock:
            if not self.is_playing:
                return
                
            if generation is not None and generation != self._current_generation:
                print(f"🔇 Ignoring stop for old generation {generation} (current: {self._current_generation})")
                return
                
            self.is_playing = False
            self._stop_event.set()
            
            generation_stopped = self._current_generation
            
        # Wait for thread to finish
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=0.5)
            
        print(f"🔇 Stopped thinking sound for generation {generation_stopped}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.is_playing:
                self._stop_event.set()
                if self.play_thread:
                    self.play_thread.join(timeout=1.0)
        except Exception as e:
            print(f"Error during thinking sound cleanup: {e}")


# Test
if __name__ == "__main__":
    async def test():
        player = SynchronizedThinkingSoundPlayer()
        
        # Mock echo cancellation callback
        frame_count = 0
        def echo_callback(data):
            nonlocal frame_count
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  Echo cancellation received frame {frame_count}")
        
        player.set_echo_cancellation_callback(echo_callback)
        
        print("Starting synchronized thinking sound test...")
        await player.start(1)
        
        # Let it play for 3 seconds
        await asyncio.sleep(3)
        
        print("Stopping...")
        await player.stop(1)
        
        print(f"Total frames sent to echo cancellation: {frame_count}")
        player.cleanup()
    
    asyncio.run(test())
 
