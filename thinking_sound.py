"""Thinking sound player for voice conversation system."""
import asyncio
import threading
import numpy as np
import pyaudio
from typing import Optional
import time


class ThinkingSoundPlayer:
    """Plays a looping thinking sound until stopped."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_playing = False
        self.play_thread = None
        self._stop_event = threading.Event()
        self._current_generation = 0
        self._lock = threading.Lock()
        
        # Generate a soft, pleasant thinking sound
        self.thinking_sound = self._generate_thinking_sound()
        
    def _generate_thinking_sound(self) -> np.ndarray:
        """Generate a soft, rhythmic thinking sound."""
        duration = 0.15  # 150ms per pulse
        silence_duration = 0.35  # 350ms silence between pulses
        
        # Create time arrays
        t_sound = np.linspace(0, duration, int(self.sample_rate * duration))
        t_silence = np.zeros(int(self.sample_rate * silence_duration))
        
        # Generate a soft sine wave with envelope
        frequency = 440  # A4 note
        envelope = np.sin(np.pi * t_sound / duration) ** 2  # Smooth fade in/out
        sound = envelope * np.sin(2 * np.pi * frequency * t_sound) * 0.1  # Low volume
        
        # Add a subtle lower harmonic for warmth
        sound += envelope * np.sin(2 * np.pi * frequency/2 * t_sound) * 0.05
        
        # Combine sound and silence for one complete pulse
        pulse = np.concatenate([sound, t_silence])
        
        # Convert to int16
        pulse = (pulse * 32767).astype(np.int16)
        
        return pulse
    
    def _play_loop(self):
        """Worker thread that plays the thinking sound in a loop."""
        local_stream = None
        try:
            # Open new stream
            if self.p:  # Only if PyAudio is still valid
                local_stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    output=True,
                    frames_per_buffer=1024
                )
                # Store reference for external access
                self.stream = local_stream
            
            while not self._stop_event.is_set() and local_stream:
                # Play one pulse
                try:
                    local_stream.write(self.thinking_sound.tobytes())
                except Exception as e:
                    # Stream might have been closed
                    if "Stream closed" not in str(e):
                        print(f"Error playing thinking sound: {e}")
                    break
                
        except Exception as e:
            print(f"Error in thinking sound setup: {e}")
        finally:
            # Clean up the local stream
            if local_stream:
                try:
                    local_stream.stop_stream()
                    local_stream.close()
                except Exception:
                    pass  # Stream might already be closed
            # Clear the reference
            self.stream = None
    
    async def start(self, generation: int = None):
        """Start playing the thinking sound for a specific generation."""
        with self._lock:
            if generation is not None:
                self._current_generation = generation
            
            # Check if thread is alive, not just is_playing flag
            if self.play_thread and self.play_thread.is_alive():
                # Already playing - just update generation
                print(f"🎵 Thinking sound already playing, updated to generation {self._current_generation}")
                return
            
            # Clean up dead thread if exists
            if self.play_thread and not self.play_thread.is_alive():
                self.play_thread = None
                self.is_playing = False
            
            self.is_playing = True
            self._stop_event.clear()
            self.play_thread = threading.Thread(target=self._play_loop)
            self.play_thread.daemon = True
            self.play_thread.start()
            print(f"🎵 Started thinking sound for generation {self._current_generation}")
    
    async def stop(self, generation: int = None):
        """Stop playing the thinking sound only if it matches the generation."""
        with self._lock:
            if generation is not None and generation != self._current_generation:
                print(f"🎵 Ignoring stop for old generation {generation}, current is {self._current_generation}")
                return
                
            if not self.is_playing:
                return
            
            self.is_playing = False
            self._stop_event.set()
        
        # Wait for thread to finish (outside lock to avoid deadlock)
        if self.play_thread and self.play_thread.is_alive():
            await asyncio.get_event_loop().run_in_executor(
                None, self.play_thread.join, 1.0
            )
        
        print(f"🔇 Stopped thinking sound for generation {generation or self._current_generation}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Stop playback first
            if self.is_playing:
                self.is_playing = False
                self._stop_event.set()
                if self.play_thread and self.play_thread.is_alive():
                    self.play_thread.join(timeout=1.0)
            
            # Close stream if open
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            
            # Terminate PyAudio
            if self.p:
                try:
                    self.p.terminate()
                except Exception:
                    pass
                self.p = None
        except Exception as e:
            print(f"Error during thinking sound cleanup: {e}")


# Example usage
async def test_thinking_sound():
    """Test the thinking sound player."""
    player = ThinkingSoundPlayer()
    
    try:
        print("Starting thinking sound...")
        await player.start()
        
        # Play for 3 seconds
        await asyncio.sleep(3)
        
        print("Stopping thinking sound...")
        await player.stop()
        
    finally:
        player.cleanup()


if __name__ == "__main__":
    asyncio.run(test_thinking_sound())