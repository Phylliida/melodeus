"""
Practical echo cancellation solution that combines multiple techniques.
"""

import numpy as np
import collections
from typing import Optional, Tuple

class PracticalEchoProcessor:
    """
    Combines multiple strategies for practical echo reduction:
    1. Speex AEC with fixed delay for predictable sources (thinking sound)
    2. Frequency-domain noise gate for unpredictable TTS
    3. Voice activity detection to reduce false positives
    """
    
    def __init__(self, frame_size: int, filter_length: int, sample_rate: int):
        # Import here to avoid issues if not installed
        try:
            import speexdsp
            self.speex_available = True
            self.echo_canceller = speexdsp.EchoCanceller(
                frame_size=frame_size,
                filter_length=filter_length
            )
        except ImportError:
            self.speex_available = False
            self.echo_canceller = None
            
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        
        # Reference signal tracking
        self.is_playing_tts = False
        self.is_playing_thinking = False
        self.reference_energy = 0.0
        
        # Noise gate parameters
        self.gate_threshold = 0.02  # Adjustable
        self.gate_ratio = 0.1  # How much to reduce
        
        # Voice activity detection
        self.vad_threshold = 0.1
        self.speech_frames = collections.deque(maxlen=10)
        
    def set_playback_state(self, tts_active: bool, thinking_active: bool):
        """Update what's currently playing."""
        self.is_playing_tts = tts_active
        self.is_playing_thinking = thinking_active
        
    def add_reference_audio(self, audio_data: bytes, source: str = "tts"):
        """Add reference audio with source information."""
        if self.speex_available and self.echo_canceller and source == "thinking":
            # Only use Speex for predictable thinking sounds
            self.echo_canceller.add_reference_audio(audio_data)
            
        # Track reference signal energy
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        self.reference_energy = np.sqrt(np.mean(audio_array ** 2)) / 32768.0
        
    def _detect_voice_activity(self, audio_array: np.ndarray) -> bool:
        """Simple energy-based voice activity detection."""
        energy = np.sqrt(np.mean(audio_array ** 2)) / 32768.0
        self.speech_frames.append(energy > self.vad_threshold)
        
        # Require consistent speech over multiple frames
        return sum(self.speech_frames) >= 3
        
    def _apply_noise_gate(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply frequency-domain noise gate when TTS is playing."""
        if not self.is_playing_tts or self.reference_energy < 0.01:
            return audio_array
            
        # Simple spectral subtraction
        fft = np.fft.rfft(audio_array)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Reduce magnitude based on reference energy
        reduction_factor = 1.0 - (self.reference_energy * 0.5)
        magnitude *= reduction_factor
        
        # Reconstruct signal
        fft_reduced = magnitude * np.exp(1j * phase)
        return np.fft.irfft(fft_reduced, n=len(audio_array)).astype(np.float32)
        
    def process(self, near_data: bytes) -> Tuple[bytes, bool]:
        """
        Process microphone input with hybrid approach.
        Returns: (processed_audio, is_speech_detected)
        """
        near_array = np.frombuffer(near_data, dtype=np.int16).astype(np.float32)
        
        # Step 1: Apply Speex AEC if available and thinking sound is playing
        if self.speex_available and self.echo_canceller and self.is_playing_thinking:
            # Speex expects bytes
            processed = self.echo_canceller.process(near_data, near_data)  # Using near as far for now
            processed_array = np.frombuffer(processed, dtype=np.int16).astype(np.float32)
        else:
            processed_array = near_array.copy()
            
        # Step 2: Apply noise gate if TTS is playing
        if self.is_playing_tts:
            processed_array = self._apply_noise_gate(processed_array)
            
        # Step 3: Detect voice activity
        is_speech = self._detect_voice_activity(processed_array)
        
        # Step 4: Additional suppression if no speech detected during playback
        if (self.is_playing_tts or self.is_playing_thinking) and not is_speech:
            processed_array *= 0.1  # Heavy suppression
            
        # Convert back to int16
        processed_array = np.clip(processed_array, -32768, 32767)
        return processed_array.astype(np.int16).tobytes(), is_speech
 