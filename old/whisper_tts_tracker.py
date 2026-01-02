#!/usr/bin/env python3
"""
Whisper TTS Tracker - Captures and transcribes TTS audio output in real-time
Uses whisper_streaming to determine what was actually spoken vs generated
"""

import asyncio
import numpy as np
import threading
import time
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass
import queue
import sys
import os

# Add whisper_streaming to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'whisper_streaming'))

from whisper_streaming.whisper_online import FasterWhisperASR, OnlineASRProcessor

@dataclass
class SpokenContent:
    """Represents content that was actually spoken by TTS."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0

class WhisperTTSTracker:
    """Tracks TTS audio output using real-time Whisper streaming."""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        """
        Initialize the Whisper TTS tracker.
        
        Args:
            sample_rate: Audio sample rate (16kHz required for Whisper)
            chunk_size: Audio chunk size for processing
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.is_tracking = False
        self.is_stopping = False
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.processing_thread: Optional[threading.Thread] = None
        
        # Whisper components
        self.asr: Optional[FasterWhisperASR] = None
        self.online_processor: Optional[OnlineASRProcessor] = None
        
        # Tracking state
        self.spoken_content: List[SpokenContent] = []
        self.current_session_start: Optional[float] = None
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Callbacks
        self.on_spoken_content: Optional[Callable[[SpokenContent], None]] = None
        self.on_transcription_complete: Optional[Callable[[List[SpokenContent]], None]] = None
        
        self._initialize_whisper()
    
    def _initialize_whisper(self):
        """Initialize Whisper ASR and online processor."""
        try:
            # Create a CPU-compatible ASR class
            class CPUFasterWhisperASR(FasterWhisperASR):
                def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
                    from faster_whisper import WhisperModel
                    
                    if model_dir is not None:
                        model_size_or_path = model_dir
                    elif modelsize is not None:
                        model_size_or_path = modelsize
                    else:
                        raise ValueError("modelsize or model_dir parameter must be set")
                    
                    # Use CPU with int8 for macOS compatibility
                    model = WhisperModel(
                        model_size_or_path, 
                        device="cpu", 
                        compute_type="int8",
                        download_root=cache_dir
                    )
                    return model
            
            # Initialize CPU-compatible faster-whisper ASR
            self.asr = CPUFasterWhisperASR("en", "base")  # Using base model for speed
            
            # Initialize online processor
            self.online_processor = OnlineASRProcessor(
                self.asr, 
                buffer_trimming=("segment", 10),  # Trim segments longer than 10s
                logfile=sys.stderr
            )
            
            print("✅ Whisper TTS Tracker initialized (CPU mode)")
            
        except Exception as e:
            print(f"❌ Failed to initialize Whisper: {e}")
            self.asr = None
            self.online_processor = None
    
    def start_tracking(self, session_id: Optional[str] = None) -> bool:
        """
        Start tracking TTS audio output.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            True if tracking started successfully
        """
        if not self.asr or not self.online_processor:
            print("❌ Cannot start tracking - Whisper not initialized")
            return False
            
        if self.is_tracking:
            print("⚠️ Already tracking")
            return True
            
        self.is_tracking = True
        self.is_stopping = False
        self.spoken_content.clear()
        self.current_session_start = time.time()
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Reset online processor
        self.online_processor.init()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._audio_processing_worker,
            daemon=True
        )
        self.processing_thread.start()
        
        print(f"🎙️ Started tracking TTS audio output")
        return True
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """
        Add audio chunk for processing.
        
        Args:
            audio_data: Audio data as numpy array (mono, float32)
        """
        if not self.is_tracking:
            return
            
        # Ensure correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Resample if needed (basic resampling)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)  # Convert to mono
            
        # Add to queue for processing
        try:
            self.audio_queue.put(audio_data, block=False)
        except queue.Full:
            print("⚠️ Audio queue full, dropping chunk")
    
    def _audio_processing_worker(self):
        """Background thread for processing audio through Whisper."""
        print("🔄 Audio processing worker started")
        
        while self.is_tracking and not self.is_stopping:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Add to Whisper processor
                self.online_processor.insert_audio_chunk(audio_chunk)
                
                # Process and check for results
                result = self.online_processor.process_iter()
                
                if result and result[0] is not None:
                    beg_timestamp, end_timestamp, text = result
                    
                    if text.strip():  # Only process non-empty text
                        spoken_content = SpokenContent(
                            text=text.strip(),
                            start_time=beg_timestamp,
                            end_time=end_timestamp,
                            confidence=1.0  # Whisper doesn't provide confidence directly
                        )
                        
                        self.spoken_content.append(spoken_content)
                        
                        # Call callback if available
                        if self.on_spoken_content:
                            try:
                                self.on_spoken_content(spoken_content)
                            except Exception as e:
                                print(f"Error in spoken content callback: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
                
        print("🛑 Audio processing worker stopped")
    
    def stop_tracking(self) -> List[SpokenContent]:
        """
        Stop tracking and return all spoken content.
        
        Returns:
            List of all spoken content from this session
        """
        if not self.is_tracking:
            return self.spoken_content.copy()
            
        self.is_stopping = True
        
        # Process final audio
        if self.online_processor:
            try:
                final_result = self.online_processor.finish()
                if final_result and final_result[0] is not None:
                    beg_timestamp, end_timestamp, text = final_result
                    if text.strip():
                        spoken_content = SpokenContent(
                            text=text.strip(),
                            start_time=beg_timestamp,
                            end_time=end_timestamp,
                            confidence=1.0
                        )
                        self.spoken_content.append(spoken_content)
            except Exception as e:
                print(f"Error processing final audio: {e}")
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        self.is_tracking = False
        
        # Call completion callback
        if self.on_transcription_complete:
            try:
                self.on_transcription_complete(self.spoken_content.copy())
            except Exception as e:
                print(f"Error in transcription complete callback: {e}")
        
        result = self.spoken_content.copy()
        print(f"🏁 Stopped tracking - captured {len(result)} spoken segments")
        
        return result
    
    def get_spoken_text(self) -> str:
        """
        Get all spoken text as a single string.
        
        Returns:
            Combined spoken text
        """
        return " ".join(content.text for content in self.spoken_content)
    
    def get_spoken_duration(self) -> float:
        """
        Get total duration of spoken content.
        
        Returns:
            Duration in seconds
        """
        if not self.spoken_content:
            return 0.0
        
        return max(content.end_time for content in self.spoken_content) - \
               min(content.start_time for content in self.spoken_content)
    
    def clear_session(self):
        """Clear current session data."""
        self.spoken_content.clear()
        self.current_session_start = None
        if self.online_processor:
            self.online_processor.init()


# Test function
async def test_whisper_tts_tracker():
    """Test the Whisper TTS tracker with sample audio."""
    tracker = WhisperTTSTracker()
    
    def on_spoken_content(content: SpokenContent):
        print(f"🎵 Spoken: '{content.text}' ({content.start_time:.2f}s - {content.end_time:.2f}s)")
    
    def on_complete(content_list: List[SpokenContent]):
        print(f"🏁 Session complete: {len(content_list)} segments")
        full_text = " ".join(c.text for c in content_list)
        print(f"Full spoken text: '{full_text}'")
    
    tracker.on_spoken_content = on_spoken_content
    tracker.on_transcription_complete = on_complete
    
    # Start tracking
    if tracker.start_tracking():
        print("Testing with silence (no audio)...")
        await asyncio.sleep(2)
        
        # Simulate audio chunks (silence for testing)
        for i in range(10):
            silence = np.zeros(1024, dtype=np.float32)
            tracker.add_audio_chunk(silence)
            await asyncio.sleep(0.1)
        
        # Stop tracking
        results = tracker.stop_tracking()
        print(f"Test complete - captured {len(results)} segments")
    else:
        print("Failed to start tracker")


if __name__ == "__main__":
    asyncio.run(test_whisper_tts_tracker()) 