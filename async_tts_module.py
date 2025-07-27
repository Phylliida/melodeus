#!/usr/bin/env python3
"""
Async TTS Module for ElevenLabs WebSocket Streaming
Provides interruptible text-to-speech with real-time audio playback.
Now includes Whisper-based tracking of actual spoken content.
"""

import asyncio
import websockets
import json
import base64
import pyaudio
import threading
import queue
import time
import numpy as np
import re
from typing import Optional, AsyncGenerator, Dict, Any, List, Tuple
from dataclasses import dataclass
from scipy import signal

# Import Whisper TTS Tracker
try:
    from whisper_tts_tracker import WhisperTTSTracker, SpokenContent
    WHISPER_AVAILABLE = True
except ImportError:
    print("⚠️ Whisper TTS Tracker not available - spoken content tracking disabled")
    WHISPER_AVAILABLE = False
    SpokenContent = None

@dataclass
class TTSConfig:
    """Configuration for TTS settings."""
    api_key: str
    voice_id: str = "T2KZm9rWPG5TgXTyjt7E"  # Catalyst voice
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "pcm_22050"
    sample_rate: int = 22050
    speed: float = 1.0
    stability: float = 0.5
    similarity_boost: float = 0.8
    chunk_size: int = 1024
    buffer_size: int = 2048
    # Multi-voice support
    emotive_voice_id: Optional[str] = None  # Voice for text in asterisks (*emotive text*)
    emotive_speed: float = 1.0
    emotive_stability: float = 0.5
    emotive_similarity_boost: float = 0.8

class AsyncTTSStreamer:
    """Async TTS Streamer with interruption capabilities and spoken content tracking."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.is_streaming = False
        self.playback_thread = None
        self.websocket = None
        self.audio_task = None
        
        # Audio setup
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Control flags
        self._stop_requested = False
        self._interrupted = False
        
        # Whisper tracking
        self.whisper_tracker: Optional[WhisperTTSTracker] = None
        self.track_spoken_content = WHISPER_AVAILABLE
        self.current_spoken_content: List[SpokenContent] = []
        self.generated_text = ""  # Text that was sent to TTS
        
        # Initialize Whisper tracker if available
        if WHISPER_AVAILABLE:
            try:
                self.whisper_tracker = WhisperTTSTracker(sample_rate=16000)
                print("✅ Whisper TTS tracking enabled")
            except Exception as e:
                print(f"⚠️ Failed to initialize Whisper tracker: {e}")
                self.whisper_tracker = None
                self.track_spoken_content = False
        
    async def speak_text(self, text: str) -> bool:
        """
        Speak the given text with interruption support.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
        if self.is_streaming:
            await self.stop()
            
        try:
            self._stop_requested = False
            self._interrupted = False
            
            # Store generated text and start Whisper tracking
            self.generated_text = text
            self.current_spoken_content.clear()
            
            if self.whisper_tracker and self.track_spoken_content:
                self.whisper_tracker.start_tracking(f"tts_session_{int(time.time())}")
            
            await self._start_streaming(text)
            
            # If we reached here without stop being requested, it completed successfully
            completed_successfully = not self._stop_requested
            self._interrupted = self._stop_requested  # Set interrupted flag based on actual completion
            
            return completed_successfully
            
        except Exception as e:
            print(f"TTS error: {e}")
            return False
        finally:
            await self._cleanup_stream()
    
    def _parse_emotive_text(self, text: str) -> List[Tuple[str, bool]]:
        """
        Parse text to separate regular text from emotive text (in asterisks).
        
        Args:
            text: Input text that may contain *emotive* parts
            
        Returns:
            List of (text_chunk, is_emotive) tuples
        """
        if not self.config.emotive_voice_id:
            # No emotive voice configured, return all as regular text
            return [(text, False)]
        
        parts = []
        current_pos = 0
        
        # Find all *text* patterns
        for match in re.finditer(r'\*([^*]+)\*', text):
            # Add regular text before the emotive part
            if match.start() > current_pos:
                regular_text = text[current_pos:match.start()]
                if regular_text.strip():
                    parts.append((regular_text, False))
            
            # Add emotive text (content inside asterisks)
            emotive_text = match.group(1)
            if emotive_text.strip():
                parts.append((emotive_text, True))
            
            current_pos = match.end()
        
        # Add remaining regular text
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            if remaining_text.strip():
                parts.append((remaining_text, False))
        
        return parts if parts else [(text, False)]

    async def speak_stream_multi_voice(self, text_generator: AsyncGenerator[str, None]) -> bool:
        """
        Speak streaming text with multi-voice support for emotive expressions.
        Detects *emotive text* and uses different voice for those parts.
        If no emotive voice is configured, uses main voice for everything.
        
        Args:
            text_generator: Async generator yielding text chunks
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
            
        if self.is_streaming:
            await self.stop()
            
        try:
            self._stop_requested = False
            self._interrupted = False
            
            # Store generated text and start Whisper tracking
            self.generated_text = ""
            self.current_spoken_content.clear()
            
            if self.track_spoken_content and self.whisper_tracker:
                self.whisper_tracker.start_tracking()
            
            # Ensure audio playback is ready
            if not self.is_playing:
                self._start_audio_playback()
            self.is_streaming = True
            
            text_buffer = ""
            
            async for chunk in text_generator:
                if self._stop_requested:
                    self._interrupted = True
                    break
                    
                text_buffer += chunk
                self.generated_text += chunk
                
                # Process complete sentences/phrases for voice switching
                sentences = self._split_into_sentences(text_buffer)
                
                # Keep the last incomplete sentence in buffer
                if sentences and not text_buffer.rstrip().endswith(('.', '!', '?')):
                    complete_sentences = sentences[:-1]
                    text_buffer = sentences[-1]
                else:
                    complete_sentences = sentences
                    text_buffer = ""
                
                # Process each complete sentence with appropriate voice
                for sentence in complete_sentences:
                    if sentence.strip():
                        await self._speak_sentence_with_voice_switching(sentence)
                        
                        if self._stop_requested:
                            self._interrupted = True
                            break
            
            # Process any remaining text in buffer
            if text_buffer.strip() and not self._stop_requested:
                await self._speak_sentence_with_voice_switching(text_buffer)
            
            # Wait for all audio to finish playing
            if not self._stop_requested:
                print("⏳ Waiting for audio queue to drain...")
                await self._wait_for_audio_completion()
                print("✅ Audio queue drained")
            
            # Stop Whisper tracking
            if self.track_spoken_content and self.whisper_tracker:
                self.current_spoken_content = self.whisper_tracker.stop_tracking()
            
            return not self._interrupted
            
        except Exception as e:
            print(f"Multi-voice TTS error: {e}")
            self._interrupted = True
            # Stop Whisper tracking on error
            if self.track_spoken_content and self.whisper_tracker:
                try:
                    self.whisper_tracker.stop_tracking()
                except Exception:
                    pass
            return False
        finally:
            self.is_streaming = False
            await self._cleanup_multi_voice()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for voice processing."""
        # Simple sentence splitting - can be enhanced
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    async def _speak_sentence_with_voice_switching(self, sentence: str):
        """Speak a sentence with appropriate voice switching for emotive parts."""
        parts = self._parse_emotive_text(sentence)
        
        for text_part, is_emotive in parts:
            if text_part.strip():
                voice_id = self.config.emotive_voice_id if is_emotive else self.config.voice_id
                voice_settings = self._get_voice_settings(is_emotive)
                
                # Create a separate connection for this voice part
                await self._speak_text_part(text_part, voice_id, voice_settings)
                
                if self._stop_requested:
                    break

    def _get_voice_settings(self, is_emotive: bool) -> Dict[str, float]:
        """Get voice settings for regular or emotive speech."""
        if is_emotive:
            return {
                "speed": self.config.emotive_speed,
                "stability": self.config.emotive_stability,
                "similarity_boost": self.config.emotive_similarity_boost
            }
        else:
            return {
                "speed": self.config.speed,
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost
            }

    async def _speak_text_part(self, text: str, voice_id: str, voice_settings: Dict[str, float]):
        """Speak a single text part with specified voice and settings."""
        try:
            # Check if we should stop before starting
            if self._stop_requested:
                return
                
            # Create websocket connection for this voice
            uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
            params = f"?model_id={self.config.model_id}&output_format={self.config.output_format}"
            
            async with websockets.connect(uri + params) as websocket:
                # Check interruption again
                if self._stop_requested:
                    return
                    
                # Send initial configuration
                initial_message = {
                    "text": " ",
                    "voice_settings": voice_settings,
                    "xi_api_key": self.config.api_key
                }
                await websocket.send(json.dumps(initial_message))
                
                # Check interruption before sending text
                if self._stop_requested:
                    return
                    
                # Send the text
                text_message = {"text": text}
                await websocket.send(json.dumps(text_message))
                
                # Send completion signal
                if not self._stop_requested:
                    await websocket.send(json.dumps({"text": ""}))
                
                # Handle audio responses (this will check for interruption)
                await self._handle_websocket_audio(websocket)
                
        except Exception as e:
            if not self._stop_requested:  # Don't log errors if we were interrupted
                print(f"Error speaking text part '{text[:50]}...': {e}")

    async def _handle_websocket_audio(self, websocket):
        """Handle audio responses from a websocket connection."""
        try:
            async for message in websocket:
                if self._stop_requested:
                    break
                    
                data = json.loads(message)
                
                if data.get("audio"):
                    # Decode and queue audio
                    audio_data = base64.b64decode(data["audio"])
                    self.audio_queue.put(audio_data)
                    
                    # NOTE: For multi-voice, Whisper tracking is handled by the audio playback thread
                    # to avoid double-tracking from multiple websocket connections
                
                elif data.get("isFinal"):
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"WebSocket audio handling error: {e}")

    async def _wait_for_audio_completion(self):
        """Wait for the audio queue to drain completely in multi-voice mode."""
        try:
            # Wait for queue to drain (similar to regular streaming)
            max_wait_time = 15.0  # Maximum wait time in seconds
            wait_start = time.time()
            
            print(f"🔊 Waiting for audio queue (current size: {self.audio_queue.qsize()})")
            
            while not self.audio_queue.empty() and not self._stop_requested:
                if time.time() - wait_start > max_wait_time:
                    print("⏰ Audio queue drain timeout")
                    break
                    
                # Show progress occasionally
                if int(time.time() - wait_start) % 2 == 0:
                    print(f"🔊 Still waiting... queue size: {self.audio_queue.qsize()}")
                    
                await asyncio.sleep(0.1)
            
            # Additional small wait to ensure last audio chunk finishes playing
            if not self._stop_requested:
                print("🔊 Queue empty, waiting for final playback to complete...")
                await asyncio.sleep(1.0)  # Give more time for final audio to play
                
        except Exception as e:
            print(f"Audio completion wait error: {e}")

    async def _cleanup_multi_voice(self):
        """Clean up multi-voice streaming resources."""
        try:
            # Use the existing cleanup method for consistency
            await self._cleanup_stream()
                
        except Exception as e:
            print(f"Multi-voice cleanup error: {e}")

    async def speak_stream(self, text_generator: AsyncGenerator[str, None]) -> bool:
        """
        Speak streaming text with interruption support.
        
        Args:
            text_generator: Async generator yielding text chunks
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
        if self.is_streaming:
            await self.stop()
            
        try:
            self._stop_requested = False
            self._interrupted = False
            
            # Initialize for Whisper tracking
            self.generated_text = ""
            self.current_spoken_content.clear()
            
            if self.whisper_tracker and self.track_spoken_content:
                self.whisper_tracker.start_tracking(f"tts_stream_session_{int(time.time())}")
            
            await self._start_streaming_generator(text_generator)
            
            # If we reached here without stop being requested, it completed successfully
            completed_successfully = not self._stop_requested
            self._interrupted = self._stop_requested  # Set interrupted flag based on actual completion
            
            return completed_successfully
            
        except Exception as e:
            print(f"TTS streaming error: {e}")
            return False
        finally:
            await self._cleanup_stream()
    
    async def stop(self):
        """Stop current TTS playback immediately and wait for complete shutdown."""
        print("🛑 TTS stop requested")
        self._stop_requested = True
        self._interrupted = True
        
        # Clear audio queue immediately
        self._clear_audio_queue()
        
        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                print(f"Error closing TTS WebSocket: {e}")
            self.websocket = None
        
        # Cancel audio task and wait for it to finish
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
        
        # Stop audio playback and wait for thread to exit
        await self._stop_audio_playback_async()
        
        self.is_streaming = False
        print("✅ TTS stopped")
    
    def is_currently_playing(self) -> bool:
        """Check if TTS is currently playing (both streaming and audio output)."""
        return (self.is_streaming and 
                self.is_playing and 
                self.playback_thread and 
                self.playback_thread.is_alive() and
                not self._stop_requested)
    
    async def _start_streaming(self, text: str):
        """Start streaming a single text."""
        await self._setup_websocket()
        
        # Send text
        if not self._stop_requested:
            message = {
                "text": text,
                "try_trigger_generation": True
            }
            await self.websocket.send(json.dumps(message))
            print(f"🔊 TTS: {text}")
        
        # Signal completion
        if not self._stop_requested:
            await self.websocket.send(json.dumps({"text": ""}))
        
        # Wait for completion
        await self._wait_for_completion()
    
    async def _start_streaming_generator(self, text_generator: AsyncGenerator[str, None]):
        """Start streaming from a text generator."""
        await self._setup_websocket()
        
        text_buffer = ""
        
        async for text_chunk in text_generator:
            if self._stop_requested:
                break
                
            text_buffer += text_chunk
            # Accumulate for Whisper tracking
            self.generated_text += text_chunk
            
            # Send chunks at natural breaks
            if any(punct in text_buffer for punct in ['.', '!', '?', ',', ';']) or len(text_buffer) > 40:
                if not self._stop_requested:
                    message = {
                        "text": text_buffer,
                        "try_trigger_generation": True
                    }
                    await self.websocket.send(json.dumps(message))
                    print(f"🔊 TTS: {text_buffer.strip()}")
                    text_buffer = ""
        
        # Send remaining text
        if text_buffer.strip() and not self._stop_requested:
            message = {
                "text": text_buffer,
                "try_trigger_generation": True
            }
            await self.websocket.send(json.dumps(message))
            print(f"🔊 TTS Final: {text_buffer.strip()}")
        
        # Signal completion
        if not self._stop_requested:
            await self.websocket.send(json.dumps({"text": ""}))
        
        # Wait for completion
        await self._wait_for_completion()
    
    async def _setup_websocket(self):
        """Setup WebSocket connection and audio playback."""
        # Clear any leftover audio
        self._clear_audio_queue()
        
        # Start audio playback
        self._start_audio_playback()
        
        # Connect to WebSocket
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.config.voice_id}/stream-input"
        params = f"?model_id={self.config.model_id}&output_format={self.config.output_format}"
        
        self.websocket = await websockets.connect(uri + params)
        self.is_streaming = True
        
        # Send initial configuration
        initial_message = {
            "text": " ",
            "voice_settings": {
                "speed": self.config.speed,
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost
            },
            "xi_api_key": self.config.api_key
        }
        await self.websocket.send(json.dumps(initial_message))
        
        # Start audio response handler
        self.audio_task = asyncio.create_task(self._handle_audio_responses())
    
    async def _handle_audio_responses(self):
        """Handle incoming audio responses from ElevenLabs."""
        try:
            async for message in self.websocket:
                if self._stop_requested:
                    break
                
                if message is None:
                    continue
                
                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    continue
                
                if "audio" in data and data["audio"] and not self._stop_requested:
                    try:
                        audio_data = base64.b64decode(data["audio"])
                        if len(audio_data) > 0:
                            self.audio_queue.put(audio_data)
                    except Exception as e:
                        print(f"Failed to decode TTS audio: {e}")
                        continue
                
                if data.get("isFinal", False):
                    print("🏁 TTS generation completed")
                    break
                    
        except asyncio.CancelledError:
            print("🔇 TTS audio handler cancelled")
        except Exception as e:
            if not self._stop_requested:
                print(f"TTS audio handling error: {e}")
    
    async def _wait_for_completion(self):
        """Wait for audio processing to complete."""
        if self.audio_task and not self._stop_requested:
            try:
                await asyncio.wait_for(self.audio_task, timeout=15.0)
            except asyncio.TimeoutError:
                print("⏰ TTS audio task timed out")
                if self.audio_task:
                    self.audio_task.cancel()
            except asyncio.CancelledError:
                pass
        
        # Wait for queue to drain
        if not self._stop_requested:
            while not self.audio_queue.empty() and not self._stop_requested:
                await asyncio.sleep(0.1)
    
    def _start_audio_playback(self):
        """Start the audio playback thread."""
        if not self.is_playing:
            # Clean up any existing stream first
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            
            # Open new stream
            try:
                self.stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.config.sample_rate,
                    output=True,
                    frames_per_buffer=self.config.buffer_size
                )
                
                self.is_playing = True
                self.playback_thread = threading.Thread(target=self._audio_playback_worker)
                self.playback_thread.daemon = True
                self.playback_thread.start()
                
            except Exception as e:
                print(f"Failed to start audio playback: {e}")
                self.is_playing = False
    
    def _stop_audio_playback(self):
        """Stop the audio playback thread (sync version)."""
        if self.is_playing:
            self.is_playing = False
            
            if self.playback_thread:
                self.playback_thread.join(timeout=2.0)
                
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"Error stopping audio stream: {e}")
                self.stream = None
    
    async def _stop_audio_playback_async(self):
        """Stop the audio playback thread and wait for complete shutdown."""
        if not self.is_playing:
            return  # Already stopped
        
        print("🔄 Stopping audio playback...")
        self.is_playing = False
        
        # Wait for playback thread to exit in a non-blocking way
        if self.playback_thread and self.playback_thread.is_alive():
            # Use asyncio to wait for thread without blocking
            max_wait_time = 1.0  # Maximum 1 second wait
            start_time = asyncio.get_event_loop().time()
            
            while (self.playback_thread.is_alive() and 
                   (asyncio.get_event_loop().time() - start_time) < max_wait_time):
                await asyncio.sleep(0.01)  # Check every 10ms
            
            if self.playback_thread.is_alive():
                print("⚠️ Audio thread didn't exit cleanly, forcing...")
                # Thread should exit on its own when is_playing = False
        
        # Clean up audio stream
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                print("🔇 Audio stream closed")
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
            self.stream = None
        
        self.playback_thread = None
        print("✅ Audio playback fully stopped")
    
    def _audio_playback_worker(self):
        """Worker thread for audio playback with responsive stop handling and audio capture."""
        print("🎵 Audio playback worker started")
        
        while self.is_playing and not self._stop_requested:
            try:
                # Use shorter timeout for more responsive stopping
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Double-check stop status before writing
                if audio_chunk and self.stream and not self._stop_requested and self.is_playing:
                    try:
                        # Play the audio
                        self.stream.write(audio_chunk)
                        
                        # Capture audio for Whisper tracking
                        if self.whisper_tracker and self.track_spoken_content:
                            try:
                                # Convert bytes to numpy array for Whisper
                                # ElevenLabs returns PCM 22050Hz, Whisper needs 16kHz
                                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                                
                                # Proper resampling from 22050Hz to 16000Hz using scipy
                                if len(audio_np) > 0:
                                    # Use scipy's high-quality resampling with anti-aliasing
                                    original_rate = self.config.sample_rate  # 22050
                                    target_rate = 16000
                                    
                                    # Calculate target number of samples
                                    num_samples = int(len(audio_np) * target_rate / original_rate)
                                    if num_samples > 0:
                                        # Use scipy.signal.resample for high-quality resampling
                                        audio_16k = signal.resample(audio_np, num_samples)
                                        self.whisper_tracker.add_audio_chunk(audio_16k.astype(np.float32))
                            except Exception as e:
                                print(f"Whisper audio capture error: {e}")
                        
                        self.audio_queue.task_done()
                    except Exception as e:
                        print(f"Audio write error: {e}")
                        self.audio_queue.task_done()
                        break
                elif audio_chunk:
                    # If stopped, mark task as done but don't play
                    self.audio_queue.task_done()
                    
            except queue.Empty:
                # Check stop condition more frequently during silence
                if self._stop_requested or not self.is_playing:
                    break
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
                break
        
        print("🔇 Audio playback worker exited")
    
    def _clear_audio_queue(self):
        """Clear the audio queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
    
    async def _cleanup_stream(self):
        """Clean up streaming resources."""
        self.is_streaming = False
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
        
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
            self.audio_task = None
        
        self._stop_audio_playback()
        
        # Stop Whisper tracking and collect spoken content
        if self.whisper_tracker and self.track_spoken_content:
            try:
                self.current_spoken_content = self.whisper_tracker.stop_tracking()
                print(f"🎙️ Captured {len(self.current_spoken_content)} spoken segments")
            except Exception as e:
                print(f"Error stopping Whisper tracking: {e}")
    
    def get_spoken_content(self) -> List[SpokenContent]:
        """
        Get the actual spoken content from the last TTS session.
        
        Returns:
            List of SpokenContent objects representing what was actually spoken
        """
        return self.current_spoken_content.copy()
    
    def get_spoken_text(self) -> str:
        """
        Get the actual spoken text as a single string.
        
        Returns:
            Combined text of what was actually spoken
        """
        return " ".join(content.text for content in self.current_spoken_content)
    
    def get_spoken_text_heuristic(self) -> str:
        """
        Get spoken text using character-count heuristic from original generated text.
        Uses Whisper's character count but preserves exact LLM vocabulary/formatting.
        Only applies heuristic when there was an interruption - otherwise returns full text.
        
        Returns:
            Portion of original generated text that was likely spoken
        """
        if not self.generated_text:
            return ""
        
        # If TTS was interrupted, use Whisper character count heuristic
        if self._interrupted and self.current_spoken_content:
            # Get total character count from Whisper
            whisper_char_count = sum(len(content.text) for content in self.current_spoken_content)
            
            if whisper_char_count == 0:
                return ""
            
            # Find this position in original text
            target_position = min(whisper_char_count, len(self.generated_text))
            
            # Round up to nearest complete word boundary
            if target_position >= len(self.generated_text):
                # If Whisper captured more than generated (shouldn't happen), return full text
                return self.generated_text
            
            # Find the end of the word at target position
            word_end_position = target_position
            
            # If we're in the middle of a word, find the end
            while (word_end_position < len(self.generated_text) and 
                   self.generated_text[word_end_position] not in [' ', '.', ',', '!', '?', ';', ':', '\n']):
                word_end_position += 1
            
            return self.generated_text[:word_end_position].strip()
        
        else:
            # TTS completed successfully - return full generated text
            return self.generated_text
    
    def get_generated_vs_spoken(self) -> Dict[str, str]:
        """
        Compare generated text vs actually spoken text.
        
        Returns:
            Dictionary with 'generated', 'spoken_whisper', and 'spoken_heuristic' keys
        """
        return {
            "generated": self.generated_text,
            "spoken_whisper": self.get_spoken_text(),
            "spoken_heuristic": self.get_spoken_text_heuristic()
        }
    
    def was_fully_spoken(self) -> bool:
        """
        Check if the generated text was fully spoken (not interrupted).
        Uses heuristic approach based on character count.
        
        Returns:
            True if likely fully spoken, False if interrupted
        """
        spoken_heuristic = self.get_spoken_text_heuristic()
        if not self.generated_text or not spoken_heuristic:
            return False
        
        # If heuristic captured at least 90% of generated text, consider it fully spoken
        return len(spoken_heuristic) >= (len(self.generated_text) * 0.9)

    async def cleanup(self):
        """Clean up all resources."""
        await self.stop()
        
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'p') and self.p:
            try:
                self.p.terminate()
            except Exception:
                pass

# Convenience function for quick usage
async def speak_text(text: str, api_key: str, voice_id: str = "T2KZm9rWPG5TgXTyjt7E") -> bool:
    """
    Convenience function to speak text with default settings.
    
    Args:
        text: Text to speak
        api_key: ElevenLabs API key
        voice_id: Voice ID to use
        
    Returns:
        bool: True if completed successfully, False if interrupted
    """
    config = TTSConfig(api_key=api_key, voice_id=voice_id)
    tts = AsyncTTSStreamer(config)
    
    try:
        result = await tts.speak_text(text)
        return result
    finally:
        await tts.cleanup()

# Example usage
async def main():
    """Example usage of the TTS module."""
    api_key = input("Enter your ElevenLabs API key: ").strip()
    
    config = TTSConfig(api_key=api_key)
    tts = AsyncTTSStreamer(config)
    
    try:
        print("🎤 Starting TTS test...")
        
        # Test single text
        result = await tts.speak_text("Hello! This is a test of the async TTS module.")
        print(f"✅ Completed: {result}")
        
        # Test interruption
        print("\n🛑 Testing interruption...")
        task = asyncio.create_task(
            tts.speak_text("This is a longer message that we will interrupt before it finishes playing completely.")
        )
        
        # Interrupt after 2 seconds
        await asyncio.sleep(2.0)
        await tts.stop()
        
        result = await task
        print(f"✅ Interrupted result: {result}")
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        await tts.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 