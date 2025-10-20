#!/usr/bin/env python3
"""
Async STT Module for Deepgram Real-time Speech-to-Text
Provides callback-based speech recognition with speaker identification.
"""

import asyncio
import time
import json
import threading
import queue
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List, Tuple
from datetime import datetime
from enum import Enum

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

try:
    from mel_aec_audio import (
        ensure_stream_started,
        prepare_capture_chunk,
        shared_sample_rate,
    )
except ImportError:  # pragma: no cover - fallback when running from repo root
    import os
    import sys

    CURRENT_DIR = os.path.dirname(__file__)
    if CURRENT_DIR and CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from mel_aec_audio import (
        ensure_stream_started,
        prepare_capture_chunk,
        shared_sample_rate,
    )

try:
    from titanet_voice_fingerprinting import TitaNetVoiceFingerprinter, create_word_timing_from_deepgram_word
    VOICE_FINGERPRINTING_AVAILABLE = True
    print("✅ TitaNet voice fingerprinting available")
except ImportError:
    VOICE_FINGERPRINTING_AVAILABLE = False
    print("❌ TitaNet voice fingerprinting not available (requires nemo_toolkit)")
    print("💡 Install with: pip install 'nemo_toolkit[asr]'")

class STTEventType(Enum):
    """Types of STT events."""
    UTTERANCE_COMPLETE = "utterance_complete"
    INTERIM_RESULT = "interim_result"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    SPEAKER_CHANGE = "speaker_change"
    CONNECTION_OPENED = "connection_opened"
    CONNECTION_CLOSED = "connection_closed"
    ERROR = "error"

@dataclass
class STTResult:
    """Result from speech-to-text recognition."""
    text: str
    confidence: float
    is_final: bool
    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Optional[Dict[str, Any]] = None

@dataclass
class STTConfig:
    """Configuration for STT settings."""
    api_key: str
    model: str = "nova-3"
    language: str = "en-US"
    sample_rate: int = 16000
    chunk_size: int = 8000
    channels: int = 1
    smart_format: bool = True
    interim_results: bool = True
    punctuate: bool = True
    diarize: bool = True
    utterance_end_ms: int = 1000
    vad_events: bool = True
    # Audio input device
    input_device_name: Optional[str] = None  # None = default device, or specify device name
    # Speaker identification settings
    enable_speaker_id: bool = False
    speaker_profiles_path: Optional[str] = None
    # Custom vocabulary/keywords for better recognition
    keywords: Optional[List[Tuple[str, float]]] = None  # List of (word, weight) tuples
    # Debug settings
    debug_speaker_data: bool = False  # Enable detailed speaker/timing debug output

class AsyncSTTStreamer:
    """Async STT Streamer with callback support."""
    
    def __init__(self, config: STTConfig, speakers_config=None):
        self.config = config
        self.speakers_config = speakers_config
        self.is_listening = False
        self.connection = None
        self.event_loop = None
        self.connection_alive = False
        self.audio_stream = None
        self.audio_stop_event = None
        self._audio_callback_registered = False
        self._chunk_counter = 0
        self.stream_start_time = None
        self._last_prepare_error_log = 0.0
        self._last_fp_error_log = 0.0
        self._last_send_error_log = 0.0
        self._last_queue_full_log = 0.0

        self._send_queue = None  # type: Optional["queue.Queue[bytes]"]
        self._sender_thread: Optional[threading.Thread] = None
        self._sender_stop_event: Optional[threading.Event] = None
        
        # Audio setup (shared mel-aec duplex stream)
        self.stream_sample_rate = shared_sample_rate()
        
        # Deepgram client
        self.deepgram = DeepgramClient(config.api_key)
        
        # Callback registry
        self.callbacks: Dict[STTEventType, List[Callable]] = {
            event_type: [] for event_type in STTEventType
        }
        
        # State tracking
        self.current_speaker = None
        self.last_utterance_time = None
        self.session_speakers = {}  # Maps session speaker IDs to names
        self.is_paused = False  # For pause/resume functionality
        
        # TitaNet voice fingerprinting (optional)
        self.voice_fingerprinter = None
        if VOICE_FINGERPRINTING_AVAILABLE and speakers_config and config.enable_speaker_id:
            try:
                # Enable debug audio saving if debug_speaker_data is enabled
                debug_save_audio = getattr(config, 'debug_speaker_data', False)
                self.voice_fingerprinter = TitaNetVoiceFingerprinter(speakers_config, debug_save_audio=debug_save_audio)
                print(f"🤖 TitaNet voice fingerprinting enabled")
                if debug_save_audio:
                    print(f"🐛 Debug audio saving enabled - extracted segments will be saved to debug_audio_segments/")
            except Exception as e:
                print(f"⚠️  TitaNet voice fingerprinting failed to initialize: {e}")
        
        # Speaker identification (optional)
        self.speaker_identifier = None
        if config.enable_speaker_id:
            self._setup_speaker_identification()
        
        # Reconnection settings
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0  # seconds
        self.is_reconnecting = False
        
        # Keepalive settings
        self.keepalive_task = None
        self.keepalive_interval = 10.0  # seconds

        # Hold final transcripts until the corresponding UtteranceEnd arrives
        self._pending_final_results: Dict[int, STTResult] = {}
        
    
    def on(self, event_type: STTEventType, callback: Callable):
        """Register a callback for an event type."""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def off(self, event_type: STTEventType, callback: Callable):
        """Unregister a callback for an event type."""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)

    def _get_channel_index(self, result_obj: Any) -> int:
        """Extract a stable integer channel index from Deepgram callbacks."""
        if result_obj is None:
            return 0

        def _coerce(value: Any) -> Optional[int]:
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, (list, tuple)) and value:
                return _coerce(value[0])
            if isinstance(value, dict):
                # Deepgram sometimes nests channel information in dicts
                if "channel_index" in value:
                    return _coerce(value["channel_index"])
                if "channel" in value:
                    return _coerce(value["channel"])
            return None

        # Direct attribute on the result object
        channel_index = _coerce(getattr(result_obj, "channel_index", None))
        if channel_index is not None:
            return channel_index

        # Sometimes channel info lives under result.channel
        channel_attr = getattr(result_obj, "channel", None)
        channel_index = _coerce(getattr(channel_attr, "channel_index", None))
        if channel_index is not None:
            return channel_index

        # If channel_attr itself is a list/dict, attempt to coerce it
        channel_index = _coerce(channel_attr)
        if channel_index is not None:
            return channel_index

        return 0
    
    async def start_listening(self) -> bool:
        """Start listening for speech."""
        if self.is_listening:
            await self.stop_listening()
        
        # Store the current event loop for scheduling tasks from sync callbacks
        self.event_loop = asyncio.get_event_loop()
        
        try:
            # Setup microphone in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Debug: Check what sample rates are actually supported
            print(f"🔧 [AUDIO DEBUG] Requested sample rate: {self.config.sample_rate}Hz")
            print(f"🔧 [AUDIO DEBUG] Chunk size: {self.config.chunk_size} samples")
            
            # Ensure shared mel-aec stream is available
            try:
                ensure_stream_started()
            except Exception as e:
                await self._emit_event(STTEventType.ERROR, {
                    "error": f"Failed to start mel-aec audio stream: {e}"
                })
                return False
            
            if self.config.input_device_name:
                print(f"🎧 Preferred input device: {self.config.input_device_name}")
            
            # Log actual capture sample rate for diagnostics
            actual_rate = self.stream_sample_rate
            print(f"🔧 [AUDIO DEBUG] mel-aec capture rate: {actual_rate}Hz")
            if actual_rate != self.config.sample_rate:
                print(f"⚠️ [AUDIO DEBUG] Sample rate mismatch – "
                      f"capturing at {actual_rate}Hz and resampling to {self.config.sample_rate}Hz for Deepgram.")
            
            # Setup Deepgram connection
            self.connection = self.deepgram.listen.websocket.v("1")
            
            # Configure options with explicit audio format
            options_dict = {
                "model": self.config.model,
                "language": self.config.language,
                "encoding": "linear16",  # Critical: Explicit audio encoding
                "sample_rate": self.config.sample_rate,  # Explicit sample rate
                "channels": self.config.channels,  # Explicit channel count
                "smart_format": self.config.smart_format,
                "interim_results": self.config.interim_results,
                "punctuate": self.config.punctuate,
                "diarize": self.config.diarize,
                "utterance_end_ms": self.config.utterance_end_ms,
                "vad_events": self.config.vad_events
            }
            
            # Add keywords/keyterms based on model
            if self.config.keywords:
                if self.config.model == "nova-3":
                    # Nova-3 uses keyterm parameter with space-separated terms
                    # Extract just the words (ignore weights for nova-3)
                    keyterms = []
                    for word, weight in self.config.keywords:
                        # Keep the original formatting for proper nouns
                        keyterms.append(word)
                    
                    if keyterms:
                        # Join with spaces for nova-3
                        keyterm_string = " ".join(keyterms)
                        options_dict["keyterm"] = keyterm_string
                        print(f"🔤 Using keyterms (Nova-3): {keyterms[:5]}...")
                        print(f"   Full keyterm string: {keyterm_string[:100]}...")
                else:
                    # Other models use keywords with weights
                    sanitized_keywords = []
                    for word, weight in self.config.keywords:
                        # Replace spaces with underscores for older models
                        sanitized_word = word.replace(" ", "_").replace(",", "")
                        if sanitized_word:
                            sanitized_keywords.append(f"{sanitized_word}:{weight}")
                    
                    if sanitized_keywords:
                        keyword_string = ",".join(sanitized_keywords)
                        options_dict["keywords"] = keyword_string
                        print(f"🔤 Using keywords: {sanitized_keywords[:5]}...")
                        print(f"   Full keyword string: {keyword_string[:100]}...")
            
            options = LiveOptions(**options_dict)
            
            # Set up event handlers
            self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self.connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
            self.connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
            self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self._on_close)
            
            # Start connection with proper async handling
            connection_success = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.connection.start(options)
            )
            
            if connection_success:
                self.is_listening = True
                # Don't set connection_alive yet - wait for _on_open callback
                
                # Minimal wait for connection to stabilize
                await asyncio.sleep(0.01)
                
                # Verify connection is actually alive before starting audio
                if self.connection_alive:
                    # Register mel-aec capture callback for streaming audio
                    if not self._start_audio_stream():
                        await self._emit_event(STTEventType.ERROR, {
                            "error": "Failed to start mel-aec capture callback"
                        })
                        return False
                    
                    # Start keepalive task to prevent connection timeout
                    self.keepalive_task = asyncio.create_task(self._keepalive_loop())
                    
                    await self._emit_event(STTEventType.CONNECTION_OPENED, {
                        "message": "STT connection established"
                    })
                    
                    # Reset reconnection attempts on successful connection
                    self.reconnect_attempts = 0
                    
                    return True
                else:
                    await self._emit_event(STTEventType.ERROR, {
                        "error": "Connection opened but not confirmed alive"
                    })
                    return False
            else:
                await self._emit_event(STTEventType.ERROR, {
                    "error": "Failed to start Deepgram connection"
                })
                return False
                
        except Exception as e:
            await self._emit_event(STTEventType.ERROR, {
                "error": f"Failed to start listening: {e}"
            })
            return False
    
    async def pause(self):
        """Pause STT processing (keeps connection alive but stops sending audio)."""
        if not self.is_paused:
            self.is_paused = True
            print("⏸️ STT paused")
            
    async def resume(self):
        """Resume STT processing."""
        if self.is_paused:
            self.is_paused = False
            print("▶️ STT resumed")
    
    async def _handle_reconnection(self):
        """Handle automatic reconnection with exponential backoff."""
        self.is_reconnecting = True
        
        while self.reconnect_attempts < self.max_reconnect_attempts and self.is_listening:
            self.reconnect_attempts += 1
            wait_time = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))  # Exponential backoff
            
            print(f"🔄 Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {wait_time}s...")
            await asyncio.sleep(wait_time)
            
            try:
                # Clean up old connection
                if self.connection:
                    try:
                        self.connection.finish()
                    except:
                        pass
                
                # Stop audio capture callback to avoid sending stale audio
                self._stop_audio_stream()
                
                # Try to reconnect
                print("🔌 Attempting to reconnect to Deepgram...")
                success = await self._reconnect()
                
                if success:
                    print("✅ Successfully reconnected to Deepgram!")
                    self.is_reconnecting = False
                    return
                    
            except Exception as e:
                print(f"❌ Reconnection attempt {self.reconnect_attempts} failed: {e}")
        
        # Max attempts reached
        print(f"❌ Failed to reconnect after {self.max_reconnect_attempts} attempts")
        self.is_reconnecting = False
        self.is_listening = False
        
        await self._emit_event(STTEventType.ERROR, {
            "error": "Failed to reconnect to Deepgram after multiple attempts"
        })
    
    async def _reconnect(self):
        """Attempt to reconnect to Deepgram."""
        # Reset connection state
        self.connection = None
        self.connection_alive = False
        
        # Recreate connection
        self.connection = self.deepgram.listen.websocket.v("1")
        
        # Reapply all the same options
        options_dict = {
            "model": self.config.model,
            "language": self.config.language,
            "encoding": "linear16",
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "smart_format": self.config.smart_format,
            "interim_results": self.config.interim_results,
            "punctuate": self.config.punctuate,
            "diarize": self.config.diarize,
            "utterance_end_ms": self.config.utterance_end_ms,
            "vad_events": self.config.vad_events
        }
        
        # Re-add keywords/keyterms
        if self.config.keywords:
            if self.config.model == "nova-3":
                keyterms = [word for word, weight in self.config.keywords]
                if keyterms:
                    options_dict["keyterm"] = " ".join(keyterms)
            else:
                sanitized_keywords = []
                for word, weight in self.config.keywords:
                    sanitized_word = word.replace(" ", "_").replace(",", "")
                    if sanitized_word:
                        sanitized_keywords.append(f"{sanitized_word}:{weight}")
                if sanitized_keywords:
                    options_dict["keywords"] = ",".join(sanitized_keywords)
        
        options = LiveOptions(**options_dict)
        
        # Re-setup event handlers
        self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
        self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        self.connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
        self.connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
        self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
        self.connection.on(LiveTranscriptionEvents.Close, self._on_close)
        
        # Start connection
        loop = asyncio.get_event_loop()
        connection_success = await loop.run_in_executor(
            None, lambda: self.connection.start(options)
        )
        
        if connection_success:
            # Wait for connection to be confirmed
            await asyncio.sleep(0.1)
            
            if self.connection_alive:
                if not self._start_audio_stream():
                    return False
                
                # Restart keepalive
                self.keepalive_task = asyncio.create_task(self._keepalive_loop())
                
                return True
        
        return False
    
    async def stop_listening(self):
        """Stop listening for speech."""
        print("🛑 STT stop requested")
        self.is_listening = False
        self.connection_alive = False
        
        # Cancel keepalive task if it's running
        if self.keepalive_task and not self.keepalive_task.done():
            print("🔄 Cancelling keepalive task...")
            self.keepalive_task.cancel()
            try:
                await asyncio.wait_for(self.keepalive_task, timeout=1.0)
            except asyncio.TimeoutError:
                print("⚠️ Keepalive task took too long to exit; continuing shutdown.")
            except asyncio.CancelledError:
                pass
            finally:
                self.keepalive_task = None
        
        # Stop audio capture callback
        self._stop_audio_stream()
        
        # Close Deepgram connection
        if self.connection:
            try:
                # Use executor to avoid blocking on connection close
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.connection.finish)
            except Exception as e:
                print(f"Error closing Deepgram connection: {e}")
            self.connection = None
        
        await self._emit_event(STTEventType.CONNECTION_CLOSED, {
            "message": "STT connection closed"
        })
        
        print("✅ STT stopped")
    
    def is_currently_listening(self) -> bool:
        """Check if STT is currently listening."""
        return self.is_listening
    
    def get_session_speakers(self) -> Dict[int, str]:
        """Get the mapping of session speaker IDs to names."""
        return self.session_speakers.copy()
    
    def register_speaker(self, session_speaker_id: int, name: str):
        """Register a name for a session speaker ID."""
        self.session_speakers[session_speaker_id] = name
        print(f"📝 Registered speaker {session_speaker_id} as '{name}'")
    
    async def _keepalive_loop(self):
        """Send keepalive messages to prevent connection timeout."""
        try:
            while self.is_listening and self.connection_alive:
                # Send a keepalive message every interval
                if self.connection:
                    # Deepgram expects a JSON message for keepalive
                    keepalive_message = json.dumps({"type": "KeepAlive"})
                    self.connection.send(keepalive_message)
                await asyncio.sleep(self.keepalive_interval)
        except asyncio.CancelledError:
            # Expected during shutdown; exit quietly
            raise
        except Exception as e:
            print(f"❌ Keepalive error: {e}")

    def _start_audio_stream(self) -> bool:
        """Register the mel-aec capture callback for streaming audio."""
        try:
            stream = ensure_stream_started()
        except Exception as exc:
            print(f"❌ Failed to start mel-aec audio stream: {exc}")
            return False

        self.audio_stream = stream

        if self.audio_stop_event is None:
            self.audio_stop_event = threading.Event()
        else:
            self.audio_stop_event.clear()
        self._chunk_counter = 0
        self.stream_start_time = None
        self._last_prepare_error_log = 0.0
        self._last_fp_error_log = 0.0
        self._last_send_error_log = 0.0
        self._last_queue_full_log = 0.0

        if not self._start_sender_thread():
            print("❌ Failed to start Deepgram send worker thread")
            return False

        try:
            stream.set_input_callback(self._handle_audio_capture)
            self._audio_callback_registered = True
        except Exception as exc:
            print(f"❌ Failed to register audio capture callback: {exc}")
            self._audio_callback_registered = False
            return False

        return True

    def _start_sender_thread(self) -> bool:
        """Ensure a background thread is running to send audio to Deepgram."""
        if self._sender_thread and self._sender_thread.is_alive():
            return True

        # Fresh queue for this session
        self._send_queue = queue.Queue(maxsize=128)
        self._sender_stop_event = threading.Event()

        def _sender_worker():
            while True:
                if self._sender_stop_event and self._sender_stop_event.is_set():
                    break

                try:
                    chunk = self._send_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if chunk is None:
                    break

                if self._sender_stop_event and self._sender_stop_event.is_set():
                    break

                if self.audio_stop_event and self.audio_stop_event.is_set():
                    break

                if not self.is_listening or not self.connection or not self.connection_alive:
                    continue

                try:
                    self.connection.send(chunk)
                    self._chunk_counter += 1
                    if self._chunk_counter % 100 == 0:
                        print(f"📊 Sent {self._chunk_counter} audio chunks")
                except Exception as send_error:
                    now = time.time()
                    if now - self._last_send_error_log > 2.0:
                        print(f"❌ Failed to send audio data to Deepgram: {send_error}")
                        self._last_send_error_log = now
                    self.connection_alive = False
                    if self.audio_stop_event:
                        self.audio_stop_event.set()
                    self._schedule_event(STTEventType.ERROR, {
                        "error": f"Connection send failed: {send_error}"
                    })
                    break

            # Drain any remaining items to unblock producers
            if self._send_queue:
                try:
                    while True:
                        self._send_queue.get_nowait()
                except queue.Empty:
                    pass

        self._sender_thread = threading.Thread(
            target=_sender_worker,
            daemon=True,
            name="DeepgramSendWorker",
        )
        self._sender_thread.start()
        return True

    def _stop_audio_stream(self):
        """Stop streaming audio by disabling the capture callback."""
        if self.audio_stop_event:
            self.audio_stop_event.set()
        self._stop_sender_thread()

        if self.audio_stream and self._audio_callback_registered:
            try:
                self.audio_stream.set_input_callback(lambda _: None)
            except Exception as exc:
                print(f"⚠️  Failed to clear audio capture callback: {exc}")

        self._audio_callback_registered = False
        self.stream_start_time = None
        self._chunk_counter = 0
        self.audio_stream = None

    def _stop_sender_thread(self):
        """Signal the background sender thread to exit."""
        thread = self._sender_thread
        if not thread:
            return

        if self._sender_stop_event and not self._sender_stop_event.is_set():
            self._sender_stop_event.set()

        if self._send_queue:
            try:
                self._send_queue.put_nowait(None)
            except queue.Full:
                try:
                    self._send_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._send_queue.put_nowait(None)
                except queue.Full:
                    pass

        if thread.is_alive():
            thread.join(timeout=1.0)

        self._sender_thread = None
        self._sender_stop_event = None
        self._send_queue = None

    def _enqueue_audio_chunk(self, chunk: bytes) -> None:
        """Queue audio bytes for asynchronous sending to Deepgram."""
        if not self._send_queue:
            return
        if self._sender_stop_event and self._sender_stop_event.is_set():
            return

        try:
            self._send_queue.put_nowait(chunk)
        except queue.Full:
            now = time.time()
            if now - self._last_queue_full_log > 2.0:
                print("⚠️ Deepgram send queue full; dropping audio chunk")
                self._last_queue_full_log = now

    def _handle_audio_capture(self, audio_data) -> None:
        """mel-aec capture callback that forwards audio to Deepgram."""
        if self.audio_stop_event and self.audio_stop_event.is_set():
            return

        if not self.is_listening or not self.connection or not self.connection_alive:
            return

        if self.is_paused:
            return

        try:
            pcm_bytes = prepare_capture_chunk(audio_data, self.config.sample_rate)
        except Exception as prep_error:
            now = time.time()
            if now - self._last_prepare_error_log > 5.0:
                print(f"⚠️  Failed to prepare capture chunk: {prep_error}")
                self._last_prepare_error_log = now
            return

        if not pcm_bytes:
            return

        processed_data = pcm_bytes

        current_time = time.time()
        if self.stream_start_time is None:
            self.stream_start_time = current_time
            if self.voice_fingerprinter and getattr(self.config, 'debug_speaker_data', False):
                print(f"🕐 [STREAM] Capture callback started at {self.stream_start_time:.3f}")

        if self.voice_fingerprinter:
            try:
                audio_np = np.frombuffer(processed_data, dtype=np.int16).astype(np.float32) / 32768.0
                stream_relative_time = current_time - (self.stream_start_time or current_time)
                self.voice_fingerprinter.add_audio_chunk(audio_np, stream_relative_time)
            except Exception as fp_error:
                now = time.time()
                if now - self._last_fp_error_log > 5.0:
                    print(f"⚠️  Voice fingerprinting error: {fp_error}")
                    self._last_fp_error_log = now

        self._enqueue_audio_chunk(processed_data)
    
    def _on_open(self, *args, **kwargs):
        """Handle Deepgram connection open."""
        print("🔗 Deepgram STT connection opened")
        self.connection_alive = True
    
    def _on_transcript(self, *args, **kwargs):
        """Handle incoming transcripts."""
        result = kwargs.get('result')
        if not result or not hasattr(result, 'channel'):
            return
        
        try:
            channel = result.channel
            alternative = channel.alternatives[0]
            transcript = alternative.transcript.strip()
            confidence = alternative.confidence
            is_final = result.is_final
            
            # # Debug: Print raw response structure for final results
            # if is_final and hasattr(self.config, 'debug_speaker_data') and self.config.debug_speaker_data:
            #     print(f"🔬 [RAW DEBUG] Full response structure:")
            #     print(f"  result.is_final: {is_final}")
            #     print(f"  channel has alternatives: {hasattr(channel, 'alternatives')}")
            #     if hasattr(channel, 'alternatives') and channel.alternatives and len(channel.alternatives) > 0 and len(channel.alternatives[0].words) > 0:
            #         alt = channel.alternatives[0]
            #         print(f"  alternative.transcript: '{alt.transcript}'")
            #         print(f"  alternative.confidence: {alt.confidence}")
            #         print(f"  alternative has words: {hasattr(alt, 'words')}")
            #         if hasattr(alt, 'words'):
            #             print(f"  words count: {len(alt.words) if alt.words else 0}")
            
            if not transcript:
                return
            
            # Extract speaker information from words (for live streaming)
            speaker_id = None
            speaker_name = None
            
            # Get speaker information from words array (this is where Deepgram puts it for live streaming)
            if hasattr(alternative, 'words') and alternative.words:
                # Debug: Print detailed word-level information (show when diarization enabled or debug flag set)
                # if is_final and (self.config.debug_speaker_data or self.config.diarize):
                #     print(f"🔍 [SPEAKER DEBUG] Found {len(alternative.words)} words with timing:")
                #     for i, word in enumerate(alternative.words):
                #         word_text = getattr(word, 'word', 'NO_WORD')
                #         word_speaker = getattr(word, 'speaker', 'NO_SPEAKER')
                #         word_start = getattr(word, 'start', 'NO_START')
                #         word_end = getattr(word, 'end', 'NO_END')
                #         word_confidence = getattr(word, 'confidence', 'NO_CONF')
                #         print(f"  [{i:2d}] '{word_text}' | Speaker: {word_speaker} | Time: {word_start:.3f}-{word_end:.3f}s | Conf: {word_confidence:.3f}")
                
                # Find the most common speaker in this utterance
                speaker_counts = {}
                for word in alternative.words:
                    if hasattr(word, 'speaker'):
                        speaker = word.speaker
                        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                
                if speaker_counts:
                    # Use the speaker that spoke the most words in this utterance
                    speaker_id = max(speaker_counts, key=speaker_counts.get)
                    speaker_name = self.session_speakers.get(speaker_id)
                    # if is_final and (self.config.debug_speaker_data or self.config.diarize):
                    #     print(f"🎯 [SPEAKER DEBUG] Speaker analysis: {speaker_counts} → Primary speaker: {speaker_id}")
                    
                    # Process words for voice fingerprinting
                    if is_final and self.voice_fingerprinter:
                        try:
                            #print(f"🔊 [VOICE FINGERPRINT] Processing {len(alternative.words)} words for fingerprinting")
                            # Convert Deepgram words to our format
                            word_timings = []
                            utterance_start_time = time.time() - (alternative.words[-1].end if alternative.words else 0.0)
                            
                            for word in alternative.words:
                                word_timing = create_word_timing_from_deepgram_word(word)
                                word_timing.utterance_start = utterance_start_time
                                word_timings.append(word_timing)
                            
                            # Process synchronously (called from sync callback context)
                            if word_timings:
                                #print(f"🔊 [VOICE FINGERPRINT] Processing transcript words")
                                self.voice_fingerprinter.process_transcript_words(word_timings, utterance_start_time)
                        except Exception as fp_error:
                            print(f"⚠️  Voice fingerprinting word processing error: {fp_error}")
                    
                    # Check if we have a speaker name from voice fingerprinting
                    if self.voice_fingerprinter and speaker_id is not None:
                        fingerprint_name = self.voice_fingerprinter.get_speaker_name(speaker_id)
                        if fingerprint_name:
                            speaker_name = fingerprint_name
                            self.session_speakers[speaker_id] = fingerprint_name
            
            # Fallback: check metadata (for pre-recorded files)
            if speaker_id is None and hasattr(channel, 'metadata') and channel.metadata:
                speaker_id = getattr(channel.metadata, 'speaker', None)
                if speaker_id is not None:
                    speaker_name = self.session_speakers.get(speaker_id)
            
            # Create STT result
            stt_result = STTResult(
                text=transcript,
                confidence=confidence,
                is_final=is_final,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                timestamp=datetime.now(),
                raw_data=kwargs
            )
            
            # Apply speaker identification if enabled
            if self.config.enable_speaker_id and self.speaker_identifier:
                stt_result = self._apply_speaker_identification(stt_result)
            
            # Emit appropriate event
            channel_index = self._get_channel_index(result)
            if is_final:
                # Hold final transcript until we receive the matching UtteranceEnd event
                self._pending_final_results[channel_index] = stt_result
            else:
                self._schedule_event(
                    STTEventType.INTERIM_RESULT,
                    stt_result
                )
                
        except Exception as e:
            self._schedule_event(STTEventType.ERROR, {
                "error": f"Transcript processing error: {e}"
            })
    
    def _on_utterance_end(self, *args, **kwargs):
        """Handle utterance end events."""
        result_obj = kwargs.get("result")
        channel_index = self._get_channel_index(result_obj)

        stt_result = self._pending_final_results.pop(channel_index, None)
        if stt_result is None and self._pending_final_results:
            # Fallback: pull the most recent pending result if channel index was unavailable
            _, stt_result = self._pending_final_results.popitem()
        
        if stt_result is not None:
            # Emit the deferred utterance complete event now that Deepgram confirmed the end
            self._schedule_event(
                STTEventType.UTTERANCE_COMPLETE,
                stt_result
            )

            # Check for speaker change with the final information
            speaker_id = stt_result.speaker_id
            speaker_name = stt_result.speaker_name
            previous_speaker = self.current_speaker
            if speaker_id is not None and speaker_id != self.current_speaker:
                self.current_speaker = speaker_id
                self._schedule_event(
                    STTEventType.SPEAKER_CHANGE,
                    {
                        "speaker_id": speaker_id,
                        "speaker_name": speaker_name,
                        "previous_speaker": previous_speaker
                    }
                )

            self.last_utterance_time = datetime.now()
        
        self._schedule_event(STTEventType.SPEECH_ENDED, {
            "timestamp": datetime.now()
        })
    
    def _on_speech_started(self, *args, **kwargs):
        """Handle speech start events."""
        self._schedule_event(STTEventType.SPEECH_STARTED, {
            "timestamp": datetime.now()
        })
    
    def _on_error(self, *args, **kwargs):
        """Handle STT errors."""
        error = kwargs.get('error', 'Unknown error')
        self._schedule_event(STTEventType.ERROR, {
            "error": f"Deepgram error: {error}"
        })
    
    def _on_close(self, *args, **kwargs):
        """Handle connection close."""
        print("🔌 Deepgram connection closed by server")
        self.connection_alive = False
        
        # Cancel keepalive task if running
        if self.keepalive_task and not self.keepalive_task.done():
            self.keepalive_task.cancel()
        self.keepalive_task = None
        
        # Trigger reconnection if not already reconnecting and not shutting down
        if self.is_listening and not self.is_reconnecting:
            print("🔄 Attempting automatic reconnection...")
            asyncio.create_task(self._handle_reconnection())
        else:
            self.is_listening = False
            self._schedule_event(STTEventType.CONNECTION_CLOSED, {
                "message": "Deepgram connection closed by server"
            })
    
    def add_reference_audio(self, audio_data: bytes):
        """Compatibility hook; mel-aec handles echo cancellation internally."""
        return None
    
    def _setup_speaker_identification(self):
        """Setup speaker identification if enabled."""
        if self.config.speaker_profiles_path:
            try:
                # Import speaker identification module if available
                from improved_speaker_identification import SpeakerIdentifier
                self.speaker_identifier = SpeakerIdentifier(
                    profiles_file=self.config.speaker_profiles_path
                )
                print("✅ Speaker identification enabled")
            except ImportError:
                print("⚠️ Speaker identification module not found")
                self.speaker_identifier = None
    
    def _apply_speaker_identification(self, stt_result: STTResult) -> STTResult:
        """Apply speaker identification to the result."""
        if self.speaker_identifier and stt_result.raw_data:
            try:
                # Extract audio features and identify speaker
                # This would need to be implemented based on your speaker ID system
                identified_name = self.speaker_identifier.identify_speaker(
                    # Pass appropriate audio data
                )
                if identified_name:
                    stt_result.speaker_name = identified_name
            except Exception as e:
                print(f"Speaker identification error: {e}")
        
        return stt_result
    
    async def _emit_event(self, event_type: STTEventType, data: Any):
        """Emit an event to all registered callbacks."""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    print(f"Callback error for {event_type}: {e}")
    
    def _schedule_event(self, event_type: STTEventType, data: Any):
        """Schedule an event emission from a synchronous context."""
        try:
            # Try to get the current event loop
            current_loop = None
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = self.event_loop
            
            if current_loop and current_loop.is_running():
                if current_loop == self.event_loop:
                    # We're in the same loop, schedule normally
                    future = asyncio.run_coroutine_threadsafe(
                        self._emit_event(event_type, data), 
                        current_loop
                    )
                    # Don't wait for result to avoid blocking
                else:
                    # Different loop, use stored event loop
                    if self.event_loop and self.event_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._emit_event(event_type, data), 
                            self.event_loop
                        )
                    else:
                        # Fallback to sync emission
                        self._emit_event_sync(event_type, data)
            else:
                # No running loop, emit synchronously
                self._emit_event_sync(event_type, data)
                
        except Exception as e:
            print(f"Error scheduling event {event_type}: {e}")
            # Final fallback
            self._emit_event_sync(event_type, data)
    
    def _emit_event_sync(self, event_type: STTEventType, data: Any):
        """Emit an event synchronously (fallback for when no event loop is available)."""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if not asyncio.iscoroutinefunction(callback):
                        callback(data)
                    # Skip async callbacks in sync context
                except Exception as e:
                    print(f"Sync callback error for {event_type}: {e}")
    
    async def cleanup(self):
        """Clean up all resources."""
        await self.stop_listening()
        
        # Clear event loop reference
        self.event_loop = None
        
    def __del__(self):
        """Destructor to ensure cleanup."""
        # Don't call terminate in destructor - too risky for segfaults
        pass

# Convenience functions for quick usage
async def start_listening(api_key: str, on_utterance: Callable[[STTResult], None]) -> AsyncSTTStreamer:
    """
    Convenience function to start listening with default settings.
    
    Args:
        api_key: Deepgram API key
        on_utterance: Callback for completed utterances
        
    Returns:
        AsyncSTTStreamer: The STT streamer instance
    """
    config = STTConfig(api_key=api_key)
    stt = AsyncSTTStreamer(config)
    stt.on(STTEventType.UTTERANCE_COMPLETE, on_utterance)
    
    success = await stt.start_listening()
    if success:
        return stt
    else:
        await stt.cleanup()
        raise Exception("Failed to start STT")

# Example usage
async def main():
    """Example usage of the STT module."""
    api_key = input("Enter your Deepgram API key: ").strip()
    
    config = STTConfig(        api_key=api_key,
        interim_results=True,
        diarize=True
    )
    stt = AsyncSTTStreamer(config)
    
    # Set up callbacks
    async def on_utterance_complete(result: STTResult):
        speaker_info = f" (Speaker {result.speaker_id})" if result.speaker_id is not None else ""
        print(f"🎯 Final{speaker_info}: {result.text} (confidence: {result.confidence:.2f})")
    
    def on_interim_result(result: STTResult):
        print(f"💭 Interim: {result.text}")
    
    async def on_speech_started(data):
        print("🎤 Speech started")
    
    async def on_speech_ended(data):
        print("🔇 Speech ended")
    
    async def on_error(data):
        print(f"❌ Error: {data['error']}")
    
    # Register callbacks
    stt.on(STTEventType.UTTERANCE_COMPLETE, on_utterance_complete)
    stt.on(STTEventType.INTERIM_RESULT, on_interim_result)
    stt.on(STTEventType.SPEECH_STARTED, on_speech_started)
    stt.on(STTEventType.SPEECH_ENDED, on_speech_ended)
    stt.on(STTEventType.ERROR, on_error)
    
    try:
        print("🎤 Starting STT test...")
        success = await stt.start_listening()
        
        if success:
            print("✅ STT started successfully!")
            print("💡 Speak into your microphone. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while stt.is_currently_listening():
                await asyncio.sleep(1)
        else:
            print("❌ Failed to start STT")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        await stt.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 
