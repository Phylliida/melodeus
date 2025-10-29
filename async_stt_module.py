
import asyncio
import queue
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum

import numpy as np
import json

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    ListenV1MetadataEvent,
    ListenV1ResultsEvent,
    ListenV1SpeechStartedEvent,
    ListenV1UtteranceEndEvent,
)

from mel_aec_audio import ensure_stream_started, prepare_capture_chunk
from titanet_voice_fingerprinting import (
    TitaNetVoiceFingerprinter,
)

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
        # Deepgram client
        self.deepgram = AsyncDeepgramClient(api_key=config.api_key)
        self.audio_queue = queue.Queue(maxsize=1280)
        self.listening = False
        self.num_audio_frames_recieved = 0
        stream = ensure_stream_started()
        stream.set_input_callback(self._handle_audio_capture)
        self.audio_task = None
        self.websocket_task = None


        self.callbacks: Dict[STTEventType, List[Callable]] = {
            event_type: [] for event_type in STTEventType
        }

        # TitaNet voice fingerprinting (optional)
        self.voice_fingerprinter = None
        if config.enable_speaker_id:
            try:
                # Enable debug audio saving if debug_speaker_data is enabled
                debug_save_audio = getattr(config, 'debug_speaker_data', False)
                self.voice_fingerprinter = TitaNetVoiceFingerprinter(speakers_config, debug_save_audio=debug_save_audio)
                print(f"🤖 TitaNet voice fingerprinting enabled")
                if debug_save_audio:
                    print(f"🐛 Debug audio saving enabled - extracted segments will be saved to debug_audio_segments/")
            except Exception as e:
                print(f"⚠️  TitaNet voice fingerprinting failed to initialize: {e}")
        

    def on(self, event_type: STTEventType, callback: Callable):
        """Register a callback for an event type."""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def off(self, event_type: STTEventType, callback: Callable):
        """Unregister a callback for an event type."""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)

    def _build_listen_params(self) -> Dict[str, Any]:
        """Prepare query parameters for Deepgram Listen v1 connection."""
        params: Dict[str, Any] = {
            "model": self.config.model,
            "encoding": "linear16",
            "sample_rate": str(self.config.sample_rate),
            "channels": str(self.config.channels),
            "smart_format": str(self.config.smart_format).lower(),
            "interim_results": str(self.config.interim_results).lower(),
            "punctuate": str(self.config.punctuate).lower(),
            "diarize": str(self.config.diarize).lower(),
            "vad_events": str(self.config.vad_events).lower(),
        }

        if self.config.language:
            params["language"] = self.config.language
        if self.config.utterance_end_ms is not None:
            params["utterance_end_ms"] = str(self.config.utterance_end_ms)

        # Add keywords/keyterms based on model
        if self.config.keywords:
            if self.config.model == "nova-3":
                keyterms = [word for word, _ in self.config.keywords if word]
                if keyterms:
                    params["keyterm"] = " ".join(keyterms)
            else:
                sanitized_keywords = []
                for word, weight in self.config.keywords:
                    sanitized_word = word.replace(" ", "_").replace(",", "")
                    if sanitized_word:
                        sanitized_keywords.append(f"{sanitized_word}:{weight}")
                if sanitized_keywords:
                    params["keywords"] = ",".join(sanitized_keywords)

        return params
    

    def _on_open(self, *args, **kwargs):
        """Handle Deepgram connection open."""
        print("🔗 Deepgram STT connection opened")
    
    def _handle_socket_message(self, message: Any):
        """Dispatch Deepgram websocket messages to the appropriate handlers."""
        try:
            if isinstance(message, ListenV1ResultsEvent):
                self._on_transcript(message)
            elif isinstance(message, ListenV1UtteranceEndEvent):
                self._on_utterance_end(message)
            elif isinstance(message, ListenV1SpeechStartedEvent):
                self._on_speech_started()
            elif isinstance(message, ListenV1MetadataEvent):
                # Currently unused but available for future diagnostics
                pass
        except Exception as dispatch_error:
            print(f"⚠️  Failed to process Deepgram message:")
            print(traceback.print_exc())

    def _handle_socket_error(self, error: Exception):
        """Handle errors emitted by the Deepgram websocket client."""
        error_message = str(error)
        print(f"⚠️ Deepgram websocket error")
        print(error_message)

    def _on_speech_started(self):
        print("Deepgram speech started")

    def _on_transcript(self, message: Any):
        print("Deepgram on transcript")
        print(json.dumps(message))

    def _on_utterance_end(self, message: Any):
        print("Deepgram utterance end")
        print(message)

    def _on_close(self, *args, **kwargs):
        """Handle Deepgram connection open."""
        print("🔗 Deepgram STT connection closed")
    

    def _handle_audio_capture(self, audio_data) -> None:
        try:
            if self.listening:
                self.audio_queue.put_nowait(audio_data)
        except Exception as prep_error:
            print(f"⚠️  Failed to prepare capture chunk")
            print(traceback.print_exc())
    
    async def create_audio_task(self, connection):
        try:
            while True:
                waitingAmount = self.audio_queue.qsize()
                if waitingAmount > 100:
                    print(f"Warning: {waitingAmount} audio in queue, we are falling behind")
                try:
                    audio_data = self.audio_queue.get_nowait()
                except queue.Empty:
                    # poll
                    await asyncio.sleep(0.05)
                    continue
                
                pcm_bytes = prepare_capture_chunk(audio_data, self.config.sample_rate)
                audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                self.voice_fingerprinter.add_audio_chunk(audio_np, self.num_audio_frames_recieved, self.config.sample_rate)
                # increment frame count
                self.num_audio_frames_recieved += len(audio_np)
                
                await connection.send_media(pcm_bytes)
        except asyncio.CancelledError:
            print("Audio task canceled")
            raise
        except:
            print(f"Error in audio task")
            print(traceback.print_exc())
            raise
        finally:
            self.audio_task = None

    async def cancel_task(self, task):
        if not task is None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass # intentional
            except:
                print(f"Error in task cancel")
                print(traceback.print_exc())

    async def create_websocket_task(self):
        try:
            while True:
                params = self._build_listen_params()
                async with asyncio.TaskGroup() as tg: # this audio awaits for it to cancel/finish when we try to exit context
                    self.num_audio_frames_recieved = 0
                    async with self.deepgram.listen.v1.connect(**params) as connection:
                        connection.on(EventType.OPEN, self._on_open)
                        connection.on(EventType.MESSAGE, self._handle_socket_message)
                        connection.on(EventType.ERROR, self._handle_socket_error)
                        connection.on(EventType.CLOSE, self._on_close)
                        # create audio task (leaving the with auto-cancels/awaits)
                        audio_task = tg.create_task(self.create_audio_task(connection))
                        await connection.start_listening()
                    audio_task.cancel()
        except asyncio.CancelledError:
            print("Websocket task canceled")
            raise
        except:
            print(traceback.print_exc())
            print("Error in websocket task")
            raise
        finally:
            self.websocket_task = None

    def clear_audio_queue(self):
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    async def start_listening(self):
        await self.cancel_task(self.websocket_task) # cancel listening task if already exists
        self.clear_audio_queue()
        self.listening = True
        self.websocket_task = asyncio.create_task(self.create_websocket_task())
        return True

    async def stop_listening(self):
        await self.cancel_task(self.websocket_task)
        self.listening = False
        self.clear_audio_queue()
        return True
    
    async def cleanup(self):
        await self.stop_listening()
