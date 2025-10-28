#!/usr/bin/env python3
"""
Async TTS Module for ElevenLabs WebSocket Streaming
Provides interruptible text-to-speech with real-time audio playback.
Now uses ElevenLabs alignment data to track spoken content.
"""

import asyncio
import websockets
import json
import base64
import time
import re
import difflib
import bisect
from collections import defaultdict
import threading
import traceback
import queue
from typing import Optional, AsyncGenerator, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    from mel_aec_audio import (
        ensure_stream_started,
        shared_sample_rate,
        write_playback_pcm,
        interrupt_playback,
        int16_bytes_to_float,
        _resample
    )
except ImportError:  # pragma: no cover - fallback when running from repo root
    import os
    import sys

    CURRENT_DIR = os.path.dirname(__file__)
    if CURRENT_DIR and CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from mel_aec_audio import (
        ensure_stream_started,
        shared_sample_rate,
        write_playback_pcm,
        interrupt_playback,
    )

@dataclass
class SpokenContent:
    """Represents content that was actually spoken by TTS."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class AlignmentChar:
    """Represents a single character alignment with its timing."""
    char: str
    start_time: float  # Seconds since audio playback start
    end_time: float    # Seconds since audio playback start
    global_index: int


@dataclass
class ToolCall:
    """Represents a tool call embedded in the text."""
    tag_name: str  # e.g., "function", "search", etc.
    content: str  # The full XML content including tags
    start_position: int  # Character position in generated_text where tool should execute
    end_position: int  # Character position where tool content ends
    executed: bool = False

@dataclass
class ToolResult:
    """Result from tool execution."""
    should_interrupt: bool  # Whether to interrupt current speech
    content: Optional[str] = None  # Optional content to insert into conversation/speak
    

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
    # Audio output device
    output_device_name: Optional[str] = None  # None = default device, or specify device name

class AsyncTTSStreamer:
    """Async TTS Streamer with interruption capabilities and spoken content tracking."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.speak_task = None
        self.generated_text = ""
    
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

    def _get_voice_key(self, voice_id: str, voice_settings: Dict[str, float]) -> str:
        """Generate a unique key for voice and settings combination."""
        settings_str = "_".join(f"{k}:{v}" for k, v in sorted(voice_settings.items()))
        return f"{voice_id}_{settings_str}"

    async def _speak_text_helper(self, text_generator: AsyncGenerator[str, None], first_audio_callback) -> bool:
        """
        Speak the given text (task that can be canceled)
        
        Args:
            text_generator: yields text to convert to speech
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
        current_voice = None
        websocket = None
        alignments = []
        self.generated_text = ""
        try:
            audio_stream = ensure_stream_started()
            audios = []

            async def flush_websocket(websocket):
                nonlocal first_audio_callback, alignments
                if websocket is not None:
                    await websocket.send(json.dumps({
                        "text": "", # final message
                    }))

                    audio_datas = []
                
                    # Get the audio and add it to audio queue
                    # we buffer all audio before playing because that prevents jitters
                    # (this is okay because we do this one sentence at a time)
                    async for message in websocket:
                        data = json.loads(message)
                        audio_base64 = data.get("audio")
                        if audio_base64 and len(audio_base64) > 0:
                            audio_data = base64.b64decode(audio_base64)
                            # call callback (this will stop the "thinking" sound)
                            if not first_audio_callback is None:
                                await first_audio_callback()
                                first_audio_callback = None

                            stream = ensure_stream_started()
                            float_audio = int16_bytes_to_float(audio_data)
                            resampled = _resample(float_audio, self.config.sample_rate, shared_sample_rate())
                            audio_datas.append(resampled)
                            alignments.append(data["alignment"])
                        elif data.get("isFinal"):
                            break # done
                    
                    # now send the audio, all in one piece
                    concat_data = np.concatenate(audio_datas)
                    stream.write(concat_data)

                    await websocket.close()
                    
            async for sentence in stream_sentences(text_generator):
                # add spaces back between the sentences
                self.generated_text = (self.generated_text + " " + sentence).strip()

                # split out *emotive* into seperate parts
                eleven_messages = []
                for text_part, is_emotive in self.extract_emotive_text(sentence):

                    if text_part.strip():
                        voice_id = self.config.emotive_voice_id if is_emotive else self.config.voice_id
                        voice_settings = self._get_voice_settings(is_emotive)

                        eleven_messages.append((voice_id, {
                            "text": text_part.strip(),
                            "voice_settings": voice_settings,
                        }))

                # send text to websockets and receive and play audio
                for voice_id, message in eleven_messages:
                    # Connect to WebSocket (if needed)
                    if current_voice != voice_id:
                        # Disconnect old websocket
                        if not websocket is None:
                            await flush_websocket(websocket)
                            websocket = None
                        current_voice = voice_id
            
                        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
                        params = f"?model_id={self.config.model_id}&output_format={self.config.output_format}"

                        websocket = await websockets.connect(uri + params)

                        # Send initial configuration
                        initial_message = {
                            "text": " ",
                            "voice_settings": message["voice_settings"],
                            "xi_api_key": self.config.api_key
                        }
                        await websocket.send(json.dumps(initial_message))
                    await websocket.send(json.dumps({
                        "text": message['text'].strip() + " ", # elevenlabs always requires ends with single space
                    }))

                # stream the speech, this allows us to start outputting speech before it's done outputting
                await flush_websocket(websocket)
                websocket = None
                current_voice = None

            # wait for buffered audio to drain (polling)
            while audio_stream.get_buffered_duration() > 0:
                await asyncio.sleep(0.05)
                
        except asyncio.CancelledError:
            print(f"Cancelled tts, interrupting playback")
            # interrupt audio, this clears the buffers
            interrupt_playback()
            raise
        except Exception as e:
            print(f"TTS error")
            print(traceback.print_exc())
        finally:
            # close websocket
            if websocket:
                try:
                    await websocket.close()
                except Exception as e:
                    print(f"TTS websocket close error")
                    print(traceback.print_exc())
                

    async def speak_text(self, text_generator: AsyncGenerator[str, None], first_audio_callback):
        # interrupt (if already running)
        await self.interrupt()

        # start the task (this way it's cancellable and we don't need to spam checks)
        self.speak_task = asyncio.create_task(self._speak_text_helper(text_generator, first_audio_callback))

        # wait for it to finish
        await self.speak_task

        self.speak_task = None
    
    def is_currently_playing(self):
        return self.speak_task is not None
    
    async def interrupt(self):
        if self.speak_task is not None: # if existing one, stop it
            try:
                self.speak_task.cancel()
                await self.speak_task # wait for it to cancel
            except asyncio.CancelledError:
                pass # intentional
            except Exception as e:
                print(f"TTS await error")
                print(traceback.print_exc()) 
            finally:
                self.speak_task = None
        return self.generated_text
        
    async def cleanup(self):
        await self.interrupt()

    def extract_emotive_text(self, text: str) -> List[Tuple[str, bool]]:
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


_SENTENCE_END = re.compile(r'([.?!][\'")\]]*)(?=\s)')

async def stream_sentences(
    text_stream: AsyncGenerator[str, None]
) -> AsyncGenerator[str, None]:
    buffer = ""

    async for chunk in text_stream:
        buffer += chunk

        while True:
            match = _SENTENCE_END.search(buffer)
            if not match:
                break

            end = match.end(1)
            sentence = buffer[:end].strip()
            buffer = buffer[end:].lstrip()

            if sentence:
                yield sentence

    tail = buffer.strip()
    if tail:
        yield tail
