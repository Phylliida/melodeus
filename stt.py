
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v2 import (
    ListenV2Connected,
    ListenV2FatalError,
    ListenV2TurnInfo,
)
from deepgram.core.api_error import ApiError
import traceback
import numpy as np
import asyncio
from dataclasses import dataclass, field
from config_loader import STTConfig
import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum
from typing import Any, Dict, Optional
import time
from datetime import datetime
import uuid

from async_callback_manager import AsyncCallbackManager

# hardcoded for deepgram and other stuff
SAMPLE_RATE = 16000


# with a of like "*wow hi there* I like bees" and b of "wow hi there i like" this will give us index of end of like inside a
def _collapse(s: str, ignore: set[str]):
    kept = []
    idx_map = []
    for idx, ch in enumerate(s):
        if ch in ignore:
            continue
        kept.append(ch)
        idx_map.append(idx)
    return "".join(kept), idx_map

def float_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 samples in [-1, 1] to signed 16-bit PCM bytes."""
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


@dataclass
class STTResult:
    """Result from speech-to-text recognition."""
    text: str
    confidence: float
    is_final: bool
    is_edit: bool
    message_id: str
    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Optional[Dict[str, Any]] = None

class AsyncSTT(object):
    def __init__(self, audio_system, config: STTConfig):
        self.deepgram_api_key = config.deepgram_api_key
        self.keyterm = config.keyterm
        self.audio_system = audio_system
        self.voice_fingerprinter = None
        self.stt_callbacks = AsyncCallbackManager()
        if config.enable_speaker_id:
            from titanet_voice_fingerprinting import (
                TitaNetVoiceFingerprinter,
                WordTiming
            )
            try:
                # Enable debug audio saving if debug_speaker_data is enabled
                self.voice_fingerprinter = TitaNetVoiceFingerprinter(config.speaker_id, debug_save_audio=False)
                print(f"🤖 TitaNet voice fingerprinting enabled")
            except Exception as e:
                print(f"⚠️  TitaNet voice fingerprinting failed to initialize: {e}")
                print(traceback.print_exc())

    async def __aenter__(self):
        self.deepgram_task = asyncio.create_task(self.deepgram_processor())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.deepgram_task.cancel()
        try:
            await self.deepgram_task
        except asyncio.CancelledError:
            pass # intentional
        except asyncio.CancelledError:
            pass # intentional
        except:
            print(f"Error in deepgram task cancel")
            print(traceback.print_exc())
        del self.deepgram_task

    async def add_callback(self, callback):
        await self.stt_callbacks.add_callback(callback)

    async def remove_callback(self, callback):
        await self.stt_callbacks.remove_callback(callback)

    async def deepgram_processor(self):
        self.connection = None
        async def audio_callback(audio_input_data, audio_output_data, aec_input_data, channel_vads):
            try:
                # if we have any input channels available and aec detected something, mix them
                # alternatively we could run deepgram per channel but that would be more expensive
                if len(aec_input_data) > 0 and len(aec_input_data[0]) > 0 and any(channel_vads) and self.connection is not None:
                    mixed_data = np.zeros(len(aec_input_data[0]))
                    for i in range(len(channel_vads)):
                        vad = channel_vads[i]
                        if vad:
                            mixed_data += np.array(aec_input_data[i])
                    mixed_data_pcm = float_to_int16_bytes(mixed_data)
                    if self.voice_fingerprinter is not None:
                        self.voice_fingerprinter.add_audio_chunk(
                            mixed_data,
                            self.num_audio_frames_recieved,
                            16000,
                        )
                        # increment frame count
                        self.num_audio_frames_recieved += len(mixed_data)
                    
                    await connection.send_media(mixed_data_pcm)
            except:
                print("Error in audio callback")
                print(traceback.print_exc())
                raise
        while True:
            try:
                self.prev_turn_idx = None
                self.prev_transcript = ""
                self.prev_audio_window_start = 0
                self.prev_audio_window_end = 0
                self.current_turn_history = []
                self.current_turn_autosent_transcript = None
                self.last_sent_message_uuid = None
                self.num_audio_frames_recieved = 0
                deepgram = AsyncDeepgramClient(api_key=self.deepgram_api_key)
                connect_kwargs = dict(
                    model="flux-general-en",
                    encoding="linear16",
                    sample_rate=str(SAMPLE_RATE),
                    eot_timeout_ms="500",
                )
                if self.keyterm:
                    connect_kwargs["keyterm"] = self.keyterm

                async with deepgram.listen.v2.connect(**connect_kwargs) as connection:
                    connection.on(EventType.OPEN, self.deepgram_on_open)
                    connection.on(EventType.MESSAGE, self.deepgram_on_message)
                    connection.on(EventType.ERROR, self.deepgram_error)
                    connection.on(EventType.CLOSE, self.deepgram_close)
                    
                    self.connection = connection
                    await self.audio_system.add_callback(audio_callback)
                    await connection.start_listening()
                    self.connection = None
                    await self.audio_system.remove_callback(audio_callback)
            except asyncio.CancelledError:
                print("Deepgram processs canceled")
                break
            except ApiError as api_err:
                print("Error in deepgram websocket connection")
                print(traceback.print_exc())
                raise
            except:
                print("Error in deepgram")
                print(traceback.print_exc())
                raise
            finally:
                # fine to do this again, only removes if present
                await self.audio_system.remove_callback(audio_callback)


    def deepgram_on_open(self, *args, **kwargs):
        """Handle Deepgram connection open."""
        print("🔗 Deepgram STT connection opened")

    async def deepgram_on_message(self, message: Any):
        """Dispatch Deepgram websocket messages to the appropriate handlers."""
        try:
            if isinstance(message, ListenV2TurnInfo):
                await self.deepgram_turn(message)
            elif isinstance(message, ListenV2FatalError):
                print(f"❌ Deepgram fatal error ({message.code}): {message.description}")
            elif isinstance(message, ListenV2Connected):
                pass
        except Exception as dispatch_error:
            print(f"⚠️  Failed to process Deepgram message:")
            print(traceback.print_exc())

    async def _emit_stt(self, stt):
        await self.stt_callbacks(stt)

    async def deepgram_turn(self, message: ListenV2TurnInfo):
        event_type = (message.event or "").lower()
        turn_idx = message.turn_index
        transcript = message.transcript or ""
        # speech started: todo
        # if event_type == "startofturn":
        #     await self._on_speech_started()
        if transcript.strip() != "":
            print(transcript)

        send_message = False
        edit_message = False
        # code to end turn earlier than deepgram decides for decreased latency
        if turn_idx != self.prev_turn_idx:
            self.current_turn_history = []    
        if len(transcript) > 0:
            self.current_turn_history.append((time.time(), transcript))
        if turn_idx == self.prev_turn_idx and len(transcript) > 0 and self.has_meaningful_change(transcript, self.current_turn_autosent_transcript):
            ms_until_autosend = 750
            # if we took longer than that and still the same, autosend
            oldest_time_still_same = time.time()
            for old_time, old_transcript in self.current_turn_history[::-1]:
                if old_transcript != transcript:
                    break
                else:
                    oldest_time_still_same = old_time
            if (time.time() - oldest_time_still_same)*1000 > ms_until_autosend:
                #print("Autosend")
                if self.current_turn_autosent_transcript is not None:
                    edit_message = True
                else:
                    send_message = True
                self.current_turn_autosent_transcript = transcript
            
        # If we are on a new turn, send old turn
        if turn_idx != self.prev_turn_idx and self.prev_turn_idx != None and self.prev_transcript:
            if self.current_turn_autosent_transcript is not None:
                if self.has_meaningful_change(self.current_turn_autosent_transcript, self.prev_transcript):
                    edit_message = True
            else:
                send_message = True
            
        if send_message or edit_message:
            if self.voice_fingerprinter:
                from titanet_voice_fingerprinting import (
                    TitaNetVoiceFingerprinter,
                    WordTiming
                )
                word_timing = WordTiming(
                    word=self.prev_transcript,
                    speaker_id=f"speaker {turn_idx}", # v2 doesn't have diarization yet
                    start_time=self.prev_audio_window_start,
                    end_time=self.prev_audio_window_end,
                    confidence=1)
                user_tag = self.voice_fingerprinter.process_transcript_words(word_timings=[word_timing], sample_rate=SAMPLE_RATE)
            else:
                user_tag = "User"
            message_uuid = str(uuid.uuid4()) if send_message else self.last_sent_message_uuid
            self.last_sent_message_uuid = message_uuid
            #print(("send" if send_message else "edit"), "with data")
            #print(self.prev_transcript)
            stt_result = STTResult(
                text = self.prev_transcript,
                confidence = 1.0,
                is_final = True,
                is_edit = edit_message,
                speaker_id = None,
                speaker_name = user_tag,
                timestamp = datetime.now(),
                message_id = message_uuid,
            )
            await self._emit_stt(
                stt_result
            )
        if turn_idx != self.prev_turn_idx:
            self.last_sent_message_uuid = None
            self.current_turn_autosent_transcript = None
            

        self.prev_turn_idx = turn_idx
        self.prev_transcript = transcript
        self.prev_audio_window_start = message.audio_window_start
        self.prev_audio_window_end = message.audio_window_end

        if transcript and not send_message and not edit_message and self.last_sent_message_uuid is None:
            self.most_recent_turn_text = transcript
            
            stt_result = STTResult(
                text = transcript,
                confidence = 1.0,
                is_final = False,
                speaker_id = None,
                speaker_name =  None,
                timestamp = datetime.now(),
                message_id = None,
                is_edit = False
            ) 

            await self._emit_stt(
                stt_result
            )
            print(f"Turn {turn_idx} {transcript}")
        else:
            #print(f"Turn {turn_idx} Dummy turn")
            pass
        #if event_type in {"turn_end", "speech_ended", "speech_end"}:
        #    self._on_utterance_end(message)


    def has_meaningful_change(self, a, b):
        if a is None or b is None: return True
        ignore=set("*\n\r\t #@\\/.,?!-+[]()&%$:")
        # only send edited if it actually changed content (we don't care about punctuation diffs)
        a_cleaned, _ = _collapse(a, ignore)
        b_cleaned, _ = _collapse(b, ignore)
        return a_cleaned.lower() != b_cleaned.lower()

    async def deepgram_error(self, error: Exception):
        """Handle errors emitted by the Deepgram websocket client."""
        error_message = str(error)
        print(f"⚠️ Deepgram websocket error")
        print(error_message)

    def deepgram_close(self, *args, **kwargs):
        """Handle Deepgram connection open."""
        print("🔗 Deepgram STT connection closed")
    


            
