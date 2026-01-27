
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

# Todo polishing for deepgram:
# don't interrupt for very small changes to text (edit distance less than 5-10 or so, or small enough audio bursts)
#     bc echo cancel isn't perfect
# a "no interruption" mode that simply responds after x amount of time
#   can be useful in noise heavy environments
#   possibly have a button that can be pressed to manually interrupt still tho



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


def has_meaningful_change(a, b):
    if a is None or b is None: return True
    ignore=set("*\n\r\t #@\\/.,?!-+[]()&%$:")
    # only send edited if it actually changed content (we don't care about punctuation diffs)
    a_cleaned, _ = _collapse(a, ignore)
    b_cleaned, _ = _collapse(b, ignore)
    return a_cleaned.lower() != b_cleaned.lower()

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


class TurnState(object):
    def __init__(self, stt, turn_index, transcript, audio_window_start, audio_window_end):
        self.turn_index = turn_index
        self.stt = stt
        self.transcript = transcript
        self.uuid = str(uuid.uuid4())
        self.last_write = (time.time(), transcript)
        self.audio_window_start = audio_window_start
        self.audio_window_end = audio_window_end
        self.is_edit = False
    
    async def emit(self, final: bool = False):
        emit_data = self.build_emit_data(final)
        await self.stt._emit_stt(emit_data)

        # later ones are edits
        self.is_edit = True

    def build_emit_data(self, is_final: bool):
        if self.stt.voice_fingerprinter:
            from titanet_voice_fingerprinting import (
                TitaNetVoiceFingerprinter,
                WordTiming
            )
            word_timing = WordTiming(
                word=self.transcript,
                speaker_id=f"speaker {self.turn_idx}", # v2 doesn't have diarization yet
                start_time=self.audio_window_start,
                end_time=self.audio_window_end,
                confidence=1)
            user_tag = self.voice_fingerprinter.process_transcript_words(word_timings=[word_timing], sample_rate=SAMPLE_RATE)
        else:
            user_tag = "User"
        return STTResult(
            text = self.transcript,
            confidence = 1.0,
            is_final = is_final,
            is_edit = self.is_edit,
            speaker_id = None,
            speaker_name = user_tag,
            timestamp = datetime.now(),
            message_id = self.uuid,
        )

    async def process(self, message):
        # moved onto new turn, reset last write
        if message.turn_index != self.turn_index:
            self.last_write = (time.time(), "")
            self.turn_index = message.turn_index
            res = TurnState(
                self.stt,
                message.turn_index,
                "",
                message.audio_window_start,
                message.audio_window_end)
            return await res.process(message)
        
        # still this turn, see if modified
        transcript = message.transcript
        last_modified_time, last_transcript = self.last_write
        if has_meaningful_change(transcript, last_transcript):
            self.last_write = (time.time(), transcript)
            self.transcript = transcript
            self.audio_window_end = message.audio_window_end
            await self.emit(final=False)
        return self



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
        self.time_of_last_speech = time.time()
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
        self.mixed_data = None
        self.cur_gain = 1.0
        self.gain_avg_mu = 0.99
        async def audio_callback(input_channels, output_channels, audio_input_data, audio_output_data, aec_input_data, channel_vads):
            try:
                # if we have any input channels available and aec detected something, mix them
                # alternatively we could run deepgram per channel but that would be more expensive
                if len(aec_input_data) > 0 and any(channel_vads) and self.connection is not None:
                    self.time_of_last_speech = time.time()
                    frame_size = len(aec_input_data)//input_channels
                    aec_frames = aec_input_data.reshape(-1, input_channels) # view, no copy
                    if self.mixed_data is None or len(self.mixed_data) != frame_size:
                        self.mixed_data = np.zeros(frame_size)
                    self.mixed_data[:] = 0
                    for channel in range(len(channel_vads)):
                        vad = channel_vads[channel]
                        if vad:
                            self.mixed_data += aec_frames[:,channel]
                    R_target = 0.95
                    avg_energy = np.sqrt(np.dot(self.mixed_data, self.mixed_data))
                    if avg_energy > 0.1: # too quiet is just noise
                        gain = R_target / np.sqrt(avg_energy)
                        # slowly move gain
                        self.cur_gain = self.gain_avg_mu*self.cur_gain + (1-self.gain_avg_mu)*gain
                        self.mixed_data *= self.cur_gain
                        #print(avg_energy, self.cur_gain)
                        np.clip(self.mixed_data, -1.0, 1.0, out=self.mixed_data)
                    mixed_data_pcm = float_to_int16_bytes(self.mixed_data)
                    if self.voice_fingerprinter is not None:
                        self.voice_fingerprinter.add_audio_chunk(
                            self.mixed_data,
                            self.num_audio_frames_recieved,
                            16000,
                        )
                        # increment frame count
                        self.num_audio_frames_recieved += len(self.mixed_data)
                    await connection.send_media(mixed_data_pcm)
            except:
                print("Error in audio callback")
                print(traceback.print_exc())
                raise
        while True:
            try:
                self.turn_state = TurnState(self, -1000, "", 0, 0)
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
                    await self.audio_system.add_audio_callback(audio_callback)
                    await connection.start_listening()
                    self.connection = None
                    await self.audio_system.remove_audio_callback(audio_callback)
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
                await self.audio_system.remove_audio_callback(audio_callback)


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
        self.turn_state = await self.turn_state.process(message)
        
    async def deepgram_error(self, error: Exception):
        """Handle errors emitted by the Deepgram websocket client."""
        error_message = str(error)
        print(f"⚠️ Deepgram websocket error")
        print(error_message)

    def deepgram_close(self, *args, **kwargs):
        """Handle Deepgram connection open."""
        print("🔗 Deepgram STT connection closed")
    


            
