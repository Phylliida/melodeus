
import traceback
import numpy as np
import asyncio
from dataclasses import dataclass, field
from config_loader import STTConfig, SecretsConfig
import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum
from typing import Any, Dict, Optional
import time
from datetime import datetime
import uuid
from elevenlabs import ElevenLabs, AudioFormat, RealtimeEvents
from contextlib import asynccontextmanager
from async_callback_manager import AsyncCallbackManager
import base64

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
    message_id: str
    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class AsyncElevenLabsSTT(object):
    def __init__(self, audio_system, key: str, config: STTConfig):
        self.elevenlabs_api_key = key
        self.keyterm = config.keyterm
        self.audio_system = audio_system
        self.voice_fingerprinter = None
        self.stt_callbacks = AsyncCallbackManager()
        
    async def __aenter__(self):
        self.has_new_text = False
        self.cur_gain = 1.0
        self.enabled = True
        self.gain_avg_mu = 0.99
        self.time_of_last_speech = time.time()
        self.transcript = ""
        self.turn_id = str(uuid.uuid4())
        client = ElevenLabs(api_key=self.elevenlabs_api_key)
        self.connection = await client.speech_to_text.realtime.connect({
            "model_id": "scribe_v2_realtime",
            "audio_format": AudioFormat.PCM_16000,
                "sample_rate": 16000,
        })
        self.loop = asyncio.get_running_loop()

        self.connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, self.partial_transcript)
        self.connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, self.commited_transcript)
        await self.audio_system.add_audio_callback(self.audio_callback)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.audio_system.remove_audio_callback(self.audio_callback)
        await self.connection.close()

    def schedule_emit(self, result):
        task = self.loop.create_task(self._emit_stt(result))
        def _log_err(t):
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                print("Error emitting STT:", exc)
        task.add_done_callback(_log_err)

    def partial_transcript(self, event):
        text = event['text']
        print("partial transcript")
        print(text)
        if has_meaningful_change(text, self.transcript):
            self.has_new_text = True
            self.schedule_emit(
                STTResult(
                    text = text,
                    speaker_id = None,
                    speaker_name = "Unknown",
                    timestamp = datetime.now(),
                    message_id = self.turn_id,
                )
            )
            self.transcript = text
        
    def commited_transcript(self, event):
        text = event['text']
        print("Commited transcript")
        print(text)
        if has_meaningful_change(text, self.transcript):
            # don't commit because this will interrupt them
            '''
            self.schedule_emit(
                STTResult(
                    text = text,
                    speaker_id = None,
                    speaker_name = "Unknown",
                    timestamp = datetime.now(),
                    message_id = self.turn_id,
                )
            )
            '''
        self.has_new_text = False
        self.turn_id = str(uuid.uuid4())
        self.transcript = ""


    async def add_callback(self, callback):
        await self.stt_callbacks.add_callback(callback)

    async def remove_callback(self, callback):
        await self.stt_callbacks.remove_callback(callback)

    async def audio_callback(self, input_channels, output_channels, audio_input_data, audio_output_data, aec_input_data, channel_vads):
        # if we have any input channels available and aec detected something, mix them
        # alternatively we could run deepgram per channel but that would be more expensive
        if (len(aec_input_data) > 0 and (True or (any(channel_vads) and self.connection is not None))):
            self.time_of_last_speech = time.time()
            frame_size = len(aec_input_data)//input_channels
            aec_frames = aec_input_data.reshape(-1, input_channels) # view, no copy
            if not hasattr(self, "mixed_data") or self.mixed_data is None or len(self.mixed_data) != frame_size:
                self.mixed_data = np.zeros(frame_size)
            self.mixed_data[:] = 0
            for channel in range(len(channel_vads)):
                vad = channel_vads[channel]
                if vad and self.enabled:
                    self.mixed_data += aec_frames[:,channel]
            R_target = 0.95
            avg_energy = np.sqrt(np.dot(self.mixed_data, self.mixed_data))
            if avg_energy > 0.05: # too quiet is just noise
                gain = R_target / np.sqrt(avg_energy)
                # slowly move gain
                self.cur_gain = self.gain_avg_mu*self.cur_gain + (1-self.gain_avg_mu)*gain
                self.cur_gain = 1
                self.mixed_data *= self.cur_gain
                #print(avg_energy, self.cur_gain)
                np.clip(self.mixed_data, -1.0, 1.0, out=self.mixed_data)
            #else:
            #    #print("too quiet, ignoring")
            #    #return
            mixed_data_pcm = float_to_int16_bytes(self.mixed_data)
            mixed_data_pcm_base_64 = base64.b64encode(mixed_data_pcm).decode()
            await self.connection.send(
                {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": mixed_data_pcm_base_64,
                    "commit": False,
                    "sample_rate": 16000,
                })
    async def commit(self):
        if self.has_new_text:
            await self.connection.commit()
            self.has_new_text = False
    async def _emit_stt(self, stt):
        await self.stt_callbacks(stt)