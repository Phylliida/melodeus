"""
Shared audio helpers that bridge Melodeus to the mel-aec engine.

We keep a singleton AudioStream so microphone capture and playback
share the same duplex stream (required for echo cancellation).
Utility functions handle float/int16 conversion and sample-rate
resampling for the STT (Deepgram) and TTS (ElevenLabs) pipelines.
"""

import math
import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy import signal

from audio_aec import AudioStream


@dataclass(frozen=True)
class StreamSettings:
    """Configuration applied to the shared mel-aec duplex stream."""

    sample_rate: int = 48000  # Hz
    channels: int = 1
    buffer_size: int = 480  # Samples per audio chunk
    enable_aec: bool = True
    aec_filter_length: int = 2048
    input_device: Optional[str] = None
    output_device: Optional[str] = None

_DEFAULT_STREAM_SETTINGS = StreamSettings()
_stream_lock = threading.Lock()
_stream_settings: StreamSettings = _DEFAULT_STREAM_SETTINGS
_shared_stream: Optional[AudioStream] = None
_stream_started = False
_MISSING = object()


def _normalize_device_name(name: Optional[Any]) -> Optional[str]:
    if name is None:
        return None
    value = str(name).strip()
    return value or None


def _coerce_optional_bool(value: Optional[Any]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def configure_audio_stream(
    *,
    sample_rate: Any = _MISSING,
    channels: Any = _MISSING,
    buffer_size: Any = _MISSING,
    enable_aec: Any = _MISSING,
    aec_filter_length: Any = _MISSING,
    input_device: Any = _MISSING,
    output_device: Any = _MISSING,
) -> None:
    """
    Update global stream settings. Stops and releases the shared stream if parameters change.
    """

    def _coerce_int(value: Optional[Any], minimum: int) -> Optional[int]:
        if value is None:
            return None
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            return None
        return candidate if candidate >= minimum else None

    global _stream_settings, _shared_stream, _stream_started

    with _stream_lock:
        current = _stream_settings

        if sample_rate in (_MISSING, None):
            new_sample_rate = current.sample_rate
        else:
            new_sample_rate = _coerce_int(sample_rate, 4000) or current.sample_rate

        if channels in (_MISSING, None):
            new_channels = current.channels
        else:
            new_channels = _coerce_int(channels, 1) or current.channels

        if buffer_size is _MISSING:
            coalesced_buffer = current.buffer_size
        else:
            coalesced_buffer = _coerce_int(buffer_size, 1)
            if coalesced_buffer is None:
                coalesced_buffer = current.buffer_size

        if buffer_size is _MISSING and sample_rate not in (_MISSING, None) and new_sample_rate != current.sample_rate:
                coalesced_buffer = max(1, new_sample_rate // 100)  # ~10 ms buffers
        new_aec_filter = (
            _coerce_int(aec_filter_length, 32) if aec_filter_length not in (_MISSING, None) else None
        )
        if new_aec_filter is None:
            new_aec_filter = current.aec_filter_length

        if enable_aec is _MISSING:
            new_enable_aec = current.enable_aec
        else:
            coerced_enable = _coerce_optional_bool(enable_aec)
            new_enable_aec = current.enable_aec if coerced_enable is None else coerced_enable

        if input_device is _MISSING:
            new_input = current.input_device
        else:
            new_input = _normalize_device_name(input_device)

        if output_device is _MISSING:
            new_output = current.output_device
        else:
            new_output = _normalize_device_name(output_device)

        new_settings = StreamSettings(
            sample_rate=new_sample_rate,
            channels=new_channels,
            buffer_size=coalesced_buffer,
            enable_aec=new_enable_aec,
            aec_filter_length=new_aec_filter,
            input_device=new_input,
            output_device=new_output,
        )

        if new_settings == current:
            return

        # Tear down any existing stream so the next access picks up new settings
        if _shared_stream is not None:
            try:
                if _stream_started:
                    _shared_stream.stop()
            except Exception:
                pass
            finally:
                _shared_stream = None
                _stream_started = False

        _stream_settings = new_settings


def configure_audio_stream_from_config(config: Any) -> None:
    """
    Apply configuration values from a VoiceAIConfig to the shared mel-aec stream.
    The function gracefully ignores missing attributes to remain backward compatible.
    """
    audio_cfg = getattr(config, "audio", None)
    stt_cfg = getattr(config, "stt", None)
    tts_cfg = getattr(config, "tts", None)
    conversation_cfg = getattr(config, "conversation", None)

    sample_rate = getattr(audio_cfg, "stream_sample_rate", _MISSING) if audio_cfg else _MISSING
    channels = getattr(audio_cfg, "stream_channels", _MISSING) if audio_cfg else _MISSING
    buffer_size = getattr(audio_cfg, "stream_buffer_size", _MISSING) if audio_cfg else _MISSING

    enable_aec = getattr(audio_cfg, "stream_enable_aec", _MISSING) if audio_cfg else _MISSING
    if enable_aec is _MISSING and conversation_cfg is not None:
        enable_aec = getattr(conversation_cfg, "enable_echo_cancellation", _MISSING)

    aec_filter_length = getattr(audio_cfg, "aec_filter_length", _MISSING) if audio_cfg else _MISSING
    if aec_filter_length is _MISSING and conversation_cfg is not None:
        aec_filter_length = getattr(conversation_cfg, "aec_filter_length", _MISSING)

    input_device = getattr(audio_cfg, "input_device_name", _MISSING) if audio_cfg else _MISSING
    if (input_device is _MISSING or not input_device) and stt_cfg is not None:
        fallback_input = getattr(stt_cfg, "input_device_name", None)
        if fallback_input:
            input_device = fallback_input
        elif input_device is _MISSING:
            input_device = _MISSING

    output_device = getattr(audio_cfg, "output_device_name", _MISSING) if audio_cfg else _MISSING
    if (output_device is _MISSING or not output_device) and tts_cfg is not None:
        fallback_output = getattr(tts_cfg, "output_device_name", None)
        if fallback_output:
            output_device = fallback_output
        elif output_device is _MISSING:
            output_device = _MISSING

    configure_audio_stream(
        sample_rate=sample_rate,
        channels=channels,
        buffer_size=buffer_size,
        enable_aec=enable_aec,
        aec_filter_length=aec_filter_length,
        input_device=input_device,
        output_device=output_device,
    )


def _current_settings() -> StreamSettings:
    with _stream_lock:
        return _stream_settings


def _get_shared_stream() -> AudioStream:
    """Return the singleton AudioStream instance, creating it on demand."""
    global _shared_stream
    with _stream_lock:
        if _shared_stream is None:
            settings = _stream_settings
            _shared_stream = AudioStream(
                sample_rate=settings.sample_rate,
                channels=settings.channels,
                buffer_size=settings.buffer_size,
                enable_aec=settings.enable_aec,
                aec_filter_length=settings.aec_filter_length,
                input_device=settings.input_device,
                output_device=settings.output_device,
            )
            print(
                "🎚️ mel-aec stream initialized "
                f"@ {settings.sample_rate}Hz, buffer={settings.buffer_size} samples, "
                f"channels={settings.channels}, AEC={'on' if settings.enable_aec else 'off'}, "
                f"input={settings.input_device or 'default'}, "
                f"output={settings.output_device or 'default'}"
            )
        return _shared_stream


def ensure_stream_started() -> AudioStream:
    """Start the duplex stream exactly once and return it."""
    global _stream_started
    stream = _get_shared_stream()
    with _stream_lock:
        if not _stream_started:
            stream.start()
            _stream_started = True
    return stream


def stop_stream():
    """Stop the shared stream (used during shutdown)."""
    global _stream_started
    with _stream_lock:
        if _shared_stream and _stream_started:
            _shared_stream.stop()
            _stream_started = False


def shared_sample_rate() -> int:
    """Expose the sample rate used by the shared stream."""
    with _stream_lock:
        return _stream_settings.sample_rate


def int16_bytes_to_float(audio_bytes: bytes) -> np.ndarray:
    """Convert signed 16-bit PCM bytes to float32 in [-1, 1]."""
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 samples in [-1, 1] to signed 16-bit PCM bytes."""
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio with sane defaults."""
    if audio.size == 0 or src_rate == dst_rate:
        return audio
    gcd = math.gcd(src_rate, dst_rate)
    up = dst_rate // gcd
    down = src_rate // gcd
    return signal.resample_poly(audio, up, down).astype(np.float32, copy=False)


def write_playback_pcm(pcm_bytes: bytes, source_rate: int) -> int:
    """
    Write PCM16 playback data (e.g. ElevenLabs chunks) into the shared stream.

    Returns the number of samples written to the mel-aec stream.
    """
    stream = ensure_stream_started()
    target_rate = shared_sample_rate()
    float_audio = int16_bytes_to_float(pcm_bytes)
    resampled = _resample(float_audio, source_rate, target_rate)
    return stream.write(resampled)


def write_playback_float(audio: np.ndarray, source_rate: int) -> int:
    """
    Write float32 audio into the shared stream, resampling if needed.
    """
    stream = ensure_stream_started()
    target_rate = shared_sample_rate()
    resampled = _resample(audio, source_rate, target_rate)
    return stream.write(resampled)


def read_capture_chunk(target_samples: int, target_rate: int) -> bytes:
    """
    Read microphone audio from the shared stream and return PCM16 bytes
    at the desired sample rate (Deepgram expects 16 kHz).
    """
    settings = _current_settings()
    stream = ensure_stream_started()
    # Request enough samples from the shared stream to satisfy the target chunk.
    samples_needed = max(
        int(math.ceil(target_samples * settings.sample_rate / target_rate)), settings.buffer_size
    )
    float_audio = stream.read(samples_needed)
    if float_audio.size == 0:
        return b""
    resampled = _resample(float_audio, settings.sample_rate, target_rate)
    # Guarantee we do not exceed the requested chunk size to avoid buffering drift.
    if resampled.size > target_samples:
        resampled = resampled[:target_samples]
    elif resampled.size < target_samples:
        # Pad with silence if the buffer was short.
        resampled = np.pad(resampled, (0, target_samples - resampled.size), "constant")
    return float_to_int16_bytes(resampled)
