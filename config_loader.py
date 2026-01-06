from melaec3 import InputDeviceConfig, OutputDeviceConfig
import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from copy import deepcopy
import json

@dataclass(slots=True)
class AudioSystemState:
    input_devices: List[dict] = field(default_factory=list)
    output_devices: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'input_devices': [cfg.to_dict() for cfg in self.input_devices],
            'output_devices': [cfg.to_dict() for cfg in self.output_devices]
        }
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, dict_values) -> "AudioSystemState":
        res = cls(**dict_values)
        res.input_devices = [InputDeviceConfig.from_dict(cfg) for cfg in res.input_devices]
        res.output_devices = [OutputDeviceConfig.from_dict(cfg) for cfg in res.output_devices]
        return res

    @classmethod
    def from_json(cls, raw: str) -> "AudioSystemState":
        dict_values = json.loads(raw)
        return cls.from_dict(dict_values)
        
@dataclass
class AudioSystemConfig:
    """Configuration for audio system."""
    loaded_devices: AudioSystemState = field(default_factory=AudioSystemState)
    output_device_frame_size: int = 16   # frame size (samples); keep very small (~1–2 ms) to avoid skips and allow quick interruption
    history_len: int = 100        # chunks buffered for alignment when frames drop or clocks drift; ~100 is typical
    calibration_packets: int = 20 # packets gathered before emitting audio; higher improves timing but slows startup
    audio_buf_seconds: int = 20   # seconds of input audio buffered; only a few seconds are usually needed
    resample_quality: int = 5     # Speex resampler quality level

@dataclass
class SpeakerProfile:
    """Configuration for a known speaker."""
    name: str = ""
    description: str = ""
    reference_audio: Optional[str] = None  # Path to reference audio file (30+ seconds)

@dataclass
class SpeakerRecognitionConfig:
    """Configuration for speaker recognition settings."""
    confidence_threshold: float = 0.7
    learning_mode: bool = True
    max_speakers: int = 4
    voice_fingerprint_length: int = 128

@dataclass
class SpeakersConfig:
    """Configuration for speaker identification and voice fingerprinting."""
    profiles: Dict[str, SpeakerProfile] = field(default_factory=dict)
    recognition: SpeakerRecognitionConfig = field(default_factory=SpeakerRecognitionConfig)

@dataclass
class STTConfig:
    """Configuration for text to speech and voice fingerprinting."""
    deepgram_api_key: str = ""
    keyterm: str = ""
    enable_speaker_id: bool = False
    speaker_id: SpeakersConfig = field(default_factory=SpeakersConfig)

@dataclass
class AudioSystemConfig:
    state: AudioSystemState = field(default_factory=AudioSystemState)

@dataclass
class MelodeusConfig:
    audio: AudioSystemConfig = field(default_factory=AudioSystemConfig)
    stt: STTConfig = field(default_factory=STTConfig)
