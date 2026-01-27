from melaec3 import InputDeviceConfig, OutputDeviceConfig
import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from copy import deepcopy
import json
import traceback


@dataclass
class ContextConfig:
    history: str = "contexts/context.jsonl"

@dataclass
class TTSVoiceConfigInner:
    model_id: str = "eleven_multilingual_v2"
    voice_id: str = "hpp4J3VqNfWAUOO0d1Us"  # default voice
    speed: float = 1.0
    stability: float = 0.5
    similarity_boost: float = 0.8

@dataclass
class TTSVoiceConfig:
    display_name: str = "A"
    prompt_name: str = "A"
    device: Dict[str, Any] = field(default_factory=dict)
    device_channels: List[int] = field(default_factory=lambda: [0])
    voice: TTSVoiceConfigInner = field(default_factory=TTSVoiceConfigInner)
    emotive_voice: TTSVoiceConfigInner = field(default_factory=TTSVoiceConfigInner)

    @classmethod
    def default(cls) -> "TTSVoiceConfig":
        return cls()

@dataclass
class TTSConfig:
    """Configuration for TTS settings."""
    elevenlabs_api_key: str = ""
    output_format: str = "pcm_22050"
    sample_rate: int = 22050
    resampler_quality: int = 5 # speex resampler quality, 5 is fine
    # have a default voices option if not initialized to something
    voices: Dict[str, TTSVoiceConfig] = field(
        default_factory=lambda: {"default": TTSVoiceConfig.default()}
    )

@dataclass(slots=True)
class AudioSystemState:
    input_devices: List[dict] = field(default_factory=list)
    output_devices: List[dict] = field(default_factory=list)

    # because this is a wrapper around a rust object we need special helpers here
    def to_dict(self) -> dict:
        return {
            'input_devices': [cfg.to_dict() for cfg in self.input_devices],
            'output_devices': [cfg.to_dict() for cfg in self.output_devices]
        }
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, dict_values) -> "AudioSystemState":
        res = cls({})
        parsed_inputs = []
        for cfg in dict_values['input_devices']:
            try:
                cfg = cfg if not hasattr(cfg, "to_dict") else cfg.to_dict()
                parsed_inputs.append(InputDeviceConfig.from_dict(cfg))
            except:
                print(f"Failed to parse input device {cfg}, ignoring:")
                print(traceback.print_exc())
        parsed_outputs = res.output_devices
        for cfg in dict_values['output_devices']:
            cfg = cfg if not hasattr(cfg, "to_dict") else cfg.to_dict()
            try:
                parsed_outputs.append(OutputDeviceConfig.from_dict(cfg))
            except:
                print(f"Failed to parse output device {cfg}, ignoring:")
                print(traceback.print_exc())
        res.input_devices = parsed_inputs
        res.output_devices = parsed_outputs
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
    auto_calibrate: bool = True
    state: AudioSystemState = field(default_factory=AudioSystemState)

@dataclass
class UIConfig:
    show_waveforms: bool = True
    host: str = "0.0.0.0"
    port: int = 5001

@dataclass
class MelodeusConfig:
    audio: AudioSystemConfig = field(default_factory=AudioSystemConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    ui: UIConfig = field(default_factory=UIConfig)    
