#!/usr/bin/env python3
"""
Configuration Loader for Voice AI System
Loads YAML configuration and creates appropriate dataclass configurations.
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from copy import deepcopy

# Import our configuration dataclasses
from async_stt_module import STTConfig
from async_tts_module import TTSConfig

@dataclass
class ConversationConfig:
    """Configuration for the unified conversation system."""
    # API Keys
    deepgram_api_key: str
    elevenlabs_api_key: str
    openai_api_key: str
    anthropic_api_key: str = ""
    
    # Voice settings
    voice_id: str = "T2KZm9rWPG5TgXTyjt7E"
    
    # Conversation timing
    pause_threshold: float = 2.0
    min_words_for_submission: int = 3
    max_wait_time: float = 10.0
    
    # Interruption settings
    interruption_confidence: float = 0.8
    
    # STT settings
    stt_model: str = "nova-3"
    stt_language: str = "en-US"
    interim_results: bool = True
    
    # TTS settings
    tts_model: str = "eleven_multilingual_v1"
    tts_speed: float = 1.0
    tts_stability: float = 0.5
    tts_similarity_boost: float = 0.8
    
    # LLM settings
    llm_provider: str = "openai"  # openai or anthropic
    llm_model: str = "chatgpt-4o-latest"
    conversation_mode: str = "chat"  # chat or prefill
    max_tokens: int = 300
    system_prompt: str = "You are a helpful AI assistant in a voice conversation. Give natural, conversational responses that work well when spoken aloud. Keep responses concise but engaging."
    
    # Prefill mode settings
    prefill_user_message: str = '<cmd>cat untitled.txt</cmd>'
    prefill_participants: List[str] = None
    prefill_system_prompt: str = 'The assistant is in CLI simulation mode, and responds to the user\'s CLI commands only with outputs of the commands.'
    
    # History file settings
    history_file: Optional[str] = None
    
    # Tools configuration
    tools_config: Optional[Dict[str, Any]] = None
    
    # Character configuration
    characters_config: Optional[Dict[str, Any]] = None
    director_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.prefill_participants is None:
            self.prefill_participants = ['H', 'Claude']

@dataclass
class SpeakerProfile:
    """Configuration for a known speaker."""
    name: str
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
class AudioConfig:
    """Audio device configuration."""
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    show_interim_results: bool = True
    show_tts_chunks: bool = False
    show_audio_debug: bool = False

@dataclass
class CameraConfig:
    """Camera capture configuration."""
    enabled: bool = False
    device_id: int = 0
    resolution: List[int] = field(default_factory=lambda: [640, 480])
    capture_on_speech: bool = True
    save_captures: bool = False
    capture_dir: str = "camera_captures"
    jpeg_quality: int = 85

@dataclass
class EchoFilterConfig:
    """Echo filter configuration for preventing TTS feedback."""
    enabled: bool = True
    similarity_threshold: float = 0.75
    time_window: float = 15.0
    min_length: int = 3

@dataclass
class DevelopmentConfig:
    """Development and testing configuration."""
    enable_debug_mode: bool = False
    test_mode: bool = False
    mock_apis: bool = False

@dataclass
class VoiceAIConfig:
    """Complete voice AI system configuration."""
    conversation: ConversationConfig
    stt: STTConfig
    tts: TTSConfig
    audio: AudioConfig
    logging: LoggingConfig
    development: DevelopmentConfig
    speakers: SpeakersConfig = field(default_factory=SpeakersConfig)
    camera: Optional[CameraConfig] = None
    echo_filter: Optional[EchoFilterConfig] = None

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override values taking precedence.
    
    Args:
        base: Base dictionary
        override: Override dictionary with values to merge in
        
    Returns:
        Merged dictionary
    """
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result

class ConfigLoader:
    """Loads and validates configuration from YAML files."""
    
    DEFAULT_CONFIG_PATHS = [
        "config.yaml",
        "config/config.yaml",
        os.path.expanduser("~/.voiceai/config.yaml"),
        "/etc/voiceai/config.yaml"
    ]
    
    PRESET_DIRS = [
        "presets",
        "config/presets",
        os.path.expanduser("~/.voiceai/presets"),
        "/etc/voiceai/presets"
    ]
    
    @classmethod
    def load(cls, config_path: Optional[str] = None, preset: Optional[str] = None) -> VoiceAIConfig:
        """
        Load configuration from YAML file with optional preset overrides.
        
        Args:
            config_path: Path to config file. If None, searches default locations.
            preset: Name of preset to apply (without .yaml extension). 
                   Can also be set via VOICE_AI_PRESET environment variable.
            
        Returns:
            VoiceAIConfig: Complete configuration object
            
        Raises:
            FileNotFoundError: If no config file is found
            ValueError: If configuration is invalid
        """
        # Determine preset to use
        if preset is None:
            preset = os.environ.get('VOICE_AI_PRESET')
        
        # Find config file
        if config_path:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            config_file = cls._find_config_file()
            if not config_file:
                raise FileNotFoundError(
                    f"No config file found in default locations: {cls.DEFAULT_CONFIG_PATHS}"
                )
        
        # Load base YAML
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        
        # Apply preset overrides if specified
        if preset:
            preset_data = cls._load_preset(preset)
            config_data = deep_merge(config_data, preset_data)
        
        # Validate and create configurations
        return cls._create_config(config_data)
    
    @classmethod
    def _find_config_file(cls) -> Optional[Path]:
        """Find the first available config file from default paths."""
        for path in cls.DEFAULT_CONFIG_PATHS:
            config_file = Path(path)
            if config_file.exists():
                return config_file
        return None
    
    @classmethod
    def _find_preset_file(cls, preset_name: str) -> Path:
        """Find preset file in default preset directories."""
        preset_filename = f"{preset_name}.yaml"
        
        for preset_dir in cls.PRESET_DIRS:
            preset_path = Path(preset_dir) / preset_filename
            if preset_path.exists():
                return preset_path
        
        raise FileNotFoundError(
            f"Preset '{preset_name}' not found in any preset directory: {cls.PRESET_DIRS}"
        )
    
    @classmethod
    def _load_preset(cls, preset_name: str) -> Dict[str, Any]:
        """Load preset configuration from YAML file."""
        preset_file = cls._find_preset_file(preset_name)
        
        try:
            with open(preset_file, 'r') as f:
                preset_data = yaml.safe_load(f) or {}
            return preset_data
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in preset file '{preset_file}': {e}")
    
    @classmethod
    def _create_config(cls, config_data: Dict[str, Any]) -> VoiceAIConfig:
        """Create configuration objects from loaded YAML data."""
        
        # Validate required API keys
        api_keys = config_data.get('api_keys', {})
        required_keys = ['deepgram', 'elevenlabs', 'openai']
        missing_keys = [key for key in required_keys if not api_keys.get(key)]
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {missing_keys}")
        
        # Check for Anthropic key if Anthropic provider is selected
        conversation_config_data = config_data.get('conversation', {})
        if conversation_config_data.get('llm_provider') == 'anthropic' and not api_keys.get('anthropic'):
            raise ValueError("Anthropic API key is required when using Anthropic as LLM provider")
        
        # Extract configuration sections
        voice_config = config_data.get('voice', {})
        stt_config_data = config_data.get('stt', {})
        tts_config_data = config_data.get('tts', {})
        audio_config_data = config_data.get('audio', {})
        logging_config_data = config_data.get('logging', {})
        dev_config_data = config_data.get('development', {})
        tools_config = config_data.get('tools', {})
        characters_config = config_data.get('characters', {})
        camera_config_data = config_data.get('camera', {})
        director_config = config_data.get('director', {})
        echo_filter_data = config_data.get('echo_filter', {})
        
        # Parse keywords if provided
        keywords = None
        if 'keywords' in stt_config_data:
            keywords = []
            for kw in stt_config_data['keywords']:
                if isinstance(kw, dict):
                    keywords.append((kw['word'], kw.get('weight', 5.0)))
                elif isinstance(kw, str):
                    keywords.append((kw, 5.0))  # Default weight
                elif isinstance(kw, (list, tuple)) and len(kw) >= 2:
                    keywords.append((kw[0], float(kw[1])))
        
        # Create STT configuration
        stt_config = STTConfig(
            api_key=api_keys['deepgram'],
            model=stt_config_data.get('model', 'nova-3'),
            language=stt_config_data.get('language', 'en-US'),
            sample_rate=stt_config_data.get('sample_rate', 16000),
            chunk_size=stt_config_data.get('chunk_size', 8000),
            channels=1,  # Always mono for voice
            smart_format=True,  # Always enabled
            interim_results=stt_config_data.get('interim_results', True),
            punctuate=stt_config_data.get('punctuate', True),
            diarize=stt_config_data.get('diarize', True),
            utterance_end_ms=stt_config_data.get('utterance_end_ms', 1000),
            vad_events=stt_config_data.get('vad_events', True),
            enable_speaker_id=stt_config_data.get('enable_speaker_id', False),
            speaker_profiles_path=stt_config_data.get('speaker_profiles_path'),
            keywords=keywords,
            debug_speaker_data=stt_config_data.get('debug_speaker_data', False)
        )
        
        # Create TTS configuration
        tts_config = TTSConfig(
            api_key=api_keys['elevenlabs'],
            voice_id=voice_config.get('id', 'T2KZm9rWPG5TgXTyjt7E'),
            model_id=tts_config_data.get('model_id', 'eleven_multilingual_v2'),
            output_format=tts_config_data.get('output_format', 'pcm_22050'),
            sample_rate=tts_config_data.get('sample_rate', 22050),
            speed=tts_config_data.get('speed', 1.0),
            stability=tts_config_data.get('stability', 0.5),
            similarity_boost=tts_config_data.get('similarity_boost', 0.8),
            chunk_size=tts_config_data.get('chunk_size', 1024),
            buffer_size=tts_config_data.get('buffer_size', 2048),
            # Multi-voice support
            emotive_voice_id=voice_config.get('emotive_id'),
            emotive_speed=tts_config_data.get('emotive_speed', 1.0),
            emotive_stability=tts_config_data.get('emotive_stability', 0.5),
            emotive_similarity_boost=tts_config_data.get('emotive_similarity_boost', 0.8)
        )
        
        # Create conversation configuration
        conversation_config = ConversationConfig(
            deepgram_api_key=api_keys['deepgram'],
            elevenlabs_api_key=api_keys['elevenlabs'],
            openai_api_key=api_keys['openai'],
            anthropic_api_key=api_keys.get('anthropic', ''),
            voice_id=voice_config.get('id', 'T2KZm9rWPG5TgXTyjt7E'),
            pause_threshold=conversation_config_data.get('pause_threshold', 2.0),
            min_words_for_submission=conversation_config_data.get('min_words_for_submission', 3),
            max_wait_time=conversation_config_data.get('max_wait_time', 10.0),
            interruption_confidence=conversation_config_data.get('interruption_confidence', 0.8),
            stt_model=stt_config_data.get('model', 'nova-3'),
            stt_language=stt_config_data.get('language', 'en-US'),
            interim_results=stt_config_data.get('interim_results', True),
            tts_model=tts_config_data.get('model_id', 'eleven_multilingual_v2'),
            tts_speed=tts_config_data.get('speed', 1.0),
            tts_stability=tts_config_data.get('stability', 0.5),
            tts_similarity_boost=tts_config_data.get('similarity_boost', 0.8),
            llm_provider=conversation_config_data.get('llm_provider', 'openai'),
            llm_model=conversation_config_data.get('llm_model', 'chatgpt-4o-latest'),
            conversation_mode=conversation_config_data.get('conversation_mode', 'chat'),
            max_tokens=conversation_config_data.get('max_tokens', 300),
            system_prompt=conversation_config_data.get('system_prompt', 
                "You are a helpful AI assistant in a voice conversation. Give natural, conversational responses that work well when spoken aloud. Keep responses concise but engaging."),
            prefill_user_message=conversation_config_data.get('prefill_user_message', '<cmd>cat untitled.txt</cmd>'),
            prefill_participants=conversation_config_data.get('prefill_participants', ['H', 'Claude']),
            prefill_system_prompt=conversation_config_data.get('prefill_system_prompt', 
                'The assistant is in CLI simulation mode, and responds to the user\'s CLI commands only with outputs of the commands.'),
            history_file=conversation_config_data.get('history_file'),
            tools_config=tools_config,
            characters_config=characters_config,
            director_config=director_config,
        )
        
        # Create other configurations
        audio_config = AudioConfig(
            input_device_index=audio_config_data.get('input_device_index'),
            output_device_index=audio_config_data.get('output_device_index')
        )
        
        logging_config = LoggingConfig(
            level=logging_config_data.get('level', 'INFO'),
            show_interim_results=logging_config_data.get('show_interim_results', True),
            show_tts_chunks=logging_config_data.get('show_tts_chunks', False),
            show_audio_debug=logging_config_data.get('show_audio_debug', False)
        )
        
        development_config = DevelopmentConfig(
            enable_debug_mode=dev_config_data.get('enable_debug_mode', False),
            test_mode=dev_config_data.get('test_mode', False),
            mock_apis=dev_config_data.get('mock_apis', False)
        )
        
        # Create camera config if provided
        camera_config = None
        if camera_config_data:
            camera_config = CameraConfig(
                enabled=camera_config_data.get('enabled', False),
                device_id=camera_config_data.get('device_id', 0),
                resolution=camera_config_data.get('resolution', [640, 480]),
                capture_on_speech=camera_config_data.get('capture_on_speech', True),
                save_captures=camera_config_data.get('save_captures', False),
                capture_dir=camera_config_data.get('capture_dir', 'camera_captures'),
                jpeg_quality=camera_config_data.get('jpeg_quality', 85)
            )
        
        # Create echo filter config
        echo_filter_config = None
        if echo_filter_data or echo_filter_data is None:  # Create with defaults if not explicitly disabled
            echo_filter_config = EchoFilterConfig(
                enabled=echo_filter_data.get('enabled', True),
                similarity_threshold=echo_filter_data.get('similarity_threshold', 0.75),
                time_window=echo_filter_data.get('time_window', 15.0),
                min_length=echo_filter_data.get('min_length', 3)
            )
        
        # Create speakers config
        speakers_data = config_data.get('speakers', {})
        
        # Parse speaker profiles
        speaker_profiles = {}
        profiles_data = speakers_data.get('profiles', {})
        for profile_id, profile_data in profiles_data.items():
            speaker_profiles[profile_id] = SpeakerProfile(
                name=profile_data.get('name', profile_id),
                description=profile_data.get('description', ''),
                reference_audio=profile_data.get('reference_audio')
            )
        
        # Parse recognition settings
        recognition_data = speakers_data.get('recognition', {})
        recognition_config = SpeakerRecognitionConfig(
            confidence_threshold=recognition_data.get('confidence_threshold', 0.7),
            learning_mode=recognition_data.get('learning_mode', True),
            max_speakers=recognition_data.get('max_speakers', 4),
            voice_fingerprint_length=recognition_data.get('voice_fingerprint_length', 128)
        )
        
        speakers_config = SpeakersConfig(
            profiles=speaker_profiles,
            recognition=recognition_config
        )
        
        return VoiceAIConfig(
            conversation=conversation_config,
            stt=stt_config,
            tts=tts_config,
            audio=audio_config,
            logging=logging_config,
            development=development_config,
            speakers=speakers_config,
            camera=camera_config,
            echo_filter=echo_filter_config
        )
    
    @classmethod
    def create_example_config(cls, output_path: str = "config.yaml"):
        """Create an example configuration file."""
        example_config = {
            'api_keys': {
                'deepgram': 'your_deepgram_api_key_here',
                'elevenlabs': 'your_elevenlabs_api_key_here',
                'openai': 'your_openai_api_key_here'
            },
            'voice': {
                'id': 'T2KZm9rWPG5TgXTyjt7E'
            },
            'stt': {
                'model': 'nova-3',
                'language': 'en-US',
                'interim_results': True
            },
            'tts': {
                'model_id': 'eleven_multilingual_v2',
                'speed': 1.0,
                'stability': 0.5
            },
            'conversation': {
                'pause_threshold': 2.0,
                'min_words_for_submission': 3,
                'llm_model': 'chatgpt-4o-latest'
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Example config created: {output_path}")

# Convenience functions
def load_config(config_path: Optional[str] = None, preset: Optional[str] = None) -> VoiceAIConfig:
    """Load configuration from YAML file with optional preset overrides."""
    return ConfigLoader.load(config_path, preset)

def create_example_config(output_path: str = "config.yaml"):
    """Create an example configuration file."""
    ConfigLoader.create_example_config(output_path)

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create-example":
        output_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        create_example_config(output_path)
    else:
        try:
            config = load_config()
            print("✅ Configuration loaded successfully!")
            print(f"   - Deepgram API key: {'✓' if config.conversation.deepgram_api_key else '✗'}")
            print(f"   - ElevenLabs API key: {'✓' if config.conversation.elevenlabs_api_key else '✗'}")
            print(f"   - OpenAI API key: {'✓' if config.conversation.openai_api_key else '✗'}")
            print(f"   - Voice ID: {config.conversation.voice_id}")
            print(f"   - STT Model: {config.stt.model}")
            print(f"   - TTS Model: {config.tts.model_id}")
            print(f"   - LLM Model: {config.conversation.llm_model}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            print("\n💡 To create an example config file, run:")
            print("   python config_loader.py create-example") 