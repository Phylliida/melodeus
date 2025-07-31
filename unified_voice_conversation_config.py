#!/usr/bin/env python3
"""
Unified Voice Conversation System with YAML Configuration
Integrates modular STT and TTS systems with YAML-based configuration management.
"""

import asyncio
import re
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncGenerator, Union
from datetime import datetime, timedelta
from openai import OpenAI
import anthropic
from anthropic import AsyncAnthropic

# Import our modular systems and config loader
from async_stt_module import AsyncSTTStreamer, STTEventType, STTResult
from async_tts_module import AsyncTTSStreamer
from config_loader import load_config, VoiceAIConfig
from tools import create_tool_registry
from character_system import create_character_manager, CharacterManager
from camera_capture import CameraCapture, CameraConfig as CameraCaptureConfig
from thinking_sound import ThinkingSoundPlayer
from websocket_ui_server import VoiceUIServer
from echo_filter import EchoFilter

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    role: str  # "user", "assistant", "system", or "tool"
    content: Union[str, List[Dict[str, Any]]]  # String or list of content blocks (for images)
    timestamp: datetime
    status: str = "completed"  # "pending", "processing", "completed", "interrupted"
    speaker_id: Optional[int] = None
    character: Optional[str] = None  # Character name for multi-character conversations
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata for special handling

@dataclass 
class ConversationState:
    """Tracks the current state of the conversation."""
    is_active: bool = False
    is_processing_llm: bool = False
    is_speaking: bool = False
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    # Track what utterance is currently being processed
    current_processing_turn: Optional[ConversationTurn] = None
    # Track the current LLM streaming task so we can cancel it
    current_llm_task: Optional[asyncio.Task] = None
    # Track pending tool response to speak after interruption
    pending_tool_response: Optional[str] = None
    # Track request generation to handle concurrent requests
    request_generation: int = 0

class UnifiedVoiceConversation:
    """Unified voice conversation system with YAML configuration."""
    
    def __init__(self, config: VoiceAIConfig):
        self.config = config
        self.state = ConversationState()
        
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        self.async_anthropic_client = None
        
        # Track processing tasks to prevent race conditions
        self._processing_task = None
        self._processing_generation = 0
        # Track director requests separately
        self._director_generation = 0
        
        if config.conversation.llm_provider == "openai":
            self.openai_client = OpenAI(api_key=config.conversation.openai_api_key)
        elif config.conversation.llm_provider == "anthropic":
            if not config.conversation.anthropic_api_key:
                raise ValueError("Anthropic API key is required when using Anthropic provider")
            # Use async client for better responsiveness
            self.async_anthropic_client = AsyncAnthropic(api_key=config.conversation.anthropic_api_key)
        
        # Initialize STT system with participant names as keywords
        self._setup_stt_keywords(config)
        self.stt = AsyncSTTStreamer(config.stt)
        
        # Initialize TTS system
        self.tts = AsyncTTSStreamer(config.tts)
        
        # Initialize tool registry
        self.tool_registry = create_tool_registry(config.conversation.tools_config)
        
        # Initialize camera if enabled
        self.camera: Optional[CameraCapture] = None
        if config.camera and config.camera.enabled:
            camera_config = CameraCaptureConfig(
                device_id=config.camera.device_id,
                resolution=tuple(config.camera.resolution),
                capture_on_speech=config.camera.capture_on_speech,
                save_captures=config.camera.save_captures,
                capture_dir=config.camera.capture_dir,
                jpeg_quality=config.camera.jpeg_quality
            )
            self.camera = CameraCapture(camera_config)
            if self.camera.start():
                print("📷 Camera capture enabled")
            else:
                print("⚠️ Failed to start camera, continuing without camera capture")
                self.camera = None
        
        # Initialize character manager (always use character mode)
        characters_config = {
            "characters": config.conversation.characters_config or {},
            "director": config.conversation.director_config or {},
            "api_keys": {
                "openai": config.conversation.openai_api_key,
                "anthropic": config.conversation.anthropic_api_key,
                "deepgram": config.conversation.deepgram_api_key,
                "elevenlabs": config.conversation.elevenlabs_api_key
            }
        }
        
        # If no characters defined, create a default Assistant character
        if not characters_config["characters"]:
            characters_config["characters"] = {
                "Assistant": {
                    "llm_provider": config.conversation.llm_provider,
                    "llm_model": config.conversation.llm_model,
                    "voice_id": config.conversation.voice_id,
                    "voice_settings": {
                        "speed": config.tts.speed,
                        "stability": config.tts.stability,
                        "similarity_boost": config.tts.similarity_boost
                    },
                    "system_prompt": config.conversation.system_prompt,
                    "max_tokens": config.conversation.max_tokens,
                    "temperature": 0.7,  # Default temperature
                    "max_prompt_tokens": 8000  # Default token limit for prompts
                }
            }
            # Simple director for single character
            if not characters_config["director"]:
                characters_config["director"] = {
                    "llm_provider": "groq",
                    "llm_model": "llama-3.1-8b-instant",
                    "system_prompt": "You are directing a conversation. Since there is only one AI assistant, always respond with 'Assistant' when asked who should speak next."
                }
        
        self.character_manager: CharacterManager = create_character_manager(characters_config)
        
        # Initialize thinking sound player
        self.thinking_sound = ThinkingSoundPlayer(sample_rate=22050)
        
        # Initialize echo filter
        if config.echo_filter and config.echo_filter.enabled:
            self.echo_filter = EchoFilter(
                similarity_threshold=config.echo_filter.similarity_threshold,
                time_window=config.echo_filter.time_window,
                min_length=config.echo_filter.min_length
            )
        else:
            self.echo_filter = None
        
        # Initialize UI server
        ui_port = config.ui_port if hasattr(config, 'ui_port') else 8765
        self.ui_server = VoiceUIServer(self, host='localhost', port=ui_port)
        
        # Show appropriate message
        if len(characters_config["characters"]) > 1:
            print(f"🎭 Multi-character mode: {', '.join(characters_config['characters'].keys())}")
        else:
            print("🎭 Character mode enabled")
        
        # Set up tool execution callback
        self.tts.on_tool_execution = self._handle_tool_execution
        
        # Set up STT callbacks
        self._setup_stt_callbacks()
        
        # Apply logging configuration
        self._setup_logging()
        
        # Create llm_logs directory
        self.llm_logs_dir = Path("llm_logs")
        self.llm_logs_dir.mkdir(exist_ok=True)
        
        # Create conversation_logs directory and initialize conversation log
        self.conversation_logs_dir = Path("conversation_logs")
        self.conversation_logs_dir.mkdir(exist_ok=True)
        
        # Initialize conversation log file
        self._init_conversation_log()
        
        # Load conversation history if specified
        self._load_history_file()
        
        # Log any loaded history to the conversation log
        self._log_loaded_history()
        
        # Conversation management task
        self.conversation_task = None
    
    def _init_conversation_log(self):
        """Initialize conversation log file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_log_file = self.conversation_logs_dir / f"conversation_{timestamp}.md"
        
        # Write header
        with open(self.conversation_log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Conversation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        print(f"📝 Conversation logging to: {self.conversation_log_file}")
    
    def _log_conversation_turn(self, role: str, content: Union[str, List[Dict[str, Any]]]):
        """Log a conversation turn in history file format.
        
        Args:
            role: 'user' or 'assistant'
            content: The message content (may already include speaker prefix) or image content
        """
        try:
            # Handle image content
            if isinstance(content, list):
                with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                    for item in content:
                        if item.get("type") == "text":
                            f.write(f"{item.get('text', '')}\n\n")
                        elif item.get("type") == "image":
                            f.write(f"[Image: {item.get('source', {}).get('media_type', 'unknown')}]\n\n")
                return
                
            # Handle string content
            # Check if content already has a speaker prefix (for multi-character mode)
            if role == "assistant" and ": " in content and content.index(": ") < 50:
                # Content already includes character name, use as-is
                with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{content}\n\n")
            else:
                # Determine participant name based on role
                if role == "user":
                    participant = "H"
                else:  # assistant
                    # Use the AI participant name from prefill config if available
                    participant = "Claude"
                    if (hasattr(self.config.conversation, 'prefill_participants') and 
                        self.config.conversation.prefill_participants and 
                        len(self.config.conversation.prefill_participants) > 1):
                        participant = self.config.conversation.prefill_participants[1]
                
                # Append to conversation log
                with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{participant}: {content}\n\n")
                
        except Exception as e:
            print(f"⚠️ Failed to log conversation turn: {e}")
            self.logger.error(f"Failed to log conversation turn: {e}")
    
    def add_image_to_conversation(self, image_path: str, text: Optional[str] = None) -> bool:
        """Add an image to the conversation history.
        
        Args:
            image_path: Path to the image file
            text: Optional text to accompany the image
            
        Returns:
            bool: True if image was successfully added
        """
        try:
            # Read the image file
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                print(f"❌ Image file not found: {image_path}")
                return False
                
            # Determine media type
            suffix = image_path.suffix.lower()
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = media_type_map.get(suffix, 'image/jpeg')
            
            # Read image data
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Encode to base64
            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Create content blocks
            content = []
            if text:
                content.append({"type": "text", "text": text})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64
                }
            })
            
            # Add to conversation history
            image_turn = ConversationTurn(
                role="user",
                content=content,
                timestamp=datetime.now(),
                status="completed"
            )
            self.state.conversation_history.append(image_turn)
            
            # Log the image
            self._log_conversation_turn("user", content)
            
            print(f"📸 Added image to conversation: {image_path.name}")
            
            # Process pending utterances (which will include this image)
            asyncio.create_task(self._process_pending_utterances())
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to add image: {e}")
            self.logger.error(f"Failed to add image: {e}")
            return False
    
    def _log_loaded_history(self):
        """Log any existing conversation history to the conversation log."""
        if not self.state.conversation_history:
            return
            
        try:
            with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                f.write("## Loaded History\n\n")
                
            # Log each message from loaded history
            for turn in self.state.conversation_history:
                self._log_conversation_turn(turn.role, turn.content)
                
            with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                f.write("## New Conversation\n\n")
                
            print(f"📜 Logged {len(self.state.conversation_history)} history messages to conversation log")
            
        except Exception as e:
            print(f"⚠️ Failed to log loaded history: {e}")
            self.logger.error(f"Failed to log loaded history: {e}")
    
    def _setup_stt_keywords(self, config: VoiceAIConfig):
        """Setup STT keywords including character names."""
        keywords = []
        
        # Add character names from character configs
        if config.conversation.characters_config:
            for char_name, char_config in config.conversation.characters_config.items():
                # Add character key name with high weight
                keywords.append((char_name, 10.0))
                
                # Add prefill name if different
                if isinstance(char_config, dict) and 'prefill_name' in char_config:
                    prefill_name = char_config['prefill_name']
                    if prefill_name != char_name:
                        # For Nova-3, we can use multi-word keyterms
                        if config.stt.model == "nova-3":
                            keywords.append((prefill_name, 10.0))
                        else:
                            # For other models, only add individual words
                            for word in prefill_name.split():
                                if len(word) > 2:  # Skip short words
                                    keywords.append((word, 10.0))
        
        # Add prefill participant names from regular mode
        elif config.conversation.prefill_participants:
            for participant in config.conversation.prefill_participants:
                if config.stt.model == "nova-3":
                    # Nova-3 can handle full phrases
                    keywords.append((participant, 10.0))
                else:
                    # Other models: add individual words
                    for word in participant.split():
                        if len(word) > 2:  # Skip short words
                            keywords.append((word, 10.0))
        
        # Add any existing keywords from config
        if hasattr(config.stt, 'keywords') and config.stt.keywords:
            keywords.extend(config.stt.keywords)
        
        # Remove duplicates while keeping highest weight
        keyword_dict = {}
        for word, weight in keywords:
            if word not in keyword_dict or weight > keyword_dict[word]:
                keyword_dict[word] = weight
        
        # Convert back to list of tuples
        config.stt.keywords = [(word, weight) for word, weight in keyword_dict.items()]
        
        if config.stt.keywords:
            print(f"🔤 Added {len(config.stt.keywords)} keywords to STT including: {list(keyword_dict.keys())[:5]}")
    
    def _setup_logging(self):
        """Configure logging based on configuration."""
        
        # Set logging level
        level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.logger = logging.getLogger(__name__)
        
        # Store logging preferences
        self.show_interim = self.config.logging.show_interim_results
        self.show_tts_chunks = self.config.logging.show_tts_chunks
        self.show_audio_debug = self.config.logging.show_audio_debug
    
    def _generate_log_filename(self, log_type: str, timestamp: float = None) -> str:
        """Generate a unique filename for LLM logs."""
        if timestamp is None:
            timestamp = time.time()
        
        dt = datetime.fromtimestamp(timestamp)
        formatted_time = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        return f"{formatted_time}_{log_type}.json"
    
    def _sanitize_messages_for_logging(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a sanitized copy of messages suitable for logging.
        Replaces image data with metadata to avoid logging raw base64 data.
        """
        sanitized = []
        for msg in messages:
            sanitized_msg = {"role": msg["role"]}
            
            # Handle content
            content = msg.get("content", "")
            if isinstance(content, list):
                # Content is a list of blocks (text/image)
                sanitized_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            # Keep text as-is
                            sanitized_content.append(item)
                        elif item.get("type") == "image":
                            # Replace image data with metadata
                            source = item.get("source", {})
                            sanitized_content.append({
                                "type": "image",
                                "source": {
                                    "type": source.get("type", "unknown"),
                                    "media_type": source.get("media_type", "unknown"),
                                    "data": f"[BASE64_IMAGE_DATA - {len(source.get('data', ''))} chars]"
                                }
                            })
                        else:
                            # Keep other types as-is
                            sanitized_content.append(item)
                    else:
                        # Non-dict items, keep as-is
                        sanitized_content.append(item)
                sanitized_msg["content"] = sanitized_content
            else:
                # Simple string content
                sanitized_msg["content"] = content
            
            # Copy any metadata fields
            for key in ["_is_prefill", "_prefill_name"]:
                if key in msg:
                    sanitized_msg[key] = msg[key]
            
            sanitized.append(sanitized_msg)
        
        return sanitized
    
    def _log_llm_request(self, messages: List[Dict[str, str]], model: str, timestamp: float, provider: str = None) -> str:
        """Log LLM request to a file and return the filename."""
        filename = self._generate_log_filename("request", timestamp)
        filepath = self.llm_logs_dir / filename
        
        # Sanitize messages to avoid logging raw image data
        sanitized_messages = self._sanitize_messages_for_logging(messages)
        
        request_data = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "provider": provider or self.config.conversation.llm_provider,
            "model": model,
            "messages": sanitized_messages,
            "max_tokens": self.config.conversation.max_tokens,
            "stream": True,
            "conversation_mode": self.config.conversation.conversation_mode,
            "request_type": "llm_completion"
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(request_data, f, indent=2, ensure_ascii=False)
            print(f"📝 LLM request logged: {filename}")
        except Exception as e:
            print(f"❌ Failed to log LLM request: {e}")
        
        return filename
    
    def _log_llm_response(self, response_content: str, request_filename: str, timestamp: float, 
                         was_interrupted: bool = False, error: str = None, provider: str = None) -> str:
        """Log LLM response to a file and return the filename."""
        filename = self._generate_log_filename("response", timestamp)
        filepath = self.llm_logs_dir / filename
        
        response_data = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "provider": provider or self.config.conversation.llm_provider,
            "request_file": request_filename,
            "response_content": response_content,
            "was_interrupted": was_interrupted,
            "error": error,
            "content_length": len(response_content) if response_content else 0,
            "response_type": "llm_completion"
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            print(f"📝 LLM response logged: {filename}")
        except Exception as e:
            print(f"❌ Failed to log LLM response: {e}")
        
        return filename

    def _convert_to_prefill_format(self, messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], str]:
        """Convert chat messages to prefill format with image support.
        Returns (user_messages, assistant_message_prefix).
        
        Images are handled by splitting the conversation into blocks:
        - Text spans between images become text user messages
        - Images become image user messages  
        - Everything after the last image goes in the assistant message
        """
        human_name = self.config.conversation.prefill_participants[0]  # Default: 'H'
        ai_name = self.config.conversation.prefill_participants[1]     # Default: 'Claude'
        
        # Process messages to find images and create blocks
        user_messages = []
        current_text_block = []
        last_image_index = -1
        
        # Find the index of the last image
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # Check if this message contains an image
                for content_item in msg["content"]:
                    if content_item.get("type") == "image":
                        last_image_index = i
                        break
        
        # Process messages up to and including the last image
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                continue  # Skip system messages in prefill mode
                
            # Check if this is an image message
            is_image_message = False
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for content_item in msg["content"]:
                    if content_item.get("type") == "image":
                        is_image_message = True
                        break
            
            if is_image_message and i <= last_image_index:
                # First, add any accumulated text block
                if current_text_block:
                    text_content = "\n\n".join(current_text_block)
                    user_messages.append({"role": "user", "content": text_content})
                    current_text_block = []
                
                # Then add the image message as-is
                user_messages.append(msg)
            elif i <= last_image_index:
                # Regular text message before the last image - accumulate it
                if msg["role"] == "user":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        if not re.match(r'^[^:]+:', content):
                            content = f"{human_name}: {content}"
                        current_text_block.append(content)
                elif msg["role"] == "assistant":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        current_text_block.append(content)
        
        # Add any remaining text block before the last image
        if current_text_block and last_image_index >= 0:
            text_content = "\n\n".join(current_text_block)
            user_messages.append({"role": "user", "content": text_content})
            current_text_block = []
        
        # Build assistant content from everything after the last image
        assistant_turns = []
        for i, msg in enumerate(messages):
            if i > last_image_index:
                if msg["role"] == "system":
                    continue
                elif msg["role"] == "user":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        if not re.match(r'^[^:]+:', content):
                            content = f"{human_name}: {content}"
                        assistant_turns.append(content)
                elif msg["role"] == "assistant":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        assistant_turns.append(content)
        
        # If no images were found, put everything in assistant message
        if last_image_index == -1:
            for msg in messages:
                if msg["role"] == "system":
                    continue
                elif msg["role"] == "user":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        if not re.match(r'^[^:]+:', content):
                            content = f"{human_name}: {content}"
                        assistant_turns.append(content)
                elif msg["role"] == "assistant":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        assistant_turns.append(content)
        
        # Join assistant turns and add prefill
        assistant_content = "\n\n".join(assistant_turns)
        if assistant_content:
            assistant_content += f"\n\n{ai_name}:"
        else:
            assistant_content = f"{ai_name}:"
        
        # If no user messages were created, use the default prefill message
        if not user_messages:
            user_messages = [{"role": "user", "content": self.config.conversation.prefill_user_message}]
            
        return user_messages, assistant_content

    def _convert_from_prefill_format(self, prefill_response: str) -> str:
        """Extract the actual response from prefill format.
        Removes the participant prefix if present.
        """
        ai_name = self.config.conversation.prefill_participants[1]  # Default: 'Claude'
        
        # Remove leading AI name prefix if present
        if prefill_response.startswith(f"{ai_name}:"):
            return prefill_response[len(f"{ai_name}:"):].strip()
        
        return prefill_response.strip()

    def _parse_history_file(self, file_path: str) -> List[Dict[str, str]]:
        """Parse history file and convert to message format.
        
        Supports multi-speaker conversations. All speakers are preserved in the conversation
        history for LLM context, but for voice interaction only the configured participants
        are used for the actual conversation flow.
        
        Expected format:
        SpeakerName: message content
        
        AnotherSpeaker: response content
        
        Returns messages with 'character' field for multi-character conversations.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Get configured participant names for the current voice conversation
            human_name = self.config.conversation.prefill_participants[0]  # e.g., 'H'
            ai_name = self.config.conversation.prefill_participants[1]     # e.g., 'Claude 3 Opus'
            
            # Store all detected speakers for stop sequences
            self.detected_speakers = set()
            
            import re
            messages = []
            current_speaker = None
            current_content = []
            
            for line in content.split('\n'):
                line = line.strip()
                
                # Match speaker pattern: "SpeakerName: content" but be much more restrictive
                # Only match lines that look like actual speaker names, not formatting or bullet points
                speaker_match = re.match(r'^([A-Za-z0-9_\s\.]+):\s*(.*)', line)
                
                if speaker_match:
                    speaker_name = speaker_match.group(1).strip()
                    message_content = speaker_match.group(2).strip()
                    
                    # Filter out obvious non-speaker patterns
                    if self._is_valid_speaker_name(speaker_name):
                        # Track all detected speakers
                        self.detected_speakers.add(speaker_name)
                        
                        # Check if this is the same speaker continuing
                        if speaker_name == current_speaker:
                            # Same speaker - just add the line as continuation
                            if message_content:
                                current_content.append(f"{speaker_name}: {message_content}")
                            else:
                                current_content.append(f"{speaker_name}:")
                        else:
                            # Different speaker - save previous and start new
                            if current_speaker and current_content:
                                # Determine role based on configured participants
                                # For multi-speaker: classify as 'user' if it's the human participant,
                                # otherwise as 'assistant' to preserve the conversation context
                                role = "user" if current_speaker == human_name else "assistant"
                                
                                # In character mode, we need to preserve the character name separately
                                full_content = '\n'.join(current_content).strip()
                                
                                message = {
                                    "role": role,
                                    "content": full_content
                                }
                                
                                # Add character name if it's an assistant message and not the default AI
                                if role == "assistant" and current_speaker != ai_name:
                                    # Store the speaker name for later character matching
                                    message["_speaker_name"] = current_speaker
                                
                                messages.append(message)
                            
                            # Start new message
                            current_speaker = speaker_name
                            current_content = [message_content] if message_content else []
                    else:
                        # Invalid speaker - treat as continuation of current message or skip
                        if current_speaker:
                            # Add the full line including the colon as continuation
                            current_content.append(line)
                        # If no current speaker, just skip the line
                    
                elif line and current_speaker:
                    # Continue current message (current_speaker is already validated)
                    current_content.append(line)
                # Skip empty lines or lines without a speaker
            
            # Add final message if exists (current_speaker is already validated)
            if current_speaker and current_content:
                role = "user" if current_speaker == human_name else "assistant"
                full_content = '\n'.join(current_content).strip()
                
                message = {
                    "role": role,
                    "content": full_content
                }
                
                # Add character name if it's an assistant message and not the default AI
                if role == "assistant" and current_speaker != ai_name:
                    # Store the speaker name for later character matching
                    message["_speaker_name"] = current_speaker
                
                messages.append(message)
            
            print(f"📊 Detected speakers in history: {sorted(self.detected_speakers)}")
            return messages
            
        except Exception as e:
            print(f"❌ Failed to parse history file {file_path}: {e}")
            return []

    def _is_valid_speaker_name(self, speaker_name: str) -> bool:
        """Validate if a string looks like a legitimate speaker name."""
        # Exclude obvious non-speaker patterns
        invalid_patterns = [
            r'^\d+\.',  # Numbers like "1.", "2."
            r'^-\s',    # Bullet points like "- Panel"
            r'^\*\*',   # Bold markdown like "**Notable"
            r'^Step\s', # Step instructions
            r'^Panel\s', # Panel descriptions
            r'^\[',     # Bracket descriptions
            r'translation$', # Translation notes
            r'^The\s.*(part|truth|question|boundary)',  # Descriptive phrases
            r'^A\s(verse|poem)',  # Poetry descriptions
            r'Dynamics$',  # Ends with "Dynamics"
            r'Interactions$',  # Ends with "Interactions"
            r'Elements$',   # Ends with "Elements"
            r'Themes$',     # Ends with "Themes"
            r'^haha\s',     # Casual expressions
            r'https$',      # URLs
            r'Metaphors$',  # Ends with "Metaphors"
            r'Rigidity$',   # Ends with "Rigidity"
            r'Rejection$',  # Ends with "Rejection"
            r'Performance$', # Ends with "Performance"
            r'Engagement$',  # Ends with "Engagement"
        ]
        
        # Check against invalid patterns
        for pattern in invalid_patterns:
            if re.search(pattern, speaker_name, re.IGNORECASE):
                return False
        
        # Additional filters
        # Too long (probably a sentence)
        if len(speaker_name) > 50:
            return False
            
        # Contains too many words (probably descriptive text)
        if len(speaker_name.split()) > 4:
            return False
            
        # Contains certain punctuation that indicates it's not a name
        if any(char in speaker_name for char in ['*', '[', ']', '(', ')', '"', "'"]):
            return False
            
        return True

    def _load_history_file(self):
        """Load conversation history from file if specified."""
        if not self.config.conversation.history_file:
            return
            
        file_path = self.config.conversation.history_file
        print(f"📜 Loading conversation history from: {file_path}")
        
        history_messages = self._parse_history_file(file_path)
        
        if history_messages:
            # Convert old message format to ConversationTurn format
            for msg in history_messages:
                # Try to resolve character name from speaker name
                character_name = None
                if msg.get("_speaker_name") and hasattr(self, 'character_manager') and self.character_manager:
                    speaker = msg["_speaker_name"]
                    # Try to match by character name or prefill name
                    for char_name, char_config in self.character_manager.characters.items():
                        if (speaker == char_name or 
                            (char_config.prefill_name and speaker == char_config.prefill_name)):
                            character_name = char_name
                            break
                
                turn = ConversationTurn(
                    role=msg["role"],
                    content=msg["content"], 
                    timestamp=datetime.now(),  # We don't have original timestamps
                    status="completed",  # Historical messages are completed
                    character=character_name
                )
                self.state.conversation_history.append(turn)
                
                # Debug logging for character resolution
                if msg["role"] == "assistant" and msg.get("_speaker_name"):
                    print(f"   📝 {msg['_speaker_name']} → character: {character_name or 'None'}")
                
            print(f"✅ Loaded {len(history_messages)} messages from history file")
            print(f"📊 Conversation context: {sum(len(msg['content']) for msg in history_messages)} characters")
        else:
            print(f"⚠️  No messages loaded from history file")

    def _setup_stt_callbacks(self):
        """Set up callbacks for STT events."""
        
        # Handle completed utterances
        self.stt.on(STTEventType.UTTERANCE_COMPLETE, self._on_utterance_complete)
        
        # Handle interim results (for interruption detection)
        self.stt.on(STTEventType.INTERIM_RESULT, self._on_interim_result)
        
        # Handle speech events
        self.stt.on(STTEventType.SPEECH_STARTED, self._on_speech_started)
        self.stt.on(STTEventType.SPEECH_ENDED, self._on_speech_ended)
        
        # Handle speaker changes
        self.stt.on(STTEventType.SPEAKER_CHANGE, self._on_speaker_change)
        
        # Handle errors
        self.stt.on(STTEventType.ERROR, self._on_error)
    
    async def start_conversation(self):
        """Start the voice conversation system."""
        print("🎙️ Starting Unified Voice Conversation System (YAML Config)")
        print("=" * 60)
        
        # Show configuration summary
        self._show_config_summary()
        
        try:
            # Start STT
            print("🎤 Starting speech recognition...")
            if not await self.stt.start_listening():
                print("❌ Failed to start STT")
                return False
            
            self.state.is_active = True
            
            # Start conversation management
            self.conversation_task = asyncio.create_task(self._conversation_manager())
            
            # Start UI server
            asyncio.create_task(self.ui_server.start())
            
            print("✅ Conversation system active!")
            print("💡 Tips:")
            print("   - Speak naturally and pause when done")
            print("   - You can interrupt the AI by speaking while it talks")
            print("   - Press Ctrl+C to exit")
            print()
            
            # Keep running until stopped
            while self.state.is_active:
                await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting conversation: {e}")
            return False
    
    def _show_config_summary(self):
        """Show a summary of the loaded configuration."""
        print(f"📋 Configuration Summary:")
        print(f"   🎯 Voice ID: {self.config.conversation.voice_id}")
        print(f"   🎤 STT Model: {self.config.stt.model} ({self.config.stt.language})")
        print(f"   🔊 TTS Model: {self.config.tts.model_id}")
        print(f"   🤖 LLM Model: {self.config.conversation.llm_model}")
        print(f"   ⏱️  Pause Threshold: {self.config.conversation.pause_threshold}s")
        print(f"   📝 Min Words: {self.config.conversation.min_words_for_submission}")
        print(f"   🔇 Interruption Confidence: {self.config.conversation.interruption_confidence}")
        
        if self.config.development.enable_debug_mode:
            print(f"   🐛 Debug Mode: Enabled")
        
        print()
    
    async def stop_conversation(self):
        """Stop the conversation system."""
        print("🛑 Stopping conversation...")
        self.state.is_active = False
        
        # Cancel conversation management
        if self.conversation_task and not self.conversation_task.done():
            self.conversation_task.cancel()
            try:
                await self.conversation_task
            except asyncio.CancelledError:
                pass
        
        # Stop TTS if speaking
        if self.state.is_speaking:
            await self.tts.stop()
        
        # Stop STT
        await self.stt.stop_listening()
        
        print("✅ Conversation stopped")
    
    async def _on_utterance_complete(self, result: STTResult):
        """Handle completed utterances from STT."""
        speaker_info = f" (Speaker {result.speaker_id})" if result.speaker_id is not None else ""
        
        # Check if this is an echo of TTS output
        # Check echo filter
        is_echo = False
        matched_tts = None
        similarity = 0.0
        if self.echo_filter:
            is_echo, matched_tts, similarity = self.echo_filter.is_echo(result.text)
        
        if is_echo:
            print(f"🔇 Ignoring echo{speaker_info}: {result.text} (matched {similarity:.0%})")
            # Still broadcast to UI but mark as echo
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_transcription(
                    speaker="USER",
                    text=f"[Echo filtered] {result.text}",
                    is_final=True
                )
            return  # Don't process as real user input
            
        print(f"🎯 Final{speaker_info}: {result.text}")
        
        # Broadcast final transcription
        if hasattr(self, 'ui_server'):
            await self.ui_server.broadcast_transcription(
                speaker="USER",
                text=result.text,
                is_final=True
            )
        
        # Capture image if camera is enabled
        captured_image = None
        if self.camera and self.config.camera.capture_on_speech:
            capture_result = self.camera.capture_image()
            if capture_result:
                _, jpeg_base64 = capture_result
                captured_image = jpeg_base64
                print("📸 Captured image with user speech")
        
        # Check for interruption at ANY stage of AI response
        interrupted = False
        
        if self.tts.is_currently_playing():
            print(f"🛑 Interrupting TTS playback with: {result.text}")
            
            # Stop TTS and wait for Whisper to finish processing
            await self.tts.stop()
            
            # Get spoken content synchronously (available after stop() completes)
            spoken_content = self.tts.get_spoken_text_heuristic().strip()
            
            # Don't add assistant response here - character processing will handle it
            # This avoids duplicates
            interrupted = True
            
        elif self.state.is_processing_llm:
            print(f"🛑 Interrupting LLM generation with: {result.text}")
            # Mark current processing as interrupted
            if self.state.current_processing_turn:
                self.state.current_processing_turn.status = "interrupted"
            interrupted = True
            # Stop any ongoing TTS that might be starting
            if self.tts.is_streaming:
                await self.tts.stop()
            
        elif self.state.is_speaking:
            print(f"🛑 Interrupting TTS setup with: {result.text}")
            await self.tts.stop()
            interrupted = True
        
        if interrupted:
            # Cancel any ongoing LLM streaming task BEFORE adding the user utterance
            if self.state.current_llm_task and not self.state.current_llm_task.done():
                print("🚫 Cancelling LLM streaming task")
                self.state.current_llm_task.cancel()
                # Wait a moment for cancellation to take effect and history to be updated
                try:
                    await asyncio.wait_for(self.state.current_llm_task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                self.state.current_llm_task = None
            
            # Clear processing state
            self.state.is_speaking = False
            self.state.is_processing_llm = False
            self.state.current_processing_turn = None
            print("🔄 All AI processing interrupted and state cleared")
            
            # Broadcast state cleared
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_speaker_status(
                    current_speaker=None,
                    is_speaking=False,
                    is_processing=False,
                    thinking_sound=False
                )
        
        # Add the new user utterance to conversation history AFTER handling interruption
        # Include captured image if available
        if captured_image:
            # Create content with both text and image
            content = [
                {"type": "text", "text": result.text},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": captured_image
                    }
                }
            ]
        else:
            content = result.text
            
        user_turn = ConversationTurn(
            role="user",
            content=content,
            timestamp=result.timestamp,
            status="pending",
            speaker_id=result.speaker_id
        )
        self.state.conversation_history.append(user_turn)
        self._log_conversation_turn("user", content)
        print(f"💬 Added user utterance: '{result.text}'" + (" with image" if captured_image else ""))
        
        # Process the new utterance immediately
        print("🚀 Processing utterance")
        asyncio.create_task(self._process_pending_utterances())
        
        if self.config.development.enable_debug_mode:
            self.logger.debug(f"Utterance added to history")
    
    async def _on_interim_result(self, result: STTResult):
        """Handle interim results for showing progress."""
        # Interruption is now handled only in _on_utterance_complete for final utterances
        # This is just for showing interim results
        
        # Show interim results based on configuration
        if self.show_interim:
            print(f"💭 Interim: {result.text}")
            
        # Broadcast to UI
        if hasattr(self, 'ui_server'):
            await self.ui_server.broadcast_transcription(
                speaker="USER",
                text=result.text,
                is_interim=True
            )
    
    async def _on_speech_started(self, data):
        """Handle speech start events."""
        # Just show detection, don't interrupt yet - wait for complete utterance
        if self.tts.is_currently_playing():
            print("🚫 Speech detected during TTS - waiting for complete utterance")
        elif self.state.is_speaking:
            print("🚫 Speech detected during conversation processing")
        
        if self.show_audio_debug:
            self.logger.debug("Speech started event received")
    
    async def _on_speech_ended(self, data):
        """Handle speech end events."""
        print("🔇 Speech ended")
        self.state.last_utterance_time = data['timestamp']
        
        if self.show_audio_debug:
            self.logger.debug("Speech ended event received")
    
    async def _on_speaker_change(self, data):
        """Handle speaker change events."""
        speaker_id = data['speaker_id']
        speaker_name = data.get('speaker_name', f"Speaker {speaker_id}")
        print(f"👤 Speaker changed to: {speaker_name}")
        
        if self.config.development.enable_debug_mode:
            self.logger.debug(f"Speaker change: {speaker_id} -> {speaker_name}")
    
    async def _on_error(self, data):
        """Handle STT errors."""
        error_msg = data['error']
        print(f"❌ STT Error: {error_msg}")
        self.logger.error(f"STT Error: {error_msg}")
        
        # Check if this is a connection error that might require restart
        if "Connection send failed" in error_msg or "ConnectionClosed" in error_msg:
            print("🔄 Detected connection failure, attempting to restart STT...")
            try:
                # Try to restart STT connection
                await self.stt.stop_listening()
                await asyncio.sleep(0.01)  # Minimal pause
                success = await self.stt.start_listening()
                if success:
                    print("✅ STT connection restarted successfully")
                else:
                    print("❌ Failed to restart STT connection")
                    await self._speak_text("I'm having trouble with speech recognition. Please try again.")
            except Exception as e:
                print(f"❌ Error restarting STT: {e}")
                self.logger.error(f"STT restart error: {e}")
    
    async def _handle_tool_execution(self, tool_call):
        """Handle tool execution callback from TTS module."""
        from async_tts_module import ToolCall, ToolResult
        
        print(f"🔧 Executing tool: {tool_call.tag_name}")
        print(f"   Content: {tool_call.content}")
        print(f"   Position: {tool_call.start_position}-{tool_call.end_position}")
        
        try:
            # Create context for tool execution
            context = {
                'conversation_history': self.state.conversation_history,
                'current_session': self.tts.current_session,
                'config': self.config
            }
            
            # Execute tool using registry
            result = await self.tool_registry.execute_tool(tool_call, context)
            
            # Log tool execution result to conversation history
            if result.content:
                tool_turn = ConversationTurn(
                    role="tool",
                    content=f"[{tool_call.tag_name}] {result.content}",
                    timestamp=datetime.now(),
                    status="completed"
                )
                self.state.conversation_history.append(tool_turn)
                self._log_conversation_turn("tool", tool_turn.content)
            
            # If tool wants to interrupt and has content, we may need to speak it
            if result.should_interrupt and result.content:
                # Store for later processing after current TTS is interrupted
                self.state.pending_tool_response = result.content
            
            return result
                    
        except Exception as e:
            print(f"❌ Error executing tool: {e}")
            self.logger.error(f"Tool execution error: {e}")
            return ToolResult(should_interrupt=False, content=None)
    
    async def _process_pending_utterances(self):
        """Process pending utterances from conversation history."""
        # Increment generation to invalidate any previous processing
        self._processing_generation += 1
        current_generation = self._processing_generation
        
        # Cancel any existing processing task
        if self._processing_task and not self._processing_task.done():
            print("🚫 Cancelling previous processing task")
            self._processing_task.cancel()
            # Don't wait for it - let it cancel in background
            
        # Create new processing task
        self._processing_task = asyncio.create_task(
            self._do_process_pending_utterances(current_generation)
        )
        
    async def _do_process_pending_utterances(self, generation: int):
        """Actually process pending utterances with generation tracking."""
        try:
            # Find pending user utterances
            pending_turns = [turn for turn in self.state.conversation_history 
                            if turn.role == "user" and turn.status == "pending"]
            
            if not pending_turns:
                return
                
            # Check if we're still the current generation
            if generation != self._processing_generation:
                print(f"🚫 Processing cancelled - newer request exists (gen {generation} vs {self._processing_generation})")
                return
                
            # Process all pending turns
            # Extract text content for display
            text_contents = []
            for turn in pending_turns:
                if isinstance(turn.content, str):
                    text_contents.append(turn.content)
                elif isinstance(turn.content, list):
                    # Extract text from content blocks
                    for item in turn.content:
                        if item.get("type") == "text":
                            text_contents.append(item.get("text", ""))
            
            combined_text = " ".join(text_contents)
            print(f"🧠 Processing {len(pending_turns)} pending utterances: '{combined_text}'")
            
            # Mark all pending turns as processing
            for turn in pending_turns:
                turn.status = "processing"
            
            # Check generation again before processing
            if generation != self._processing_generation:
                print(f"🚫 Processing cancelled before LLM - newer request exists")
                # Reset status for all turns
                for turn in pending_turns:
                    turn.status = "pending"
                return
                
            # For now, pass the combined text. The actual content (including images) 
            # will be properly handled when building messages for the LLM
            await self._process_with_llm(combined_text, pending_turns[0], generation)  # Pass first turn for reference
            
        except asyncio.CancelledError:
            print("🚫 Processing task cancelled")
            # Reset status for pending turns
            for turn in self.state.conversation_history:
                if turn.role == "user" and turn.status == "processing":
                    turn.status = "pending"
            raise
        except Exception as e:
            print(f"❌ Error in processing: {e}")
            import traceback
            traceback.print_exc()
    
    async def _process_with_character_llm(self, user_input: str, reference_turn: ConversationTurn, generation: int = None):
        """Process user input with multi-character system."""
        try:
            self.state.is_processing_llm = True
            
            # Check generation early
            if generation is not None and generation != self._processing_generation:
                print(f"🚫 Character LLM processing cancelled - newer request exists")
                self.state.is_processing_llm = False
                return  # Don't stop sound here - it hasn't started yet
            
            # Increment director generation and store it
            self._director_generation += 1
            director_gen = self._director_generation
            
            # Start thinking sound with director generation
            await self.thinking_sound.start(generation=director_gen)
            
            # Broadcast thinking sound status
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_speaker_status(thinking_sound=True)
            
            # Check if this is a manual trigger from UI
            next_speaker = None
            if hasattr(reference_turn, 'metadata') and reference_turn.metadata:
                if reference_turn.metadata.get('is_manual_trigger'):
                    next_speaker = reference_turn.metadata.get('triggered_speaker')
                    print(f"🎯 Using manually triggered speaker: {next_speaker}")
            
            # If no manual trigger, let director decide who speaks next
            if not next_speaker:
                next_speaker = await self.character_manager.select_next_speaker(
                    self._get_conversation_history_for_director()
                )
            
            # Broadcast pending speaker
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_speaker_status(
                    pending_speaker=next_speaker,
                    is_processing=True
                )
            
            # Check both processing generation and director generation
            if generation is not None and generation != self._processing_generation:
                print(f"🚫 Processing cancelled after director - newer utterance exists")
                await self.thinking_sound.stop(generation=director_gen)
                self.state.is_processing_llm = False
                return
                
            if director_gen != self._director_generation:
                print(f"🚫 Processing cancelled after director - newer director request exists")
                await self.thinking_sound.stop(generation=director_gen)
                self.state.is_processing_llm = False
                return
            
            if next_speaker == "USER" or next_speaker is None:
                print("🎭 Director: User should speak next")
                await self.thinking_sound.stop(generation=director_gen)
                self.state.is_processing_llm = False
                return
            
            # Set active character
            self.character_manager.set_active_character(next_speaker)
            character_config = self.character_manager.get_character_config(next_speaker)
            
            if not character_config:
                print(f"❌ Unknown character: {next_speaker}")
                await self.thinking_sound.stop(generation=director_gen)
                self.state.is_processing_llm = False
                return
            
            print(f"🎭 {next_speaker} is responding...")
            
            # Broadcast current speaker
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_speaker_status(
                    current_speaker=next_speaker,
                    is_speaking=False,  # Not speaking yet, just processing
                    is_processing=True
                )
            
            # Request timestamp for logging
            request_timestamp = time.time()
            
            # Check if we should use prefill format
            use_prefill = (self.config.conversation.conversation_mode == "prefill" and 
                          character_config.llm_provider == "anthropic")
            
            if use_prefill:
                # For prefill mode, convert directly from raw history
                prefill_name = character_config.prefill_name or character_config.name
                raw_history = self._get_conversation_history_for_character()
                
                # Create messages in prefill format directly
                messages = self._create_character_prefill_messages(
                    raw_history, 
                    next_speaker, 
                    prefill_name,
                    character_config.system_prompt
                )
            else:
                # Standard chat format
                messages = self.character_manager.format_messages_for_character(
                    next_speaker,
                    self._get_conversation_history_for_character()
                )
            
            # Log the request
            request_filename = self._log_llm_request(
                messages, 
                character_config.llm_model, 
                request_timestamp,
                character_config.llm_provider
            )
            
            # Check both generations before starting LLM
            if generation is not None and generation != self._processing_generation:
                print(f"🚫 Processing cancelled before LLM call - newer utterance exists")
                await self.thinking_sound.stop(generation=director_gen)
                self.state.is_processing_llm = False
                return
                
            if director_gen != self._director_generation:
                print(f"🚫 Processing cancelled before LLM call - newer director request exists")
                await self.thinking_sound.stop(generation=director_gen)
                self.state.is_processing_llm = False
                return
            
            # Set callback to stop thinking sound when first audio arrives
            # Create a closure that captures the current director generation and speaker
            async def stop_thinking_for_generation():
                await self.thinking_sound.stop(generation=director_gen)
                if hasattr(self, 'ui_server'):
                    await self.ui_server.broadcast_speaker_status(
                        thinking_sound=False,
                        is_speaking=True,  # Now actually speaking
                        current_speaker=next_speaker
                    )
            
            self.tts.first_audio_callback = stop_thinking_for_generation
            
            # Stream response based on provider
            assistant_response = ""
            
            if character_config.llm_provider == "openai":
                # Temporarily set character voice
                original_config = self._set_character_voice(character_config)
                try:
                    # Create TTS task for this character
                    self.state.current_llm_task = asyncio.create_task(
                        self.tts.speak_stream_multi_voice(
                            self._stream_character_openai_response(messages, character_config, request_timestamp)
                        )
                    )
                    
                    # Wait for completion
                    try:
                        completed = await self.state.current_llm_task
                    except asyncio.CancelledError:
                        print("⚠️ Character response was interrupted")
                        completed = False
                    
                    # Update echo filter with completion status
                    session_id = f"openai_{character_config.name}_{request_timestamp}"
                    if completed:
                        if self.echo_filter:
                            self.echo_filter.on_tts_complete(session_id)
                    else:
                        # Get spoken text if interrupted
                        spoken_text = None
                        if hasattr(self.tts, 'get_spoken_text_heuristic'):
                            spoken_text = self.tts.get_spoken_text_heuristic()
                        if self.echo_filter:
                            self.echo_filter.on_tts_interrupted(session_id, spoken_text)
                    
                    # Get the appropriate text based on whether it was interrupted
                    if hasattr(self.tts, 'current_session') and self.tts.current_session:
                        if completed:
                            # Use full generated text for completed responses
                            assistant_response = self.tts.current_session.generated_text.strip()
                        else:
                            # Use spoken heuristic for interrupted responses
                            assistant_response = self.tts.get_spoken_text_heuristic().strip()
                        print(f"📝 Captured assistant response: {len(assistant_response)} chars")
                    else:
                        print("⚠️ No TTS session or generated text available")
                finally:
                    # Restore original voice config
                    self._restore_voice_config(original_config)
            elif character_config.llm_provider == "anthropic":
                # Temporarily set character voice
                original_config = self._set_character_voice(character_config)
                try:
                    # Create TTS task for this character
                    self.state.current_llm_task = asyncio.create_task(
                        self.tts.speak_stream_multi_voice(
                            self._stream_character_anthropic_response(messages, character_config, request_timestamp)
                        )
                    )
                    
                    # Wait for completion
                    try:
                        completed = await self.state.current_llm_task
                    except asyncio.CancelledError:
                        print("⚠️ Character response was interrupted")
                        completed = False
                    
                    # Update echo filter with completion status
                    session_id = f"anthropic_{character_config.name}_{request_timestamp}"
                    if completed:
                        if self.echo_filter:
                            self.echo_filter.on_tts_complete(session_id)
                    else:
                        # Get spoken text if interrupted
                        spoken_text = None
                        if hasattr(self.tts, 'get_spoken_text_heuristic'):
                            spoken_text = self.tts.get_spoken_text_heuristic()
                        if self.echo_filter:
                            self.echo_filter.on_tts_interrupted(session_id, spoken_text)
                    
                    # Get the appropriate text based on whether it was interrupted
                    if hasattr(self.tts, 'current_session') and self.tts.current_session:
                        if completed:
                            # Use full generated text for completed responses
                            assistant_response = self.tts.current_session.generated_text.strip()
                        else:
                            # Use spoken heuristic for interrupted responses
                            assistant_response = self.tts.get_spoken_text_heuristic().strip()
                        print(f"📝 Captured assistant response: {len(assistant_response)} chars")
                    else:
                        print("⚠️ No TTS session or generated text available")
                finally:
                    # Restore original voice config
                    self._restore_voice_config(original_config)
            
            # Log the response
            if assistant_response.strip():
                response_timestamp = time.time()
                self._log_llm_response(
                    assistant_response,
                    request_filename,
                    response_timestamp,
                    was_interrupted=False,
                    error=None,
                    provider=character_config.llm_provider
                )
                print(f"✅ Logged assistant response: {len(assistant_response)} chars")
            else:
                print(f"⚠️ No assistant response to log (empty or whitespace)")
            
            # Add character's response to conversation history
            if assistant_response.strip():
                print(f"💬 Adding assistant response to history: {len(assistant_response)} chars from {next_speaker}")
                # Determine status based on completion
                status = "completed" if completed else "interrupted"
                assistant_turn = ConversationTurn(
                    role="assistant",
                    content=assistant_response,
                    timestamp=datetime.now(),
                    status=status,
                    character=next_speaker
                )
                self.state.conversation_history.append(assistant_turn)
                # For multi-character mode, log with character name prefix (no brackets)
                self._log_conversation_turn("assistant", f"{next_speaker}: {assistant_response}")
                print(f"✅ Added assistant response to conversation history")
                
                # If the response was interrupted, add a system message
                if status == "interrupted":
                    system_message = f"[{next_speaker} was interrupted by user speaking]"
                    system_turn = ConversationTurn(
                        role="system",
                        content=system_message,
                        timestamp=datetime.now(),
                        status="completed"
                    )
                    self.state.conversation_history.append(system_turn)
                    self._log_conversation_turn("system", system_message)
                    print(f"📝 Added interruption notice to conversation history")
                
                # Add to character manager context
                self.character_manager.add_turn_to_context(next_speaker, assistant_response)
            else:
                print(f"⚠️ Skipping empty assistant response")
            
            # After character speaks, check if another character should speak
            await asyncio.sleep(0.5)  # Brief pause
            
            # Recursively check for next speaker (with depth limit)
            if not hasattr(self, '_character_depth'):
                self._character_depth = 0
            
            self._character_depth += 1
            if self._character_depth < 3:  # Max 3 characters in a row
                await self._process_with_character_llm("", reference_turn)
            else:
                self._character_depth = 0
                
        except Exception as e:
            print(f"❌ Character LLM error: {e}")
            self.logger.error(f"Character LLM error: {e}")
        finally:
            # Always stop thinking sound (no generation means force stop)
            await self.thinking_sound.stop()
            self.state.is_processing_llm = False
            
            # Broadcast that speaking/processing has ended
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_speaker_status(
                    current_speaker=None,
                    is_speaking=False,
                    is_processing=False,
                    thinking_sound=False
                )
    
    async def _stream_character_openai_response(self, messages, character_config, request_timestamp):
        """Stream response from OpenAI for a specific character."""
        client = self.character_manager.get_character_client(character_config.name)
        
        # Create session ID for echo tracking
        session_id = f"openai_{character_config.name}_{request_timestamp}"
        if self.echo_filter:
            self.echo_filter.on_tts_start(session_id, character_config.name)
        
        try:
            response = await client.chat.completions.create(
                model=character_config.llm_model,
                messages=messages,
                stream=True,
                max_tokens=character_config.max_tokens,
                temperature=character_config.temperature
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # Track for echo filter
                    if self.echo_filter:
                        self.echo_filter.on_tts_chunk(session_id, content)
                    yield content
                    # Broadcast to UI
                    if hasattr(self, 'ui_server'):
                        await self.ui_server.broadcast_ai_stream(
                            speaker=character_config.name,
                            text=content,
                            session_id=f"session_{request_timestamp}"
                        )
                    
        except Exception as e:
            print(f"❌ OpenAI streaming error for {character_config.name}: {e}")
            raise
    
    async def _stream_character_anthropic_response(self, messages, character_config, request_timestamp):
        """Stream response from Anthropic for a specific character."""
        client = self.character_manager.get_character_client(character_config.name)
        
        # Create session ID for echo tracking
        session_id = f"anthropic_{character_config.name}_{request_timestamp}"
        if self.echo_filter:
            self.echo_filter.on_tts_start(session_id, character_config.name)
        
        try:
            # Check if messages are already in prefill format
            is_prefill_format = False
            prefill_name = None
            
            # Check if last message is assistant (typical prefill pattern)
            if len(messages) > 0:
                last_msg = messages[-1]
                if last_msg.get("role") == "assistant":
                    is_prefill_format = True
                    # Extract prefill name from the assistant message content
                    content = last_msg.get("content", "")
                    if content.endswith(":"):
                        # Extract the name before the final colon
                        parts = content.rsplit("\n", 1)
                        if len(parts) > 0:
                            last_line = parts[-1].strip()
                            if last_line.endswith(":"):
                                prefill_name = last_line[:-1]
            
            messages_to_send = messages
            
            # Extract system content and prepare messages
            system_content = ""
            anthropic_messages = []
            
            for msg in messages_to_send:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    # Create a clean copy without metadata
                    clean_msg = {"role": msg["role"], "content": msg["content"]}
                    anthropic_messages.append(clean_msg)
            
            # Generate stop sequences for multi-character conversations
            stop_sequences = []
            if is_prefill_format:
                # Add all character names as stop sequences
                for char_name in self.character_manager.characters.keys():
                    stop_sequences.append(f"\n\n{char_name}:")
                    # Also add prefill names if different
                    char_config = self.character_manager.get_character_config(char_name)
                    if char_config and char_config.prefill_name and char_config.prefill_name != char_name:
                        stop_sequences.append(f"\n\n{char_config.prefill_name}:")
                
                # Add human names
                if hasattr(self.config.conversation, 'prefill_participants'):
                    human_name = self.config.conversation.prefill_participants[0]
                    stop_sequences.append(f"\n\n{human_name}:")
                
                # Add any detected speakers
                if hasattr(self, 'detected_speakers') and self.detected_speakers:
                    for speaker in self.detected_speakers:
                        speaker_stop = f"\n\n{speaker}:"
                        if speaker_stop not in stop_sequences:
                            stop_sequences.append(speaker_stop)
                
                # Add System: to stop sequences
                stop_sequences.append("\n\nSystem:")
                
                # Remove duplicates while preserving order
                stop_sequences = list(dict.fromkeys(stop_sequences))
                print(f"🛑 Character using stop sequences: {stop_sequences}")
            
            response = await client.messages.create(
                model=character_config.llm_model,
                messages=anthropic_messages,
                system=system_content,
                stream=True,
                max_tokens=character_config.max_tokens,
                temperature=character_config.temperature,
                stop_sequences=stop_sequences if stop_sequences else None
            )
            
            # Track if we need to skip the prefill prefix
            skip_prefix = is_prefill_format
            prefix_buffer = "" if skip_prefix else None
            
            async for chunk in response:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    text = chunk.delta.text
                    
                    # Skip the character name prefix in prefill mode
                    if skip_prefix and prefix_buffer is not None:
                        prefix_buffer += text
                        # Check if we've seen the full prefix
                        expected_prefix = f"{prefill_name}: "
                        if len(prefix_buffer) >= len(expected_prefix):
                            # We've seen enough, check if it matches
                            if prefix_buffer.startswith(expected_prefix):
                                # Skip the prefix, yield the rest
                                text_to_yield = prefix_buffer[len(expected_prefix):]
                                if self.echo_filter:
                                    self.echo_filter.on_tts_chunk(session_id, text_to_yield)
                                yield text_to_yield
                            else:
                                # Doesn't match expected prefix, yield everything
                                if self.echo_filter:
                                    self.echo_filter.on_tts_chunk(session_id, prefix_buffer)
                                yield prefix_buffer
                            prefix_buffer = None  # Stop checking
                        # else continue buffering
                    else:
                        # Track for echo filter
                        if self.echo_filter:
                            self.echo_filter.on_tts_chunk(session_id, text)
                        yield text
                        # Broadcast to UI
                        if hasattr(self, 'ui_server'):
                            await self.ui_server.broadcast_ai_stream(
                                speaker=character_config.name,
                                text=text,
                                session_id=f"session_{request_timestamp}"
                            )
                    
        except Exception as e:
            print(f"❌ Anthropic streaming error for {character_config.name}: {e}")
            raise
    
    def _get_conversation_history_for_director(self):
        """Get conversation history formatted for director.
        Excludes images since director only needs text to decide who speaks next.
        """
        history = []
        for turn in self.state.conversation_history[-20:]:  # Last 20 turns
            # Include both completed and interrupted turns
            if turn.status in ["completed", "interrupted"]:
                # Extract text content only
                if isinstance(turn.content, str):
                    content = turn.content
                elif isinstance(turn.content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for item in turn.content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    content = " ".join(text_parts)
                else:
                    content = str(turn.content)
                
                entry = {
                    "role": turn.role,
                    "content": content
                }
                if turn.character:
                    entry["character"] = turn.character
                history.append(entry)
        return history
    
    def _set_character_voice(self, character_config):
        """Set TTS voice for a specific character."""
        if not character_config:
            return None
            
        voice_settings = self.character_manager.get_character_voice_settings(
            character_config.name
        )
        
        # Save original config
        original_config = {
            "voice_id": self.tts.config.voice_id,
            "speed": self.tts.config.speed,
            "stability": self.tts.config.stability,
            "similarity_boost": self.tts.config.similarity_boost
        }
        
        # Apply character voice settings
        if voice_settings.get("voice_id"):
            self.tts.config.voice_id = voice_settings["voice_id"]
            self.tts.config.speed = voice_settings.get("speed", 1.0)
            self.tts.config.stability = voice_settings.get("stability", 0.5)
            self.tts.config.similarity_boost = voice_settings.get("similarity_boost", 0.8)
            
            print(f"🎤 Set voice for {character_config.name}: {voice_settings['voice_id']}")
        
        return original_config
    
    def _restore_voice_config(self, original_config):
        """Restore original TTS voice configuration."""
        if original_config:
            self.tts.config.voice_id = original_config["voice_id"]
            self.tts.config.speed = original_config["speed"]
            self.tts.config.stability = original_config["stability"]
            self.tts.config.similarity_boost = original_config["similarity_boost"]
    
    def _create_character_prefill_messages(self, raw_history: List[Dict[str, Any]], character_name: str, 
                                          prefill_name: str, system_prompt: str) -> List[Dict[str, Any]]:
        """Create prefill format messages directly from raw conversation history for a character.
        
        This creates the proper prefill structure with images:
        1. System message
        2. User messages with pre-image history 
        3. User messages with images
        4. Assistant message with post-image history and character's prefill
        """
        # Find last image in history
        last_image_index = -1
        for i, turn in enumerate(raw_history):
            if turn.get("role") == "user" and isinstance(turn.get("content"), list):
                for item in turn["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        last_image_index = i
                        break
        
        # Build prefill messages
        if last_image_index >= 0:
            # Has images - use image-aware format
            user_messages = []
            text_blocks = []
            
            # Process messages up to and including last image
            for i, turn in enumerate(raw_history[:last_image_index + 1]):
                role = turn.get("role")
                content = turn.get("content")
                character = turn.get("character")
                
                # Check if this turn has an image
                has_image = False
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            has_image = True
                            break
                
                if has_image:
                    # First, add any accumulated text as a user message
                    if text_blocks:
                        user_messages.append({"role": "user", "content": "\n\n".join(text_blocks)})
                        text_blocks = []
                    
                    # Then add the image message
                    user_messages.append({"role": "user", "content": content})
                else:
                    # Regular text message - format and accumulate
                    formatted_text = self._format_turn_for_prefill(role, content, character, character_name, prefill_name)
                    if formatted_text:
                        text_blocks.append(formatted_text)
            
            # Add any remaining pre-image text
            if text_blocks:
                user_messages.append({"role": "user", "content": "\n\n".join(text_blocks)})
            
            # Build assistant prefix from post-image history
            assistant_parts = []
            for turn in raw_history[last_image_index + 1:]:
                formatted_text = self._format_turn_for_prefill(
                    turn.get("role"), 
                    turn.get("content"), 
                    turn.get("character"),
                    character_name,
                    prefill_name
                )
                if formatted_text:
                    assistant_parts.append(formatted_text)
            
            # Create assistant message with prefill
            assistant_content = "\n\n".join(assistant_parts) + f"\n\n{prefill_name}:" if assistant_parts else f"{prefill_name}:"
            
            return [{"role": "system", "content": system_prompt}] + user_messages + [{"role": "assistant", "content": assistant_content}]
        else:
            # No images - simpler format
            conversation_parts = []
            
            for turn in raw_history:
                formatted_text = self._format_turn_for_prefill(
                    turn.get("role"),
                    turn.get("content"),
                    turn.get("character"),
                    character_name,
                    prefill_name
                )
                if formatted_text:
                    conversation_parts.append(formatted_text)
            
            # Create prefill format
            assistant_content = "\n\n".join(conversation_parts) + f"\n\n{prefill_name}:" if conversation_parts else f"{prefill_name}:"
            
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.config.conversation.prefill_user_message},
                {"role": "assistant", "content": assistant_content}
            ]
    
    def _format_turn_for_prefill(self, role: str, content: Any, character: Optional[str], 
                                 current_character: str, prefill_name: str) -> Optional[str]:
        """Format a single turn for prefill conversation."""
        # Extract text from content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = " ".join(text_parts)
        
        if not content or not isinstance(content, str):
            return None
        
        if role == "user":
            # Human messages
            return f"H: {content}"
        elif role == "assistant":
            # Character messages
            if character == current_character:
                # This character's own messages
                return f"{prefill_name}: {content}"
            elif character:
                # Other character's messages - get their prefill name
                other_config = self.character_manager.get_character_config(character)
                if other_config and other_config.prefill_name:
                    return f"{other_config.prefill_name}: {content}"
                else:
                    return f"{character}: {content}"
            else:
                # Generic assistant
                return f"Assistant: {content}"
        elif role == "system":
            # System messages (like interruptions)
            return f"System: {content}"
        
        return None
    
    def _convert_character_messages_to_prefill(self, messages: List[Dict[str, str]], character_prefill_name: str) -> List[Dict[str, str]]:
        """Convert character messages to prefill format with character-specific name, preserving images.
        
        This handles messages that have already been formatted by format_messages_for_character,
        which means other characters' messages are already prefixed with their names.
        """
        # Extract system prompt
        system_prompt = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append(msg)
        
        # Use the same image-aware conversion as regular prefill
        # First, find the last image index
        last_image_index = -1
        for i, msg in enumerate(chat_messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # Check if this message contains an image
                for content_item in msg["content"]:
                    if content_item.get("type") == "image":
                        last_image_index = i
                        break
        
        # If there are images, use the image-aware prefill format
        if last_image_index >= 0:
            # Build user messages and assistant prefix using image-aware logic
            user_messages = []
            current_text_block = []
            
            # Process messages up to and including the last image
            for i, msg in enumerate(chat_messages):
                # Check if this is an image message
                is_image_message = False
                if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                    for content_item in msg["content"]:
                        if content_item.get("type") == "image":
                            is_image_message = True
                            break
                
                if is_image_message and i <= last_image_index:
                    # First, add any accumulated text block
                    if current_text_block:
                        text_content = "\n\n".join(current_text_block)
                        user_messages.append({"role": "user", "content": text_content})
                        current_text_block = []
                    
                    # Then add the image message as-is
                    user_messages.append(msg)
                elif i <= last_image_index:
                    # Regular text message before the last image - format and accumulate
                    content = msg.get('content', '')
                    if msg["role"] == "user":
                        # User messages might already have character names from format_messages_for_character
                        # Only add "H:" if there's no existing prefix
                        if isinstance(content, str) and not re.match(r'^[^:]+:', content):
                            content = f"H: {content}"
                        current_text_block.append(content)
                    else:  # assistant
                        # Assistant messages are already properly formatted
                        # Just use the content as-is
                        current_text_block.append(content)
            
            # Add any remaining text block before the last image
            if current_text_block and last_image_index >= 0:
                text_content = "\n\n".join(current_text_block)
                user_messages.append({"role": "user", "content": text_content})
                current_text_block = []
            
            # Build assistant prefix from everything after the last image
            assistant_parts = []
            for i, msg in enumerate(chat_messages):
                if i > last_image_index:
                    content = msg.get('content', '')
                    if isinstance(content, list):
                        # Extract text from list content
                        text_parts = []
                        for item in content:
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                        content = " ".join(text_parts)
                    
                    if msg["role"] == "user":
                        # User messages might already have character names from format_messages_for_character
                        # Only add "H:" if there's no existing prefix
                        if isinstance(content, str) and not re.match(r'^[^:]+:', content):
                            content = f"H: {content}"
                        assistant_parts.append(content)
                    else:  # assistant
                        # Assistant messages are already properly formatted
                        # Just use the content as-is
                        assistant_parts.append(content)
            
            # Create assistant prefix
            if assistant_parts:
                assistant_prefix = "\n\n".join(assistant_parts) + f"\n\n{character_prefill_name}:"
            else:
                assistant_prefix = f"{character_prefill_name}:"
            
            # If no user messages were created, use the default
            if not user_messages:
                user_messages = [{"role": "user", "content": self.config.conversation.prefill_user_message}]
            
            # Build final messages
            prefill_messages = [{"role": "system", "content": system_prompt}] + user_messages + [{"role": "assistant", "content": assistant_prefix}]
        else:
            # No images - use the original text-only approach
            conversation_parts = []
            human_name = "H"
            
            for msg in chat_messages:
                content = msg["content"]
                
                if msg["role"] == "user":
                    if isinstance(content, str) and not re.match(r'^[^:]+:', content):
                        content = f"{human_name}: {content}"
                    conversation_parts.append(content)
                else:  # assistant
                    # Assistant messages are already properly formatted
                    # Just use the content as-is
                    conversation_parts.append(content)
            
            # Create prefill format
            if conversation_parts:
                assistant_prefix = "\n\n".join(conversation_parts) + f"\n\n{character_prefill_name}:"
            else:
                assistant_prefix = f"{character_prefill_name}:"
            
            prefill_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.config.conversation.prefill_user_message},
                {"role": "assistant", "content": assistant_prefix}
            ]
        
        # Add metadata to indicate this is prefill format
        prefill_messages[-1]["_is_prefill"] = True
        prefill_messages[-1]["_prefill_name"] = character_prefill_name
        
        return prefill_messages
    
    def _get_conversation_history_for_character(self):
        """Get conversation history formatted for character LLM.
        Preserves full content including images.
        """
        history = []
        for turn in self.state.conversation_history:
            # Include both completed and interrupted turns
            if turn.status in ["completed", "interrupted"]:
                entry = {
                    "role": turn.role,
                    "content": turn.content  # Keep as-is (string or list)
                }
                if turn.character:
                    entry["character"] = turn.character
                history.append(entry)
        return history
    
    
    async def _conversation_manager(self):
        """Simplified conversation manager - processing is now handled by _process_pending_utterances."""
        print("🔄 Conversation manager started (simplified)")
        
        while self.state.is_active:
            try:
                await asyncio.sleep(0.1)  # Less frequent checking since processing is event-driven
                
                # Just show debug info occasionally
                if self.config.development.enable_debug_mode:
                    pending_count = sum(1 for turn in self.state.conversation_history 
                                      if turn.role == "user" and turn.status == "pending")
                    if pending_count > 0:
                        print(f"🐛 Debug: {pending_count} pending utterances, "
                              f"Processing: {self.state.is_processing_llm}, "
                              f"Speaking: {self.state.is_speaking}")
                
            except Exception as e:
                print(f"Conversation manager error: {e}")
                self.logger.error(f"Conversation manager error: {e}")
                import traceback
                traceback.print_exc()
    
    def _seems_complete(self, text: str) -> bool:
        """Heuristic to determine if a statement seems complete."""
        if not text:
            return False
        
        # Check for ending punctuation
        if text.rstrip().endswith(('.', '!', '?')):
            return True
        
        # Check for common complete phrases
        complete_patterns = [
            r'\bthat(?:[\'s]|s) (it|all|everything)\b',
            r'\b(okay|alright|thanks?|thank you)\b\s*$',
            r'\b(done|finished|complete)\b\s*$'
        ]
        
        for pattern in complete_patterns:
            if re.search(pattern, text.lower()):
                return True
        
        return False
    
    async def _process_with_llm(self, user_input: str, reference_turn: ConversationTurn, generation: int = None):
        """Process user input with LLM and speak response."""
        if self.state.is_processing_llm:
            print(f"⚠️ Already processing LLM, skipping: {user_input}")
            return
        
        # Check generation if provided
        if generation is not None and generation != self._processing_generation:
            print(f"🚫 LLM processing cancelled - newer request exists")
            return
            
        # Mark all processing user turns as completed NOW since we're committing to process them
        for turn in self.state.conversation_history:
            if turn.role == "user" and turn.status == "processing":
                turn.status = "completed"
                print(f"   ✅ Marked user turn as completed: '{turn.content[:50]}...'")
        
        # Always use character processing
        await self._process_with_character_llm(user_input, reference_turn, generation)
    async def _speak_text(self, text: str):
        """Speak a simple text message."""
        try:
            self.state.is_speaking = True
            result = await self.tts.speak_text(text)
            
            # Log Whisper tracking for testing
            spoken_heuristic = self.tts.get_spoken_text_heuristic().strip()
            if spoken_heuristic:
                print(f"🎙️ Simple speech heuristic result: '{spoken_heuristic[:100]}...'")
            
            if not result:
                print("🛑 Speech was interrupted")
        finally:
            self.state.is_speaking = False
    
    async def cleanup(self):
        """Clean up all resources."""
        # First stop the conversation to prevent new operations
        try:
            await self.stop_conversation()
        except Exception as e:
            print(f"Error stopping conversation: {e}")
        
        # Stop thinking sound early to prevent conflicts
        try:
            await self.thinking_sound.stop()  # Force stop any playing sound
            await asyncio.sleep(0.1)  # Give it time to stop
        except Exception as e:
            print(f"Error stopping thinking sound: {e}")
        
        # Clean up STT
        try:
            await self.stt.cleanup()
        except Exception as e:
            print(f"Error cleaning up STT: {e}")
            
        # Clean up camera
        try:
            if self.camera:
                self.camera.stop()
                print("📷 Camera stopped")
        except Exception as e:
            print(f"Error cleaning up camera: {e}")
        
        # Clean up TTS
        try:
            await self.tts.cleanup()
        except Exception as e:
            print(f"Error cleaning up TTS: {e}")
            
        # Clean up UI server
        try:
            await self.ui_server.stop()
        except Exception as e:
            print(f"Error cleaning up UI server: {e}")
            
        # Finally clean up thinking sound resources
        try:
            self.thinking_sound.cleanup()
        except Exception as e:
            print(f"Error cleaning up thinking sound: {e}")

async def main():
    """Main function for the unified voice conversation system."""
    print("🎙️ Unified Voice Conversation System (YAML Config)")
    print("==================================================")
    
    conversation = None
    
    try:
        # Load configuration from YAML
        print("📁 Loading configuration...")
        config = load_config()
        print("✅ Configuration loaded successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 To create a config file:")
        print("   1. Copy config.yaml.example to config.yaml")
        print("   2. Fill in your API keys")
        print("   3. Adjust settings as needed")
        print("\n   OR run: python config_loader.py create-example")
        return
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return
    except Exception as e:
        print(f"❌ Unexpected error loading config: {e}")
        return
    
    try:
        # Create conversation system
        conversation = UnifiedVoiceConversation(config)
        
        # Setup signal handling for graceful shutdown
        import signal
        
        def signal_handler():
            print("\n🛑 Received shutdown signal")
            if conversation:
                conversation.state.is_active = False
        
        # Register signal handlers
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                signal.signal(sig, lambda s, f: signal_handler())
            except ValueError:
                # Signal may not be available on all platforms
                pass
        
        success = await conversation.start_conversation()
        if not success:
            print("❌ Failed to start conversation")
            return
            
        # Keep running until interrupted
        print("🎯 Conversation is active. Press Ctrl+C to exit.")
        try:
            while conversation.state.is_active:
                await asyncio.sleep(0.5)  # Shorter sleep for more responsive shutdown
        except asyncio.CancelledError:
            print("\n⏹️ Task cancelled")
            
    except KeyboardInterrupt:
        print("\n👋 Conversation ended by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conversation:
            print("🧹 Cleaning up...")
            try:
                await conversation.cleanup()
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")
            print("✅ Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main()) 
