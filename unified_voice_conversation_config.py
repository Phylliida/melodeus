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
from typing import Optional, List, Dict, Any, AsyncGenerator
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

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    role: str  # "user", "assistant", "system", or "tool"
    content: str
    timestamp: datetime
    status: str = "completed"  # "pending", "processing", "completed", "interrupted"
    speaker_id: Optional[int] = None
    character: Optional[str] = None  # Character name for multi-character conversations

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

class UnifiedVoiceConversation:
    """Unified voice conversation system with YAML configuration."""
    
    def __init__(self, config: VoiceAIConfig):
        self.config = config
        self.state = ConversationState()
        
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        self.async_anthropic_client = None
        
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
        
        # Initialize character manager if multi-character mode is enabled
        self.character_manager: Optional[CharacterManager] = None
        if config.conversation.enable_multi_character:
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
            self.character_manager = create_character_manager(characters_config)
            print("🎭 Multi-character mode enabled")
        
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
    
    def _log_conversation_turn(self, role: str, content: str):
        """Log a conversation turn in history file format.
        
        Args:
            role: 'user' or 'assistant'
            content: The message content (may already include speaker prefix)
        """
        try:
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
        
        # Add character names from multi-character mode
        if config.conversation.enable_multi_character and config.conversation.characters_config:
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
    
    def _log_llm_request(self, messages: List[Dict[str, str]], model: str, timestamp: float, provider: str = None) -> str:
        """Log LLM request to a file and return the filename."""
        filename = self._generate_log_filename("request", timestamp)
        filepath = self.llm_logs_dir / filename
        
        request_data = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "provider": provider or self.config.conversation.llm_provider,
            "model": model,
            "messages": messages,
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

    def _convert_to_prefill_format(self, messages: List[Dict[str, str]]) -> tuple[str, str]:
        """Convert chat messages to prefill format.
        Returns (user_message, assistant_message_prefix).
        
        For multi-speaker conversations, preserves speaker names in content
        but ensures the conversation ends with the configured AI participant.
        """
        user_message = self.config.conversation.prefill_user_message
        
        # Build conversation turns from history
        conversation_turns = []
        human_name = self.config.conversation.prefill_participants[0]  # Default: 'H'
        ai_name = self.config.conversation.prefill_participants[1]     # Default: 'Claude'
        
        for msg in messages:
            if msg["role"] == "system":
                continue  # Skip system messages in prefill mode
            elif msg["role"] == "user":
                # For multi-speaker support, content already includes speaker name
                # Check if content already has speaker format, if not add human name
                content = msg['content']
                # More flexible check - allow with or without space after colon
                if not re.match(r'^[^:]+:', content):
                    content = f"{human_name}: {content}"
                conversation_turns.append(content)
            elif msg["role"] == "assistant":
                # For multi-speaker support, content already includes speaker name
                # Just add it as-is since it preserves the original speaker context
                conversation_turns.append(msg['content'])
        
        # Join turns with double newlines
        assistant_content = "\n\n".join(conversation_turns)
        if assistant_content:
            assistant_content += f"\n\n{ai_name}:"
        else:
            assistant_content = f"{ai_name}:"
            
        return user_message, assistant_content

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
                        
                        # Save previous message if exists
                        if current_speaker and current_content:
                            # Determine role based on configured participants
                            # For multi-speaker: classify as 'user' if it's the human participant,
                            # otherwise as 'assistant' to preserve the conversation context
                            role = "user" if current_speaker == human_name else "assistant"
                            
                            # Include speaker name in content for multi-speaker context
                            content_with_speaker = f"{current_speaker}: " + '\n'.join(current_content).strip()
                            
                            messages.append({
                                "role": role,
                                "content": content_with_speaker
                            })
                        
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
                content_with_speaker = f"{current_speaker}: " + '\n'.join(current_content).strip()
                
                messages.append({
                    "role": role,
                    "content": content_with_speaker
                })
            
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
                turn = ConversationTurn(
                    role=msg["role"],
                    content=msg["content"], 
                    timestamp=datetime.now(),  # We don't have original timestamps
                    status="completed"  # Historical messages are completed
                )
                self.state.conversation_history.append(turn)
                
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
        print(f"🎯 Final{speaker_info}: {result.text}")
        
        # Check for interruption at ANY stage of AI response
        interrupted = False
        
        if self.tts.is_currently_playing():
            print(f"🛑 Interrupting TTS playback with: {result.text}")
            
            # Stop TTS and wait for Whisper to finish processing
            await self.tts.stop()
            
            # Get spoken content synchronously (available after stop() completes)
            spoken_content = self.tts.get_spoken_text_heuristic().strip()
            
            # Don't add assistant response here - let _stream_llm_to_tts handle it
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
        
        # Add the new user utterance to conversation history AFTER handling interruption
        user_turn = ConversationTurn(
            role="user",
            content=result.text,
            timestamp=result.timestamp,
            status="pending",
            speaker_id=result.speaker_id
        )
        self.state.conversation_history.append(user_turn)
        self._log_conversation_turn("user", result.text)
        print(f"💬 Added user utterance: '{result.text}'")
        
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
        # Use a lock to prevent concurrent processing
        if self.state.is_processing_llm or self.state.is_speaking:
            print("⏸️ Already processing, skipping pending utterances")
            return
            
        # Find pending user utterances
        pending_turns = [turn for turn in self.state.conversation_history 
                        if turn.role == "user" and turn.status == "pending"]
        
        if not pending_turns:
            return
            
        # Process all pending turns as a single input
        combined_content = " ".join(turn.content for turn in pending_turns)
        print(f"🧠 Processing {len(pending_turns)} pending utterances: '{combined_content}'")
        
        # Mark all pending turns as processing
        for turn in pending_turns:
            turn.status = "processing"
            
        await self._process_with_llm(combined_content, pending_turns[0])  # Pass first turn for reference
    
    async def _process_with_character_llm(self, user_input: str, reference_turn: ConversationTurn):
        """Process user input with multi-character system."""
        try:
            self.state.is_processing_llm = True
            
            # Let director decide who speaks next
            next_speaker = await self.character_manager.select_next_speaker(
                self._get_conversation_history_for_director()
            )
            
            if next_speaker == "USER" or next_speaker is None:
                print("🎭 Director: User should speak next")
                self.state.is_processing_llm = False
                return
            
            # Set active character
            self.character_manager.set_active_character(next_speaker)
            character_config = self.character_manager.get_character_config(next_speaker)
            
            if not character_config:
                print(f"❌ Unknown character: {next_speaker}")
                self.state.is_processing_llm = False
                return
            
            print(f"🎭 {next_speaker} is responding...")
            
            # Get formatted messages for this character
            messages = self.character_manager.format_messages_for_character(
                next_speaker,
                self._get_conversation_history_for_character()
            )
            
            # Request timestamp for logging
            request_timestamp = time.time()
            request_filename = ""
            
            # If using prefill mode, convert messages
            if self.config.conversation.conversation_mode == "prefill" and character_config.llm_provider == "anthropic":
                # Get prefill name for this character
                prefill_name = character_config.prefill_name or character_config.name
                
                # Convert messages to prefill format with character-specific name
                messages = self._convert_character_messages_to_prefill(messages, prefill_name)
                
                # Log the converted prefill request
                request_filename = self._log_llm_request(
                    messages, 
                    character_config.llm_model, 
                    request_timestamp,
                    character_config.llm_provider
                )
            else:
                # Log the regular request
                request_filename = self._log_llm_request(
                    messages, 
                    character_config.llm_model, 
                    request_timestamp,
                    character_config.llm_provider
                )
            
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
            self.state.is_processing_llm = False
    
    async def _stream_character_openai_response(self, messages, character_config, request_timestamp):
        """Stream response from OpenAI for a specific character."""
        client = self.character_manager.get_character_client(character_config.name)
        
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
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"❌ OpenAI streaming error for {character_config.name}: {e}")
            raise
    
    async def _stream_character_anthropic_response(self, messages, character_config, request_timestamp):
        """Stream response from Anthropic for a specific character."""
        client = self.character_manager.get_character_client(character_config.name)
        
        try:
            # Check for prefill format using metadata or structure
            is_prefill_format = False
            prefill_name = None
            
            if len(messages) > 0:
                last_msg = messages[-1]
                # Check for our metadata marker
                if last_msg.get("_is_prefill"):
                    is_prefill_format = True
                    prefill_name = last_msg.get("_prefill_name", character_config.name)
                # Fallback: check if last message is assistant (typical prefill pattern)
                elif (last_msg.get("role") == "assistant" and 
                      self.config.conversation.conversation_mode == "prefill"):
                    is_prefill_format = True
                    prefill_name = character_config.prefill_name or character_config.name
            
            # Extract system content and prepare messages
            system_content = ""
            anthropic_messages = []
            
            for msg in messages:
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
                                yield prefix_buffer[len(expected_prefix):]
                            else:
                                # Doesn't match expected prefix, yield everything
                                yield prefix_buffer
                            prefix_buffer = None  # Stop checking
                        # else continue buffering
                    else:
                        yield text
                    
        except Exception as e:
            print(f"❌ Anthropic streaming error for {character_config.name}: {e}")
            raise
    
    def _get_conversation_history_for_director(self):
        """Get conversation history formatted for director."""
        history = []
        for turn in self.state.conversation_history[-20:]:  # Last 20 turns
            # Include both completed and interrupted turns
            if turn.status in ["completed", "interrupted"]:
                entry = {
                    "role": turn.role,
                    "content": turn.content
                }
                if turn.character:
                    entry["character"] = turn.character
                history.append(entry)
        return history
    
    def _set_character_voice(self, character_config):
        """Set TTS voice for a specific character."""
        if not character_config or not self.character_manager:
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
    
    def _convert_character_messages_to_prefill(self, messages: List[Dict[str, str]], character_prefill_name: str) -> List[Dict[str, str]]:
        """Convert character messages to prefill format with character-specific name."""
        # Extract system prompt
        system_prompt = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append(msg)
        
        # Convert to prefill format with character name
        conversation_parts = []
        human_name = "H"  # Default human name
        
        for msg in chat_messages:
            if msg["role"] == "user":
                # Check if it's a character message or human message
                if msg["content"].startswith("[") and "]: " in msg["content"]:
                    # It's another character's message
                    conversation_parts.append(msg["content"])
                else:
                    # It's a human message - check if it already has a speaker prefix
                    content = msg['content']
                    if not re.match(r'^[^:]+:', content):
                        content = f"{human_name}: {content}"
                    conversation_parts.append(content)
            else:  # assistant
                # This character's previous messages
                conversation_parts.append(f"{character_prefill_name}: {msg['content']}")
        
        # Create prefill format
        # For character prefill, we need to put the conversation in the assistant prefix
        # not in the user message, similar to regular prefill format
        if conversation_parts:
            assistant_prefix = "\n\n".join(conversation_parts) + f"\n\n{character_prefill_name}:"
        else:
            assistant_prefix = f"{character_prefill_name}:"
        
        # Use the standard prefill user message
        user_message = self.config.conversation.prefill_user_message
        
        # Return in Anthropic prefill format with metadata
        prefill_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_prefix}
        ]
        
        # Add metadata to indicate this is prefill format
        prefill_messages[-1]["_is_prefill"] = True
        prefill_messages[-1]["_prefill_name"] = character_prefill_name
        
        return prefill_messages
    
    def _get_conversation_history_for_character(self):
        """Get conversation history formatted for character LLM."""
        history = []
        for turn in self.state.conversation_history:
            # Include both completed and interrupted turns
            if turn.status in ["completed", "interrupted"]:
                entry = {
                    "role": turn.role,
                    "content": turn.content
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
    
    async def _process_with_llm(self, user_input: str, reference_turn: ConversationTurn):
        """Process user input with LLM and speak response."""
        if self.state.is_processing_llm:
            print(f"⚠️ Already processing LLM, skipping: {user_input}")
            return
        
        # Mark all processing user turns as completed NOW since we're committing to process them
        for turn in self.state.conversation_history:
            if turn.role == "user" and turn.status == "processing":
                turn.status = "completed"
                print(f"   ✅ Marked user turn as completed: '{turn.content[:50]}...'")
        
        # If multi-character mode, delegate to character processing
        if self.character_manager:
            await self._process_with_character_llm(user_input, reference_turn)
            return
        
        try:
            self.state.is_processing_llm = True
            self.state.current_processing_turn = reference_turn
            print(f"🔄 Starting LLM processing for: '{user_input}'")
            
            # Mark all processing user turns as completed NOW since we're committing to process them
            for turn in self.state.conversation_history:
                if turn.role == "user" and turn.status == "processing":
                    turn.status = "completed"
                    print(f"   ✅ Marked user turn as completed: '{turn.content[:50]}...'")
            
            # Build messages from conversation history (completed turns only)
            messages = [{"role": "system", "content": self.config.conversation.system_prompt}]
            
            # Debug: Show conversation history state
            print(f"📊 Building LLM context from {len(self.state.conversation_history)} total turns")
            status_counts = {}
            for turn in self.state.conversation_history:
                status_counts[turn.status] = status_counts.get(turn.status, 0) + 1
            print(f"   Status breakdown: {status_counts}")
            
            # Add ALL turns from conversation history (simpler is better)
            # The only turns we exclude are assistant turns that never started
            included_count = 0
            all_turns = self.state.conversation_history[-200:]  # Keep last 200 exchanges
            
            # Only print last 20 for brevity
            print_threshold = max(0, len(all_turns) - 20)
            
            for i, turn in enumerate(all_turns):
                # Include everything except assistant turns that were never spoken
                if turn.role == "user" or turn.status in ["completed", "interrupted"]:
                    # For prefill mode, add speaker prefix
                    content = turn.content
                    if self.config.conversation.conversation_mode == "prefill":
                        if turn.role == "user":
                            # User messages should have the user participant name
                            user_name = self.config.conversation.prefill_participants[0] if self.config.conversation.prefill_participants else "H"
                            if not content.startswith(f"{user_name}:"):
                                content = f"{user_name}: {content}"
                        else:  # assistant
                            # Assistant messages should have the assistant participant name
                            assistant_name = self.config.conversation.prefill_participants[1] if len(self.config.conversation.prefill_participants) > 1 else "Claude"
                            if not content.startswith(f"{assistant_name}:"):
                                content = f"{assistant_name}: {content}"
                    
                    messages.append({
                        "role": turn.role,
                        "content": content
                    })
                    included_count += 1
                    if i >= print_threshold:  # Only print last 20
                        print(f"   ✅ Including {turn.role} ({turn.status}): '{content[:50]}...'")
                else:
                    if i >= print_threshold:  # Only print last 20
                        print(f"   ⚠️ Excluding {turn.role} turn with status '{turn.status}': '{turn.content[:50]}...'")
            
            print(f"   ✅ Total included: {included_count} turns in LLM context")
            
            print("🤖 Getting LLM response...")
            
            # Create task for streaming LLM response to TTS
            self.state.current_llm_task = asyncio.create_task(
                self._stream_llm_to_tts(messages, user_input)
            )
            
            # Wait for the task to complete
            try:
                await self.state.current_llm_task
            except asyncio.CancelledError:
                print("🛑 LLM streaming task was cancelled")
            finally:
                self.state.current_llm_task = None
            
        except Exception as e:
            print(f"LLM processing error: {e}")
            self.logger.error(f"LLM processing error: {e}")
            await self._speak_text("Sorry, I encountered an error processing your request.")
        finally:
            self.state.is_processing_llm = False
            self.state.current_processing_turn = None
            print(f"🔄 LLM processing finished, state cleared")
    
    async def _stream_llm_to_tts(self, messages: List[Dict[str, str]], user_input: str):
        """Stream LLM completion to TTS."""
        call_id = f"{time.time():.3f}"  # Unique ID for this call
        print(f"🔍 _stream_llm_to_tts called with ID: {call_id} for input: '{user_input[:50]}...'")
        
        assistant_response = ""  # Initialize at method level to fix scoping
        request_timestamp = time.time()
        request_filename = ""
        was_interrupted = False
        error_msg = None
        tts_processing_complete = False  # Flag to prevent duplicate processing
        response_added_to_history = False  # Prevent duplicate history additions
        
        # Request will be logged when we make the actual API call
        
        try:
            # Check if task was cancelled before we even start
            if asyncio.current_task() and asyncio.current_task().cancelled():
                print(f"🚫 Task {call_id} cancelled before starting")
                return
                
            self.state.is_speaking = True
            
            # Create async generator for LLM response
            async def llm_generator():
                nonlocal assistant_response, request_filename, was_interrupted, error_msg  # Access the method-level variables
                
                if self.config.conversation.llm_provider == "openai":
                    async for chunk in self._stream_openai_response(messages, request_timestamp):
                        if isinstance(chunk, dict) and 'request_filename' in chunk:
                            request_filename = chunk['request_filename']
                            continue
                        # Check for interruption during LLM generation
                        if not self.state.is_speaking or not self.state.is_processing_llm:
                            print("🛑 LLM streaming interrupted by user")
                            was_interrupted = True
                            break
                        
                        if chunk is None:  # Error occurred
                            error_msg = "OpenAI API error"
                            yield "Sorry, I encountered an error with the language model."
                            return
                            
                        assistant_response += chunk
                        if self.show_tts_chunks:
                            print(f"🔊 TTS chunk: {chunk}")
                        yield chunk
                        
                elif self.config.conversation.llm_provider == "anthropic":
                    async for chunk in self._stream_anthropic_response(messages, request_timestamp):
                        if isinstance(chunk, dict) and 'request_filename' in chunk:
                            request_filename = chunk['request_filename']
                            continue
                        # Check for interruption during LLM generation
                        if not self.state.is_speaking or not self.state.is_processing_llm:
                            print("🛑 LLM streaming interrupted by user")
                            was_interrupted = True
                            break
                        
                        if chunk is None:  # Error occurred
                            error_msg = "Anthropic API error"
                            yield "Sorry, I encountered an error with the language model."
                            return
                            
                        assistant_response += chunk
                        if self.show_tts_chunks:
                            print(f"🔊 TTS chunk: {chunk}")
                        yield chunk
                        
                else:
                    error_msg = f"Unsupported LLM provider: {self.config.conversation.llm_provider}"
                    yield f"Sorry, unsupported language model provider: {self.config.conversation.llm_provider}"
                

            
            # Use multi-voice TTS streaming
            print("🎭 Using multi-voice streaming")
            result = await self.tts.speak_stream_multi_voice(llm_generator())
            
            # Check if we were cancelled during TTS
            if asyncio.current_task() and asyncio.current_task().cancelled():
                print(f"🚫 Task {call_id} cancelled during/after TTS")
                raise asyncio.CancelledError()
                
            print(f"🎭 Multi-voice result: {result}")
            tts_processing_complete = True
            
            if not result:
                print("⚠️ Multi-voice TTS failed or was interrupted")
                # For interrupted responses, we should capture ALL the text that was generated
                # not just what was spoken (for proper context in future turns)
                if hasattr(self.tts, 'current_session') and self.tts.current_session:
                    full_generated = self.tts.current_session.generated_text.strip()
                    if full_generated and not response_added_to_history:
                        # Add the full generated text with interrupted status
                        assistant_turn = ConversationTurn(
                            role="assistant",
                            content=full_generated,
                            timestamp=datetime.now(),
                            status="interrupted"
                        )
                        self.state.conversation_history.append(assistant_turn)
                        self._log_conversation_turn("assistant", full_generated)
                        print(f"💬 Added interrupted assistant response (full):")
                        print(f"   Generated: {len(full_generated)} chars - '{full_generated[:80]}...'")
                        response_added_to_history = True
                else:
                    # Fallback to heuristic if we can't get full text
                    spoken_heuristic = self.tts.get_spoken_text_heuristic().strip()
                    if spoken_heuristic and not response_added_to_history:
                        assistant_turn = ConversationTurn(
                            role="assistant",
                            content=spoken_heuristic,
                            timestamp=datetime.now(),
                            status="interrupted"
                        )
                        self.state.conversation_history.append(assistant_turn)
                        self._log_conversation_turn("assistant", spoken_heuristic)
                        print(f"💬 Added interrupted assistant response (heuristic): '{spoken_heuristic[:100]}...'")
                        response_added_to_history = True
                return
            
            # Use heuristic approach for conversation history (preserves exact LLM text)
            spoken_heuristic = self.tts.get_spoken_text_heuristic().strip()
            generated_vs_spoken = self.tts.get_generated_vs_spoken()
            was_fully_spoken = self.tts.was_fully_spoken()
            
            if spoken_heuristic or generated_vs_spoken.get('spoken_whisper'):
                print(f"🎙️ WHISPER TRACKING RESULTS:")
                print(f"   Generated: {len(generated_vs_spoken['generated'])} chars")
                print(f"   Whisper raw: {len(generated_vs_spoken['spoken_whisper'])} chars")
                print(f"   Heuristic: {len(spoken_heuristic)} chars")
                print(f"   Fully spoken: {was_fully_spoken}")
                print(f"   Heuristic text: '{spoken_heuristic[:200]}...'")
            
            # Determine content for history
            # If fully spoken, use the complete generated text (no need for Whisper)
            # If interrupted, use Whisper heuristic to get only what was spoken
            if was_fully_spoken:
                content_for_history = assistant_response  # Use full generated text
            else:
                content_for_history = spoken_heuristic if spoken_heuristic else assistant_response
            
            # Update conversation history based on completion status
            if content_for_history.strip() and not response_added_to_history:
                # User turns are already marked as completed at the start of LLM processing
                        
                # Add the assistant response
                assistant_turn = ConversationTurn(
                    role="assistant",
                    content=content_for_history,  # What was actually spoken
                    timestamp=datetime.now(),
                    status="completed" if was_fully_spoken else "interrupted"
                )
                self.state.conversation_history.append(assistant_turn)
                self._log_conversation_turn("assistant", content_for_history)
                response_added_to_history = True
                
                if was_fully_spoken:
                    print(f"💬 Added complete assistant response: {len(content_for_history)} chars")
                    print(f"   Content: '{content_for_history[:100]}...'")
                else:
                    print(f"💬 Added partial assistant response (Whisper heuristic):")
                    print(f"   Generated: {len(assistant_response)} chars")
                    print(f"   Spoken: {len(content_for_history)} chars - '{content_for_history[:80]}...'")
                
            else:
                # User turns remain completed even if assistant response was interrupted
                print(f"🛑 Response was interrupted - no assistant content added (call_id: {call_id})")
            
        except asyncio.CancelledError:
            print(f"🚫 LLM streaming task {call_id} was cancelled")
            # Make sure to stop TTS if it's still running
            if self.tts.is_streaming:
                await self.tts.stop()
            
            # Add the interrupted response to history before exiting
            if not response_added_to_history:
                # Try to get what was actually spoken using Whisper heuristic
                spoken_text = self.tts.get_spoken_text_heuristic().strip()
                generated_text = ""
                if hasattr(self.tts, 'current_session') and self.tts.current_session:
                    generated_text = self.tts.current_session.generated_text.strip()
                
                if spoken_text:
                    assistant_turn = ConversationTurn(
                        role="assistant",
                        content=spoken_text,
                        timestamp=datetime.now(),
                        status="interrupted"
                    )
                    self.state.conversation_history.append(assistant_turn)
                    self._log_conversation_turn("assistant", spoken_text)
                    print(f"💬 Added cancelled assistant response (Whisper):")
                    print(f"   Generated: {len(generated_text)} chars - '{generated_text[:80]}...'")
                    print(f"   Spoken:    {len(spoken_text)} chars - '{spoken_text[:80]}...'")
                    response_added_to_history = True
                # Fallback to full generated text if Whisper not available
                elif hasattr(self.tts, 'current_session') and self.tts.current_session:
                    full_generated = self.tts.current_session.generated_text.strip()
                    if full_generated:
                        assistant_turn = ConversationTurn(
                            role="assistant",
                            content=full_generated,
                            timestamp=datetime.now(),
                            status="interrupted"
                        )
                        self.state.conversation_history.append(assistant_turn)
                        self._log_conversation_turn("assistant", full_generated)
                        print(f"💬 Added cancelled assistant response (full): '{full_generated[:100]}...'")
                        response_added_to_history = True
                # Final fallback to assistant_response if no TTS session
                elif assistant_response.strip():
                    assistant_turn = ConversationTurn(
                        role="assistant",
                        content=assistant_response,
                        timestamp=datetime.now(),
                        status="interrupted"
                    )
                    self.state.conversation_history.append(assistant_turn)
                    self._log_conversation_turn("assistant", assistant_response)
                    print(f"💬 Added cancelled assistant response (fallback): '{assistant_response[:100]}...'")
                    response_added_to_history = True
            
            raise  # Re-raise to let the caller know we were cancelled
        except Exception as e:
            error_msg = str(e)
            print(f"Error in _stream_llm_to_tts: {e}")
            self.logger.error(f"Error in _stream_llm_to_tts: {e}")
            await self._speak_text("Sorry, I encountered an error processing your request.")
        finally:
            # Only do cleanup if TTS processing was attempted
            if tts_processing_complete:
                # Log the response (or error) after completion
                response_timestamp = time.time()
                if request_filename:  # Only log if we made a request
                    self._log_llm_response(
                        assistant_response, 
                        request_filename, 
                        response_timestamp, 
                        was_interrupted, 
                        error_msg,
                        self.config.conversation.llm_provider
                    )
                
                self.state.is_speaking = False
                print(f"✅ Response completed successfully (call_id: {call_id})")
            else:
                print("⚠️ TTS processing was not completed, skipping cleanup")

    async def _stream_openai_response(self, messages: List[Dict[str, str]], request_timestamp: float):
        """Stream response from OpenAI API."""
        models_to_try = [self.config.conversation.llm_model]
        
        for model in models_to_try:
            try:
                # Log the request right before making the API call
                request_filename = self._log_llm_request(messages, model, request_timestamp, "openai")
                yield {'request_filename': request_filename}
                
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=self.config.conversation.max_tokens
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # Success
                
            except Exception as e:
                print(f"Failed with OpenAI model {model}: {e}")
                if model == models_to_try[-1]:
                    yield None  # Signal error
                    return

    async def _stream_anthropic_response(self, messages: List[Dict[str, str]], request_timestamp: float):
        """Stream response from Anthropic API."""
        models_to_try = [self.config.conversation.llm_model]
        
        for model in models_to_try:
            try:
                # Convert to appropriate format based on conversation mode
                if self.config.conversation.conversation_mode == "prefill":
                    user_message, assistant_prefix = self._convert_to_prefill_format(messages)
                    
                    # Use Anthropic's completion API with prefill and CLI system prompt
                    # Generate stop sequences from configured participant names and detected speakers
                    human_name = self.config.conversation.prefill_participants[0]
                    ai_name = self.config.conversation.prefill_participants[1]
                    
                    # Start with configured participants
                    stop_sequences = [f"\n\n{human_name}:", f"\n\n{ai_name}:"]
                    
                    # Add all detected speakers from history to prevent bleeding into other speakers
                    if hasattr(self, 'detected_speakers') and self.detected_speakers:
                        for speaker in self.detected_speakers:
                            speaker_stop = f"\n\n{speaker}:"
                            if speaker_stop not in stop_sequences:
                                stop_sequences.append(speaker_stop)
                    
                    print(f"🛑 Using stop sequences: {stop_sequences}")
                    
                    # Log the prefilled messages right before the API call
                    prefill_messages = [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_prefix}
                    ]
                    request_filename = self._log_llm_request(prefill_messages, model, request_timestamp, "anthropic")
                    yield {'request_filename': request_filename}
                    
                    response = await self.async_anthropic_client.messages.create(
                        model=model,
                        max_tokens=self.config.conversation.max_tokens,
                        system=self.config.conversation.prefill_system_prompt,
                        messages=prefill_messages,
                        stop_sequences=stop_sequences,
                        stream=True
                    )
                else:
                    # Chat mode - convert system message if present
                    anthropic_messages = []
                    system_message = None
                    
                    for msg in messages:
                        if msg["role"] == "system":
                            system_message = msg["content"]
                        else:
                            anthropic_messages.append(msg)
                    
                    # Log the request right before making the API call
                    request_filename = self._log_llm_request(anthropic_messages, model, request_timestamp, "anthropic")
                    yield {'request_filename': request_filename}
                    
                    # Use Anthropic's chat API
                    kwargs = {
                        "model": model,
                        "max_tokens": self.config.conversation.max_tokens,
                        "messages": anthropic_messages,
                        "stream": True
                    }
                    
                    if system_message:
                        kwargs["system"] = system_message
                    
                    response = await self.async_anthropic_client.messages.create(**kwargs)
                
                # Stream the response
                accumulated_response = ""
                async for chunk in response:
                    if chunk.type == "content_block_delta":
                        if hasattr(chunk.delta, 'text'):
                            content = chunk.delta.text
                            accumulated_response += content
                            yield content
                
                # Convert from prefill format if needed
                if self.config.conversation.conversation_mode == "prefill":
                    # The response should already be clean since we prefilled with the participant name
                    pass  # No additional processing needed
                    
                return  # Success
                
            except Exception as e:
                print(f"Failed with Anthropic model {model}: {e}")
                if model == models_to_try[-1]:
                    yield None  # Signal error
                    return
    
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
        try:
            await self.stop_conversation()
        except Exception as e:
            print(f"Error stopping conversation: {e}")
        
        try:
            await self.stt.cleanup()
        except Exception as e:
            print(f"Error cleaning up STT: {e}")
        
        try:
            await self.tts.cleanup()
        except Exception as e:
            print(f"Error cleaning up TTS: {e}")

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
