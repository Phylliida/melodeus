#!/usr/bin/env python3
"""
Multi-Character Conversation System
Supports multiple AI models/characters in a single conversation with director orchestration
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from openai import OpenAI, AsyncOpenAI
from anthropic import AsyncAnthropic


@dataclass
class CharacterConfig:
    """Configuration for a single character/model in the conversation."""
    name: str  # Character name (e.g., "Claude", "GPT", "Assistant")
    llm_provider: str  # "openai" or "anthropic"
    llm_model: str  # Model identifier
    
    # Voice settings
    voice_id: str  # ElevenLabs voice ID
    voice_settings: Dict[str, float] = field(default_factory=lambda: {
        "speed": 1.0,
        "stability": 0.5,
        "similarity_boost": 0.8
    })
    
    # Character personality/prompt
    system_prompt: str = ""
    
    # Prefill name (for Anthropic prefill mode)
    prefill_name: Optional[str] = None
   
    # Conversation settings
    max_tokens: int = 300
    temperature: float = 0.7
    
    # API key (can be character-specific)
    api_key: Optional[str] = None


@dataclass
class DirectorConfig:
    """Configuration for the director LLM that orchestrates the conversation."""
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    
    # Director prompt template
    system_prompt: str = """You are a conversation director managing a multi-party discussion.
Your role is to decide who should speak next based on the conversation flow.

Available participants:
{participants}
- H (the human user)

Instructions:
- Analyze the conversation context in the prefill format (Name: message)
- Decide who should speak next
- Consider natural conversation flow and turn-taking
- Ensure balanced participation when appropriate
- The human is represented as "H" in the conversation

Respond with ONLY the name of the next speaker (including "H" for human turn)."""


@dataclass
class ConversationContext:
    """Shared context for all characters in the conversation."""
    topic: Optional[str] = None
    goals: List[str] = field(default_factory=list)
    shared_knowledge: Dict[str, Any] = field(default_factory=dict)
    turn_history: List[Tuple[str, str]] = field(default_factory=list)  # (speaker, message)


class CharacterManager:
    """Manages multiple characters in a conversation."""
    
    def __init__(self, characters: Dict[str, CharacterConfig], director_config: DirectorConfig):
        """
        Initialize character manager.
        
        Args:
            characters: Dictionary mapping character names to their configs
            director_config: Configuration for the director LLM
        """
        self.characters = characters
        self.director_config = director_config
        self.context = ConversationContext()
        
        # Initialize LLM clients for each unique provider/key combination
        self._llm_clients = {}
        self._init_llm_clients()
        
        # Initialize director client
        self._init_director_client()
        
        # Track active character
        self.active_character: Optional[str] = None
    
    def _init_llm_clients(self):
        """Initialize LLM clients for all characters."""
        # Group by provider and API key to avoid duplicate clients
        client_keys = {}
        
        for char_name, char_config in self.characters.items():
            key = (char_config.llm_provider, char_config.api_key or "default")
            
            if key not in client_keys:
                if char_config.llm_provider == "openai":
                    client = AsyncOpenAI(api_key=char_config.api_key)
                elif char_config.llm_provider == "anthropic":
                    client = AsyncAnthropic(api_key=char_config.api_key)
                else:
                    raise ValueError(f"Unknown LLM provider: {char_config.llm_provider}")
                
                client_keys[key] = client
            
            # Map character to client
            self._llm_clients[char_name] = client_keys[key]
    
    def _init_director_client(self):
        """Initialize the director LLM client."""
        # Get API key from director config or environment
        api_key = self.director_config.api_key
        
        if not api_key:
            # Try to get from environment based on provider
            if self.director_config.llm_provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
            elif self.director_config.llm_provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError(f"No API key found for director provider: {self.director_config.llm_provider}")
        
        if self.director_config.llm_provider == "openai":
            self._director_client = AsyncOpenAI(api_key=api_key)
        elif self.director_config.llm_provider == "anthropic":
            self._director_client = AsyncAnthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown director LLM provider: {self.director_config.llm_provider}")
    
    async def select_next_speaker(self, conversation_history: List[Dict[str, str]]) -> Optional[str]:
        """
        Use director LLM to select the next speaker.
        
        Args:
            conversation_history: Recent conversation history
            
        Returns:
            Character name or "USER" or None
        """
        import time
        from pathlib import Path
        import json
        # Build participants list
        participants = "\n".join([
            f"- {name}"
            for name, config in self.characters.items()
        ])
        
        # Format conversation for director in prefill format
        recent_turns = []
        for turn in conversation_history[-10:]:  # Last 10 turns
            role = turn.get("role", "user")
            content = turn.get("content", "")
            character = turn.get("character")
            
            if role == "user":
                # Use 'H' for human in prefill format
                recent_turns.append(f"H: {content}")
            elif role == "assistant":
                # Use character name or prefill name if available
                if character:
                    char_config = self.characters.get(character)
                    if char_config and char_config.prefill_name:
                        recent_turns.append(f"{char_config.prefill_name}: {content}")
                    else:
                        recent_turns.append(f"{character}: {content}")
                else:
                    # Fallback if no character info
                    recent_turns.append(f"Assistant: {content}")
        
        # Join with double newlines for prefill format
        conversation_text = "\n\n".join(recent_turns)
        
        # Create director prompt
        system_prompt = self.director_config.system_prompt.format(participants=participants)
        
        # For director, present the conversation as context, then ask the question
        if conversation_text:
            user_prompt = conversation_text + "\n\nWho should speak next? Respond with just the name of the participant."
        else:
            user_prompt = "This is the start of a conversation. Who should speak first? Respond with just the name of the participant."
        
        # Log director request
        request_timestamp = time.time()
        llm_logs_dir = Path("llm_logs")
        llm_logs_dir.mkdir(exist_ok=True)
        
        # Create request log
        request_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(request_timestamp * 1000) % 1000}_director_request.json"
        request_data = {
            "timestamp": request_timestamp,
            "datetime": datetime.now().isoformat(),
            "provider": self.director_config.llm_provider,
            "model": self.director_config.llm_model,
            "type": "director",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "participants": participants.split("\n")
        }
        
        with open(llm_logs_dir / request_filename, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
        print(f"📝 Director request logged: {request_filename}")
        
        try:
            if self.director_config.llm_provider == "openai":
                response = await self._director_client.chat.completions.create(
                    model=self.director_config.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,  # Low temperature for consistent decisions
                    max_tokens=10
                )
                next_speaker = response.choices[0].message.content.strip()
                
            elif self.director_config.llm_provider == "anthropic":
                response = await self._director_client.messages.create(
                    model=self.director_config.llm_model,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    system=system_prompt,
                    temperature=0.3,
                    max_tokens=10
                )
                next_speaker = response.content[0].text.strip()
            
            # Build list of valid options including prefill names
            valid_options = ["H", "USER"] + list(self.characters.keys())  # Accept both H and USER for compatibility
            for char_config in self.characters.values():
                if char_config.prefill_name and char_config.prefill_name not in valid_options:
                    valid_options.append(char_config.prefill_name)
            
            # Log director response
            response_timestamp = time.time()
            response_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(response_timestamp * 1000) % 1000}_director_response.json"
            response_data = {
                "timestamp": response_timestamp,
                "datetime": datetime.now().isoformat(),
                "provider": self.director_config.llm_provider,
                "model": self.director_config.llm_model,
                "type": "director",
                "request_file": request_filename,
                "response": next_speaker,
                "valid_options": valid_options,
                "selected_valid": next_speaker in valid_options
            }
            
            with open(llm_logs_dir / response_filename, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            print(f"📝 Director response logged: {response_filename} -> {next_speaker}")
            
            # Clean up response (take only first line if multiple)
            if '\n' in next_speaker:
                next_speaker = next_speaker.split('\n')[0].strip()
                print(f"📝 Director returned multiple lines, using first: {next_speaker}")
            
            # Validate response
            if next_speaker in ["USER", "H"]:
                return "USER"  # Both H and USER mean human turn
            
            # Check if it's a direct character name
            if next_speaker in self.characters:
                return next_speaker
            
            # Check if it's a prefill name
            for char_name, char_config in self.characters.items():
                if char_config.prefill_name and char_config.prefill_name == next_speaker:
                    print(f"🎭 Director used prefill name '{next_speaker}', mapping to character '{char_name}'")
                    return char_name
            
            print(f"⚠️ Director returned unknown speaker: {next_speaker}")
            return None
                
        except Exception as e:
            print(f"❌ Director error: {e}")
            return None
    
    def get_character_config(self, character_name: str) -> Optional[CharacterConfig]:
        """Get configuration for a specific character."""
        return self.characters.get(character_name)
    
    def get_character_client(self, character_name: str):
        """Get LLM client for a specific character."""
        return self._llm_clients.get(character_name)
    
    def set_active_character(self, character_name: Optional[str]):
        """Set the currently active character."""
        if character_name and character_name not in self.characters:
            raise ValueError(f"Unknown character: {character_name}")
        self.active_character = character_name
        
        if character_name:
            print(f"🎭 Active character: {character_name}")
    
    def add_turn_to_context(self, speaker: str, message: str):
        """Add a turn to the conversation context."""
        self.context.turn_history.append((speaker, message))
        
        # Keep only recent history to avoid unbounded growth
        if len(self.context.turn_history) > 100:
            self.context.turn_history = self.context.turn_history[-100:]
    
    def get_character_voice_settings(self, character_name: str) -> Dict[str, Any]:
        """Get voice settings for a character."""
        config = self.get_character_config(character_name)
        if not config:
            return {}
        
        return {
            "voice_id": config.voice_id,
            "speed": config.voice_settings.get("speed", 1.0),
            "stability": config.voice_settings.get("stability", 0.5),
            "similarity_boost": config.voice_settings.get("similarity_boost", 0.8)
        }
    
    def format_messages_for_character(self, character_name: str, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format conversation history for a specific character's perspective.
        
        Args:
            character_name: The character who will respond
            conversation_history: Full conversation history
            
        Returns:
            Formatted messages for the character's LLM
        """
        config = self.get_character_config(character_name)
        if not config:
            return []
        
        # Start with character's system prompt
        messages = [{"role": "system", "content": config.system_prompt}]
        
        # Add conversation history with character attribution
        for turn in conversation_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            character = turn.get("character")
            
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                # If this was spoken by current character, it's assistant
                # Otherwise, prefix with character name
                if character == character_name:
                    messages.append({"role": "assistant", "content": content})
                else:
                    # Other characters' messages are part of the conversation
                    # Use character name without brackets
                    if character:
                        messages.append({"role": "user", "content": f"{character}: {content}"})
                    else:
                        # Non-character assistant message - treat as generic context
                        messages.append({"role": "user", "content": content})
            elif role == "tool":
                # Tool results are system messages
                messages.append({"role": "system", "content": f"Tool result: {content}"})
        
        return messages


def create_character_manager(config: Dict[str, Any]) -> CharacterManager:
    """
    Create a character manager from configuration.
    
    Args:
        config: Configuration dictionary with 'characters', 'director', and 'api_keys' sections
        
    Returns:
        Configured CharacterManager instance
    """
    # Get API keys from the main config
    api_keys = config.get("api_keys", {})
    
    # Parse character configs
    characters = {}
    for char_name, char_data in config.get("characters", {}).items():
        # Add the appropriate API key based on provider
        if "api_key" not in char_data:
            provider = char_data.get("llm_provider", "")
            if provider == "openai":
                char_data["api_key"] = api_keys.get("openai")
            elif provider == "anthropic":
                char_data["api_key"] = api_keys.get("anthropic")
        
        characters[char_name] = CharacterConfig(
            name=char_name,
            **char_data
        )
    
    # Parse director config
    director_data = config.get("director", {})
    
    # Add API key to director if not present
    if "api_key" not in director_data:
        provider = director_data.get("llm_provider", "openai")
        if provider == "openai":
            director_data["api_key"] = api_keys.get("openai")
        elif provider == "anthropic":
            director_data["api_key"] = api_keys.get("anthropic")
    
    director_config = DirectorConfig(**director_data)
    
    return CharacterManager(characters, director_config)