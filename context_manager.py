#!/usr/bin/env python3
"""
Context Management System for Voice AI

Manages multiple conversation contexts with:
- Base history (initial state)
- Persistent state (current conversation)
- Auto-saving functionality
- Context switching and resetting
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging

# Import ConversationTurn from the main module
# Note: This requires careful import ordering to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unified_voice_conversation_config import ConversationTurn
else:
    # Define a compatible ConversationTurn for runtime
    @dataclass
    class ConversationTurn:
        """Represents a single turn in a conversation."""
        role: str  # "user", "assistant", "system", or "tool"
        content: Union[str, List[Dict[str, Any]]]  # String or list of content blocks
        timestamp: datetime
        status: str = "completed"  # "pending", "processing", "completed", "interrupted"
        speaker_id: Optional[int] = None
        speaker_name: Optional[str] = None  # Identified speaker name
        character: Optional[str] = None  # Character name for multi-character conversations
        metadata: Optional[Dict[str, Any]] = None  # Additional metadata
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert turn to dictionary for serialization."""
            # Handle datetime serialization
            timestamp_val = None
            if self.timestamp:
                if isinstance(self.timestamp, datetime):
                    timestamp_val = self.timestamp.isoformat()
                else:
                    timestamp_val = self.timestamp
                    
            return {
                'role': self.role,
                'content': self.content,
                'timestamp': timestamp_val,
                'status': self.status,
                'speaker_id': self.speaker_id,
                'speaker_name': self.speaker_name,
                'character': self.character,
                'metadata': self.metadata or {}
            }

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Represents a single conversation context."""
    name: str
    description: str = ""
    base_history: List[ConversationTurn] = field(default_factory=list)
    current_state: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_saved: Optional[datetime] = None
    
    def reset_to_base(self):
        """Reset current state to base history."""
        self.current_state = self.base_history.copy()
        logger.info(f"Context '{self.name}' reset to base history")
    
    def get_full_history(self) -> List[ConversationTurn]:
        """Get the complete conversation history (base + current state)."""
        return self.base_history + self.current_state
    
    def add_turn(self, turn: ConversationTurn):
        """Add a new turn to the current state."""
        self.current_state.append(turn)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'base_history': [turn.to_dict() for turn in self.base_history],
            'current_state': [turn.to_dict() for turn in self.current_state],
            'metadata': self.metadata,
            'last_saved': self.last_saved.isoformat() if self.last_saved else None
        }
    
    @staticmethod
    def _turn_from_dict(turn_data: Dict[str, Any]) -> 'ConversationTurn':
        """Create ConversationTurn from dictionary, handling timestamp conversion."""
        # Handle timestamp
        timestamp = turn_data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                # ISO format string
                timestamp = datetime.fromisoformat(timestamp)
            elif isinstance(timestamp, (int, float)):
                # Unix timestamp
                timestamp = datetime.fromtimestamp(timestamp)
            else:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()
            
        return ConversationTurn(
            role=turn_data.get('role', 'user'),
            content=turn_data.get('content', ''),
            timestamp=timestamp,
            status=turn_data.get('status', 'completed'),
            speaker_id=turn_data.get('speaker_id'),
            speaker_name=turn_data.get('speaker_name'),
            character=turn_data.get('character'),
            metadata=turn_data.get('metadata')
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create context from dictionary."""
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            base_history=[cls._turn_from_dict(turn) for turn in data.get('base_history', [])],
            current_state=[cls._turn_from_dict(turn) for turn in data.get('current_state', [])],
            metadata=data.get('metadata', {}),
            last_saved=datetime.fromisoformat(data['last_saved']) if data.get('last_saved') else None
        )


class ContextManager:
    """Manages multiple conversation contexts with persistent storage."""
    
    def __init__(self, contexts_config: Dict[str, Any], storage_dir: str = "contexts", history_parser=None):
        """
        Initialize context manager.
        
        Args:
            contexts_config: Configuration for available contexts
            storage_dir: Directory to store persistent states
            history_parser: Function to parse history files (optional)
        """
        self.storage_dir = storage_dir
        self.contexts: Dict[str, ConversationContext] = {}
        self.current_context_name: Optional[str] = None
        self._auto_save_task: Optional[asyncio.Task] = None
        self._auto_save_interval = contexts_config.get('auto_save_interval', 30)  # seconds
        self.history_parser = history_parser  # Function to parse markdown history files
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load contexts from configuration
        self._load_contexts_from_config(contexts_config)
        
        # Load persistent states
        self._load_persistent_states()
        
        # Set default context if specified
        default_context = contexts_config.get('default_context')
        if default_context and default_context in self.contexts:
            self.current_context_name = default_context
            logger.info(f"Set default context: {default_context}")
    
    def _load_contexts_from_config(self, contexts_config: Dict[str, Any]):
        """Load context definitions from configuration."""
        contexts_list = contexts_config.get('contexts', [])
        
        for context_config in contexts_list:
            name = context_config['name']
            
            # Load base history from file if specified
            base_history = []
            history_file = context_config.get('history_file')
            
            if history_file and self.history_parser:
                try:
                    # Use the provided history parser to load from markdown
                    parsed_history = self.history_parser(history_file)
                    
                    # Convert parsed messages to ConversationTurn format
                    for msg in parsed_history:
                        # Handle timestamp conversion
                        timestamp = msg.get('timestamp')
                        if timestamp and not isinstance(timestamp, datetime):
                            timestamp = datetime.fromtimestamp(timestamp) if isinstance(timestamp, (int, float)) else datetime.now()
                        elif not timestamp:
                            timestamp = datetime.now()
                            
                        turn = ConversationTurn(
                            role=msg.get('role', 'user'),
                            content=msg.get('content', ''),
                            timestamp=timestamp,
                            metadata=msg.get('metadata', {}),
                            speaker_name=msg.get('_speaker_name'),
                            character=msg.get('_speaker_name') if msg.get('role') == 'assistant' else None
                        )
                        
                        base_history.append(turn)
                    
                    logger.info(f"Loaded history from '{history_file}' for context '{name}'")
                except Exception as e:
                    logger.warning(f"Failed to load history file '{history_file}' for context '{name}': {e}")
            
            # Create context
            context = ConversationContext(
                name=name,
                description=context_config.get('description', ''),
                base_history=base_history,
                metadata=context_config.get('metadata', {})
            )
            
            # Initialize with base history
            context.reset_to_base()
            
            self.contexts[name] = context
            logger.info(f"Loaded context '{name}' with {len(base_history)} base turns")
    
    def _load_persistent_states(self):
        """Load saved persistent states from disk."""
        for context_name in self.contexts:
            state_file = os.path.join(self.storage_dir, f"{context_name}.json")
            
            if os.path.exists(state_file):
                try:
                    with open(state_file, 'r') as f:
                        data = json.load(f)
                    
                    # Update current state from saved data
                    saved_context = ConversationContext.from_dict(data)
                    self.contexts[context_name].current_state = saved_context.current_state
                    self.contexts[context_name].metadata.update(saved_context.metadata)
                    self.contexts[context_name].last_saved = saved_context.last_saved
                    
                    logger.info(f"Loaded persistent state for context '{context_name}'")
                except Exception as e:
                    logger.error(f"Error loading state for context '{context_name}': {e}")
    
    def save_context(self, context_name: Optional[str] = None):
        """Save a specific context or current context to disk."""
        if context_name is None:
            context_name = self.current_context_name
        
        if not context_name or context_name not in self.contexts:
            logger.warning(f"Cannot save invalid context: {context_name}")
            return
        
        context = self.contexts[context_name]
        context.last_saved = datetime.now()
        
        state_file = os.path.join(self.storage_dir, f"{context_name}.json")
        
        try:
            with open(state_file, 'w') as f:
                json.dump(context.to_dict(), f, indent=2)
            
            #logger.info(f"Saved context '{context_name}' to {state_file}")
        except Exception as e:
            logger.error(f"Error saving context '{context_name}': {e}")
    
    def save_all_contexts(self):
        """Save all contexts to disk."""
        for context_name in self.contexts:
            self.save_context(context_name)
    
    async def start_auto_save(self):
        """Start automatic saving of contexts."""
        if self._auto_save_task:
            return
        
        async def auto_save_loop():
            """Auto-save loop."""
            while True:
                try:
                    await asyncio.sleep(self._auto_save_interval)
                    self.save_all_contexts()
                    logger.debug("Auto-saved all contexts")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in auto-save: {e}")
        
        self._auto_save_task = asyncio.create_task(auto_save_loop())
        logger.info(f"Started auto-save with {self._auto_save_interval}s interval")
    
    async def stop_auto_save(self):
        """Stop automatic saving."""
        if self._auto_save_task:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass
            self._auto_save_task = None
            logger.info("Stopped auto-save")
        
        # Save all contexts one final time
        self.save_all_contexts()
    
    def switch_context(self, context_name: str) -> bool:
        """
        Switch to a different context.
        
        Args:
            context_name: Name of the context to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        if context_name not in self.contexts:
            logger.error(f"Context '{context_name}' not found")
            return False
        
        # Save current context before switching
        if self.current_context_name:
            self.save_context(self.current_context_name)
        
        self.current_context_name = context_name
        logger.info(f"Switched to context '{context_name}'")
        return True
    
    def reset_current_context(self):
        """Reset the current context to its base history."""
        if not self.current_context_name:
            logger.warning("No current context to reset")
            return
        
        context = self.contexts[self.current_context_name]
        context.reset_to_base()
        
        # Save the reset state
        self.save_context(self.current_context_name)
    
    def get_current_context(self) -> Optional[ConversationContext]:
        """Get the current active context."""
        if not self.current_context_name:
            return None
        return self.contexts.get(self.current_context_name)
    
    def get_current_history(self) -> List[ConversationTurn]:
        """Get the full history of the current context."""
        context = self.get_current_context()
        if not context:
            return []
        return context.get_full_history()
    
    def add_turn_to_current(self, turn: ConversationTurn):
        """Add a turn to the current context."""
        context = self.get_current_context()
        if context:
            context.add_turn(turn)
        else:
            logger.warning("No current context to add turn to")
    
    def get_context_list(self) -> List[Dict[str, str]]:
        """Get list of available contexts with their descriptions."""
        return [
            {
                'name': name,
                'description': context.description,
                'is_current': name == self.current_context_name,
                'turn_count': len(context.get_full_history())
            }
            for name, context in self.contexts.items()
        ]
    
    def export_context(self, context_name: str, file_path: str):
        """Export a context to a file."""
        if context_name not in self.contexts:
            raise ValueError(f"Context '{context_name}' not found")
        
        context = self.contexts[context_name]
        
        with open(file_path, 'w') as f:
            json.dump(context.to_dict(), f, indent=2)
        
        logger.info(f"Exported context '{context_name}' to {file_path}")
    
    def import_context(self, file_path: str, new_name: Optional[str] = None):
        """Import a context from a file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        context = ConversationContext.from_dict(data)
        
        # Use new name if provided
        if new_name:
            context.name = new_name
        
        # Add to contexts
        self.contexts[context.name] = context
        logger.info(f"Imported context '{context.name}' from {file_path}")
 