#!/usr/bin/env python3
"""Test history file loading and parsing."""

import sys
from pathlib import Path
from pprint import pprint
from config_loader import load_config
from unified_voice_conversation_config import UnifiedVoiceConversation
from character_system import CharacterConfig

def test_parse_history_file(file_path: str):
    """Test parsing a history file."""
    print(f"\n{'='*60}")
    print(f"Testing history file: {file_path}")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_config()
    
    # Create a minimal instance just for testing parsing
    # We'll directly test the parsing method without full initialization
    from unified_voice_conversation_config import UnifiedVoiceConversation
    
    # Create a mock instance with minimal setup
    conv = type('MockConv', (), {})()
    conv.config = config
    conv.detected_speakers = set()
    
    # Bind the parsing method
    conv._parse_history_file = UnifiedVoiceConversation._parse_history_file.__get__(conv, type(conv))
    conv._is_valid_speaker_name = UnifiedVoiceConversation._is_valid_speaker_name.__get__(conv, type(conv))
    
    # Parse the history file
    messages = conv._parse_history_file(file_path)
    
    print(f"📊 Parsed {len(messages)} messages\n")
    
    # Show first few messages
    for i, msg in enumerate(messages[:5]):
        print(f"Message {i+1}:")
        print(f"  Role: {msg['role']}")
        print(f"  Content: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
        if '_speaker_name' in msg:
            print(f"  _speaker_name: {msg['_speaker_name']}")
        print()
    
    # Check for speaker names in content
    print("\n🔍 Checking for speaker patterns in content:")
    import re
    for i, msg in enumerate(messages[:10]):
        content = msg['content']
        speaker_match = re.match(r'^([A-Za-z0-9_\s\.]+):\s*(.*)', content)
        if speaker_match:
            print(f"  Message {i+1} ({msg['role']}): Speaker '{speaker_match.group(1)}' found in content")
    
    # Test character resolution simulation
    print("\n🎭 Testing character resolution (simulated):")
    
    # Simulate what would happen with real characters
    # Based on opus_and_36.yaml config
    mock_characters = {
        "Opus": {"prefill_name": "Claude 3 Opus"},
        "Sonnet": {"prefill_name": "Supreme Sonnet"}
    }
    
    print("Mock characters:")
    for name, char in mock_characters.items():
        print(f"  - {name} (prefill: {char['prefill_name']})")
    
    print("\nResolution results:")
    for msg in messages[:10]:
        if msg.get('_speaker_name'):
            speaker = msg['_speaker_name']
            resolved = None
            
            # Try to match character
            for char_name, char_data in mock_characters.items():
                if (speaker == char_name or speaker == char_data['prefill_name']):
                    resolved = char_name
                    break
            
            print(f"  '{speaker}' → {resolved or 'Not found'}")
    
    return messages

def test_conversation_history_loading(file_path: str):
    """Test full conversation history loading."""
    print(f"\n{'='*60}")
    print(f"Testing full history loading")
    print(f"{'='*60}\n")
    
    # Load config with history file
    config = load_config()
    config.conversation.history_file = file_path
    
    # Create conversation instance (this loads history)
    conv = UnifiedVoiceConversation(config)
    
    print(f"\n📚 Loaded {len(conv.state.conversation_history)} turns\n")
    
    # Check first few turns
    for i, turn in enumerate(conv.state.conversation_history[:5]):
        print(f"Turn {i+1}:")
        print(f"  Role: {turn.role}")
        print(f"  Content: {turn.content[:100]}{'...' if len(turn.content) > 100 else ''}")
        print(f"  Character: {turn.character}")
        print(f"  Status: {turn.status}")
        print()
    
    # Test director formatting
    print("\n🎬 Testing director formatting:")
    director_history = conv._get_conversation_history_for_director()
    
    # Show what director sees
    print(f"Director sees {len(director_history)} messages\n")
    
    # Format like director does
    from character_system import CharacterManager
    recent_turns = []
    for turn in director_history[:5]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        character = turn.get("character")
        
        if role == "user":
            recent_turns.append(f"H: {content[:50]}...")
        elif role == "assistant":
            if character:
                # This is what would happen in director
                recent_turns.append(f"{character}: {content[:50]}...")
            else:
                # Check if content already has speaker
                import re
                speaker_match = re.match(r'^([A-Za-z0-9_\s\.]+):\s*(.*)', content)
                if speaker_match:
                    recent_turns.append(content[:50] + "...")
                else:
                    recent_turns.append(f"Assistant: {content[:50]}...")
    
    print("Director formatted turns:")
    for turn in recent_turns:
        print(f"  {turn}")
    
    return conv

def main():
    """Main test function."""
    if len(sys.argv) < 2:
        print("Usage: python test_history_loading.py <history_file>")
        print("\nExample:")
        print("  python test_history_loading.py opus2.md")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    # Test parsing only (full loading requires API keys)
    messages = test_parse_history_file(file_path)
    
    print("\n✅ Parsing test complete!")
    print("\nNote: Full conversation loading test skipped (requires API keys)")
    print("To test with real conversation loading, ensure all API keys are configured.")

if __name__ == "__main__":
    main()