#!/usr/bin/env python3
"""Test prefill format conversion with images."""

import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_voice_conversation_config import UnifiedVoiceConversation
from config_loader import load_config

def test_prefill_conversion():
    """Test the prefill conversion with image messages."""
    
    # Load config
    config = load_config("presets/opus_and_36.yaml")
    assistant = UnifiedVoiceConversation(config)
    
    # Create test messages with images
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "fake_base64_data_here"
                    }
                }
            ]
        },
        {"role": "assistant", "content": "I can see an image here."},
        {"role": "user", "content": "Tell me more"},
    ]
    
    # Test conversion
    user_messages, assistant_prefix = assistant._convert_to_prefill_format(test_messages)
    
    print("PREFILL CONVERSION TEST")
    print("=" * 50)
    print("\nOriginal messages:")
    for i, msg in enumerate(test_messages):
        print(f"{i}: {msg['role']} - {str(msg['content'])[:100]}...")
    
    print("\n\nConverted to prefill format:")
    print(f"\nUser messages ({len(user_messages)}):")
    for i, msg in enumerate(user_messages):
        if isinstance(msg['content'], str):
            print(f"{i}: {msg['role']} - {msg['content'][:100]}...")
        else:
            print(f"{i}: {msg['role']} - [List with {len(msg['content'])} items]")
            for j, item in enumerate(msg['content']):
                if item.get('type') == 'text':
                    print(f"   {j}: text - {item['text'][:50]}...")
                elif item.get('type') == 'image':
                    print(f"   {j}: image - {item.get('source', {}).get('media_type', 'unknown')}")
    
    print(f"\nAssistant prefix: {assistant_prefix[:100]}...")
    
    # Save as JSON for inspection
    output = {
        "original_messages": test_messages,
        "user_messages": user_messages,
        "assistant_prefix": assistant_prefix
    }
    
    with open("test_prefill_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n✅ Saved full output to test_prefill_output.json")

if __name__ == "__main__":
    test_prefill_conversion()