#!/usr/bin/env python3
"""Simulate image capture to test if images are included in requests."""

import asyncio
import base64
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_voice_conversation_config import UnifiedVoiceConversation, ConversationTurn
from config_loader import load_config
from datetime import datetime

async def test_image_in_conversation():
    """Test adding an image to conversation and checking if it's included in LLM request."""
    
    # Load config
    config = load_config("presets/opus_and_36.yaml")
    
    # Disable camera since it's not working
    config.camera.enabled = False
    
    # Create assistant
    assistant = UnifiedVoiceConversation(config)
    
    # Manually add a conversation turn with an image
    print("Adding conversation turn with simulated image...")
    
    # Create fake base64 image data (just a small red square)
    fake_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    # Add a user turn with image
    content = [
        {"type": "text", "text": "Look at this image"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": fake_image_data
            }
        }
    ]
    
    user_turn = ConversationTurn(
        role="user",
        content=content,
        timestamp=datetime.now(),
        status="pending"
    )
    
    assistant.state.conversation_history.append(user_turn)
    print("✅ Added user turn with image")
    
    # Now manually trigger processing
    print("\nTriggering LLM processing...")
    
    # Build messages like the assistant would
    messages = [{"role": "system", "content": config.conversation.system_prompt}]
    
    # Add conversation history
    for turn in assistant.state.conversation_history:
        if turn.role == "user" or turn.status in ["completed", "interrupted"]:
            messages.append({
                "role": turn.role,
                "content": turn.content
            })
    
    print(f"\nBuilt {len(messages)} messages")
    
    # Convert to prefill format
    if config.conversation.conversation_mode == "prefill":
        user_messages, assistant_prefix = assistant._convert_to_prefill_format(messages)
        print(f"\nPrefill conversion:")
        print(f"- User messages: {len(user_messages)}")
        for i, msg in enumerate(user_messages):
            if isinstance(msg['content'], list):
                print(f"  {i}: {msg['role']} - [List with {len(msg['content'])} items]")
                for j, item in enumerate(msg['content']):
                    if item.get('type') == 'image':
                        print(f"     - Image ({item.get('source', {}).get('media_type', 'unknown')})")
                    elif item.get('type') == 'text':
                        print(f"     - Text: {item.get('text', '')[:50]}...")
            else:
                print(f"  {i}: {msg['role']} - {msg['content'][:50]}...")
        print(f"- Assistant prefix length: {len(assistant_prefix)}")

if __name__ == "__main__":
    asyncio.run(test_image_in_conversation())