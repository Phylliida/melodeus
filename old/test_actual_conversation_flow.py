#!/usr/bin/env python3
"""Test the actual conversation flow to see where images get lost."""

import asyncio
import json
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_voice_conversation_config import UnifiedVoiceConversation, ConversationTurn
from config_loader import load_config

async def test_conversation_flow():
    """Test the full conversation flow with a simulated image."""
    
    # Load config
    config = load_config("presets/opus_and_36.yaml")
    
    # Disable camera for simpler testing
    config.camera.enabled = False
    
    # Create assistant
    assistant = UnifiedVoiceConversation(config)
    
    # Clear conversation history for clean test
    assistant.state.conversation_history = []
    
    # Add a simple conversation with image
    print("Setting up test conversation...")
    
    # Add initial exchange
    assistant.state.conversation_history.append(ConversationTurn(
        role="user",
        content="Hello",
        timestamp=datetime.now(),
        status="completed"
    ))
    
    assistant.state.conversation_history.append(ConversationTurn(
        role="assistant", 
        content="Hi there!",
        timestamp=datetime.now(),
        status="completed"
    ))
    
    # Add user message with image
    fake_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    image_content = [
        {"type": "text", "text": "What do you see in this image?"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": fake_image_data
            }
        }
    ]
    
    assistant.state.conversation_history.append(ConversationTurn(
        role="user",
        content=image_content,
        timestamp=datetime.now(),
        status="completed"
    ))
    
    print("\nBuilding messages for LLM...")
    
    # Build messages like _process_with_llm does
    messages = [{"role": "system", "content": config.conversation.system_prompt}]
    
    for turn in assistant.state.conversation_history:
        if turn.role == "user" or turn.status in ["completed", "interrupted"]:
            messages.append({
                "role": turn.role,
                "content": turn.content
            })
    
    print(f"Built {len(messages)} messages")
    for i, msg in enumerate(messages):
        if isinstance(msg.get("content"), list):
            print(f"  {i}: {msg['role']} - [Image message with {len(msg['content'])} items]")
        else:
            content_preview = str(msg.get("content", ""))[:50]
            print(f"  {i}: {msg['role']} - {content_preview}...")
    
    # Now test prefill conversion
    print("\nTesting prefill conversion...")
    user_messages, assistant_prefix = assistant._convert_to_prefill_format(messages)
    
    print(f"\nAfter conversion:")
    print(f"User messages: {len(user_messages)}")
    for i, msg in enumerate(user_messages):
        if isinstance(msg.get("content"), list):
            print(f"  {i}: {msg['role']} - [Image message]")
            for j, item in enumerate(msg['content']):
                if item.get('type') == 'image':
                    print(f"     {j}: image - {item.get('source', {}).get('media_type', 'unknown')}")
                elif item.get('type') == 'text':
                    print(f"     {j}: text - {item.get('text', '')[:50]}...")
        else:
            print(f"  {i}: {msg['role']} - {msg['content'][:50]}...")
    
    print(f"\nAssistant prefix: {assistant_prefix[:100]}...")
    
    # Create the final prefill messages that would be logged
    prefill_messages = user_messages + [{"role": "assistant", "content": assistant_prefix}]
    
    # Save this as a test log
    test_log = {
        "original_messages": messages,
        "user_messages_after_conversion": user_messages,
        "assistant_prefix": assistant_prefix,
        "final_prefill_messages": prefill_messages
    }
    
    with open("test_conversation_flow_output.json", "w") as f:
        json.dump(test_log, f, indent=2)
    
    print("\n✅ Saved full output to test_conversation_flow_output.json")
    
    # Check if images are in final messages
    image_found = False
    for msg in prefill_messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    image_found = True
                    break
    
    if image_found:
        print("✅ Images ARE preserved in final prefill messages!")
    else:
        print("❌ Images were LOST in final prefill messages!")

if __name__ == "__main__":
    asyncio.run(test_conversation_flow())