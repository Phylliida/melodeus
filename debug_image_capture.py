#!/usr/bin/env python3
"""Debug script to check if images are being captured and added to conversation."""

import asyncio
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_voice_conversation_config import UnifiedVoiceConversation
from config_loader import load_config

async def debug_conversation():
    """Check the current conversation state for images."""
    
    # Load config
    config = load_config("presets/opus_and_36.yaml")
    
    # Create assistant
    assistant = UnifiedVoiceConversation(config)
    
    print("Current conversation state:")
    print(f"Total turns: {len(assistant.state.conversation_history)}")
    
    # Look for any turns with image content
    image_turns = []
    for i, turn in enumerate(assistant.state.conversation_history):
        if isinstance(turn.content, list):
            # Check if any item is an image
            for item in turn.content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_turns.append((i, turn))
                    break
    
    print(f"\nFound {len(image_turns)} turns with images:")
    for idx, turn in image_turns:
        print(f"  Turn {idx}: {turn.role} at {turn.timestamp} - status: {turn.status}")
        if isinstance(turn.content, list):
            for item in turn.content:
                if item.get("type") == "text":
                    print(f"    Text: {item.get('text', '')[:50]}...")
                elif item.get("type") == "image":
                    print(f"    Image: {item.get('source', {}).get('media_type', 'unknown')}")
    
    # Check camera status
    if assistant.camera:
        print(f"\nCamera status: {assistant.camera.get_camera_info()}")
    else:
        print("\nCamera is not initialized")
    
    # Check last few conversation turns
    print("\nLast 5 conversation turns:")
    for turn in assistant.state.conversation_history[-5:]:
        content_preview = turn.content
        if isinstance(content_preview, list):
            content_preview = f"[List with {len(content_preview)} items]"
        elif isinstance(content_preview, str):
            content_preview = content_preview[:50] + "..."
        print(f"  {turn.role} ({turn.status}): {content_preview}")

if __name__ == "__main__":
    asyncio.run(debug_conversation())