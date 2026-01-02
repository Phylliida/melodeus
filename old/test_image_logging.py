#!/usr/bin/env python3
"""Simple test to add debug logging for image flow."""

# Add debug print to _on_utterance_complete after line 895
debug_utterance = '''
        # DEBUG: Log image capture details
        if captured_image:
            print(f"🔍 DEBUG: Image captured, base64 length: {len(captured_image)}")
            print(f"🔍 DEBUG: Turn status: {user_turn.status}")
'''

# Add debug print to _get_conversation_history_for_character in the loop
debug_history = '''
                # DEBUG: Log content type
                if isinstance(entry["content"], list):
                    has_image = any(item.get("type") == "image" for item in entry["content"] if isinstance(item, dict))
                    print(f"🔍 DEBUG: Character history - {turn.role} turn has list content, has_image: {has_image}")
'''

# Add debug print to _process_with_character_llm after getting messages
debug_character = '''
            # DEBUG: Check messages for images
            image_count = 0
            for i, msg in enumerate(messages):
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "image":
                            image_count += 1
            if image_count > 0:
                print(f"🔍 DEBUG: Found {image_count} images in {len(messages)} messages for character {next_speaker}")
'''

print("To debug image flow, add these debug prints to unified_voice_conversation_config.py:")
print("\n1. After line 895 (in _on_utterance_complete):")
print(debug_utterance)
print("\n2. After line 1509 (in _get_conversation_history_for_character):")
print(debug_history)
print("\n3. After line 1073 (in _process_with_character_llm):")
print(debug_character)
print("\nThese will help track if images are being:")
print("- Captured from camera")
print("- Added to conversation history")
print("- Included in character messages")
print("- Logged to request files")