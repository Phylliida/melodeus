#!/usr/bin/env python3
"""Add debug logging to track images through the conversation flow."""

import re

# Read the files
with open('unified_voice_conversation_config.py', 'r') as f:
    config_content = f.read()

with open('character_system.py', 'r') as f:
    character_content = f.read()

# Add debug logging to _on_utterance_complete to log when images are captured
utterance_pattern = r'(print\(f"💬 Added user utterance: \'\{result\.text\}\'" \+ \(" with image" if captured_image else ""\)\))'
utterance_replacement = r'''\1
        # DEBUG: Log image capture details
        if captured_image:
            print(f"🔍 DEBUG: Image captured, base64 length: {len(captured_image)}")
            print(f"🔍 DEBUG: Turn content type: {type(content)}")'''

config_content = re.sub(utterance_pattern, utterance_replacement, config_content)

# Add debug logging to _get_conversation_history_for_character
history_pattern = r'(history\.append\(entry\))'
history_replacement = r'''# DEBUG: Log image content
                if isinstance(entry["content"], list):
                    for item in entry["content"]:
                        if isinstance(item, dict) and item.get("type") == "image":
                            print(f"🔍 DEBUG: Including image in character history - {turn.role} turn")
                \1'''

config_content = re.sub(history_pattern, history_replacement, config_content, flags=re.MULTILINE)

# Add debug logging to format_messages_for_character in character_system.py
format_pattern = r'(messages\.append\(\{\s*"role": turn\.role,\s*"content": content\s*\}\))'
format_replacement = r'''# DEBUG: Log when images are in messages
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        print(f"🔍 DEBUG: Image found in character messages - {turn.role} turn")
            \1'''

character_content = re.sub(format_pattern, format_replacement, character_content, flags=re.MULTILINE | re.DOTALL)

# Write the modified files
with open('unified_voice_conversation_config_debug.py', 'w') as f:
    f.write(config_content)

with open('character_system_debug.py', 'w') as f:
    f.write(character_content)

print("✅ Created debug versions:")
print("  - unified_voice_conversation_config_debug.py")
print("  - character_system_debug.py")
print("\nTo use these, temporarily rename them to replace the originals.")
print("The debug versions will log when images are captured and included in messages.")