#!/usr/bin/env python3
"""Debug history file parsing."""

import re
from pathlib import Path

def debug_parse_history(file_path: str):
    """Debug parsing of history file."""
    print(f"Parsing: {file_path}\n")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    lines = content.split('\n')
    print(f"Total lines: {len(lines)}\n")
    
    # Track state
    current_speaker = None
    current_content = []
    messages = []
    
    for i, line in enumerate(lines[:50]):  # First 50 lines for debugging
        line = line.strip()
        
        # Try to match speaker
        speaker_match = re.match(r'^([A-Za-z0-9_\s\.]+):\s*(.*)', line)
        
        if speaker_match:
            speaker_name = speaker_match.group(1).strip()
            message_content = speaker_match.group(2).strip()
            
            print(f"Line {i+1}: SPEAKER MATCH")
            print(f"  Speaker: '{speaker_name}'")
            print(f"  Content start: '{message_content[:50]}{'...' if len(message_content) > 50 else ''}'")
            
            # Check if valid speaker
            # Exclude patterns like numbers, bullets, etc.
            if not re.match(r'^\d+\.|^-\s|^\*\*|^Step\s|^Panel\s|^Note\s|^Chapter\s', speaker_name):
                print(f"  ✓ Valid speaker")
                
                # Save previous if exists
                if current_speaker:
                    msg_content = '\n'.join(current_content).strip()
                    print(f"  💾 Saving previous: {current_speaker} ({len(msg_content)} chars)")
                    messages.append((current_speaker, msg_content))
                
                # Start new
                current_speaker = speaker_name
                current_content = [message_content] if message_content else []
            else:
                print(f"  ✗ Invalid speaker pattern")
                if current_speaker:
                    current_content.append(line)
        else:
            if line and current_speaker:
                print(f"Line {i+1}: CONTINUATION ({len(line)} chars)")
                current_content.append(line)
            elif line:
                print(f"Line {i+1}: NO SPEAKER, SKIPPING: '{line[:50]}...'")
    
    # Final message
    if current_speaker and current_content:
        msg_content = '\n'.join(current_content).strip()
        print(f"\n💾 Saving final: {current_speaker} ({len(msg_content)} chars)")
        messages.append((current_speaker, msg_content))
    
    print(f"\n📊 Total messages found: {len(messages)}")
    print("\nFirst 5 messages:")
    for i, (speaker, content) in enumerate(messages[:5]):
        print(f"\n{i+1}. {speaker}:")
        print(f"   {content[:100]}{'...' if len(content) > 100 else ''}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_history_parser_debug.py <history_file>")
        sys.exit(1)
    
    debug_parse_history(sys.argv[1])