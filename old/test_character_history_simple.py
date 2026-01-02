#!/usr/bin/env python3
"""
Simple test to demonstrate character-specific history file parsing
"""

import os
from pathlib import Path

# Create test character history files
def create_test_history_files():
    """Create example character history files for testing."""
    
    # Create a test directory
    os.makedirs("test_histories", exist_ok=True)
    
    # Create Claude's history file
    with open("test_histories/claude_history.txt", "w") as f:
        f.write("""User: What's your favorite programming language?

Claude: As an AI assistant, I find Python particularly elegant and versatile. Its clear syntax makes it excellent for both beginners and experienced developers, and its vast ecosystem of libraries enables everything from web development to machine learning.

User: Can you give me an example of Python code?

Claude: Here's a simple example that demonstrates Python's readability:

def greet_user(name, times=1):
    '''Greet a user multiple times'''
    for i in range(times):
        print(f"Hello, {name}! Welcome to Python!")

This showcases function definitions, default parameters, f-strings, and loops - all with very clean syntax.""")
    
    # Create GPT's history file  
    with open("test_histories/gpt_history.txt", "w") as f:
        f.write("""User: What do you think about web development?

GPT: Web development is a fascinating and rapidly evolving field! The modern web ecosystem offers incredible possibilities with frameworks like React, Vue, and Angular for frontend development, and Node.js, Django, and Rails for backend work.

User: Which framework would you recommend for beginners?

GPT: For beginners, I'd recommend starting with React for frontend development. While it has a learning curve, its component-based architecture teaches good programming practices.""")
    
    print("✅ Created test character history files")

def parse_character_history_file(file_path: str, character_name: str):
    """Parse a character-specific history file (simplified version for testing)."""
    
    if not Path(file_path).exists():
        print(f"⚠️ Character history file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        messages = []
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Simple parsing: look for "Speaker: content" pattern
            if ': ' in line:
                speaker, message = line.split(': ', 1)
                
                # Collect multi-line messages
                full_message = [message]
                i += 1
                
                # Continue collecting lines until we hit another speaker or end
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line or (': ' in next_line and any(next_line.startswith(s + ': ') 
                                                                   for s in ['User', 'Human', 'H', character_name])):
                        break
                    full_message.append(next_line)
                    i += 1
                
                # Determine role
                if speaker.lower() in ['user', 'human', 'h']:
                    role = 'user'
                else:
                    role = 'assistant'
                
                messages.append({
                    'role': role,
                    'content': '\n'.join(full_message).strip()
                })
            else:
                i += 1
        
        print(f"📚 Loaded {len(messages)} messages from character history: {file_path}")
        return messages
        
    except Exception as e:
        print(f"❌ Error parsing character history file {file_path}: {e}")
        return []

def test_character_history_parsing():
    """Test the character history parsing functionality."""
    
    # Create test files
    create_test_history_files()
    
    print("\n" + "="*60)
    print("Testing character-specific history parsing")
    print("="*60)
    
    # Test parsing for Claude
    print("\n📚 Parsing Claude's history:")
    claude_messages = parse_character_history_file("test_histories/claude_history.txt", "Claude")
    
    print(f"\nFound {len(claude_messages)} messages:")
    for i, msg in enumerate(claude_messages):
        print(f"\nMessage {i+1}:")
        print(f"  Role: {msg['role']}")
        print(f"  Content preview: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
    
    # Test parsing for GPT
    print("\n" + "-"*40)
    print("\n📚 Parsing GPT's history:")
    gpt_messages = parse_character_history_file("test_histories/gpt_history.txt", "GPT")
    
    print(f"\nFound {len(gpt_messages)} messages:")
    for i, msg in enumerate(gpt_messages):
        print(f"\nMessage {i+1}:")
        print(f"  Role: {msg['role']}")
        print(f"  Content preview: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
    
    print("\n" + "="*60)
    print("✅ Test complete!")
    print("\nHow it works:")
    print("1. Each character can have their own history file configured")
    print("2. When formatting messages for a character, their specific history is prepended")
    print("3. This gives each character unique context and personality based on past interactions")
    print("4. The common history file is still used for the current conversation context")

if __name__ == "__main__":
    test_character_history_parsing()