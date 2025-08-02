#!/usr/bin/env python3
"""
Test script to demonstrate character-specific history files
"""

import os
import asyncio
from character_system import CharacterConfig, DirectorConfig, CharacterManager

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

```python
def greet_user(name, times=1):
    '''Greet a user multiple times'''
    for i in range(times):
        print(f"Hello, {name}! Welcome to Python!")

# Usage
greet_user("Alice", 3)
```

This showcases function definitions, default parameters, f-strings, and loops - all with very clean syntax.

User: That's helpful, thanks!

Claude: You're welcome! Python's philosophy of "readability counts" really shines through in examples like this. Feel free to ask if you'd like to explore any other Python concepts!""")
    
    # Create GPT's history file  
    with open("test_histories/gpt_history.txt", "w") as f:
        f.write("""User: What do you think about web development?

GPT: Web development is a fascinating and rapidly evolving field! The modern web ecosystem offers incredible possibilities with frameworks like React, Vue, and Angular for frontend development, and Node.js, Django, and Rails for backend work. The constant innovation keeps things exciting.

User: Which framework would you recommend for beginners?

GPT: For beginners, I'd recommend starting with React for frontend development. While it has a learning curve, its component-based architecture teaches good programming practices, and its massive community means excellent documentation and support. For full-stack development, Next.js (built on React) provides an excellent all-in-one solution.

User: What about backend frameworks?

GPT: For backend beginners, Express.js with Node.js is a great starting point if you already know JavaScript. If you're open to learning a new language, Django (Python) is fantastic because it includes everything you need out of the box - authentication, admin panel, ORM, and more. Both have gentle learning curves and strong communities.""")
    
    print("✅ Created test character history files")

async def test_character_histories():
    """Test the character-specific history functionality."""
    
    # Create test history files
    create_test_history_files()
    
    # Define test characters with history files
    claude_config = CharacterConfig(
        name="Claude",
        llm_provider="anthropic",
        llm_model="claude-3-sonnet-20241022",
        voice_id="test_voice_1",
        system_prompt="You are Claude, a helpful AI assistant created by Anthropic.",
        character_history_file="test_histories/claude_history.txt"
    )
    
    gpt_config = CharacterConfig(
        name="GPT",
        llm_provider="openai", 
        llm_model="gpt-4",
        voice_id="test_voice_2",
        system_prompt="You are GPT, a helpful AI assistant created by OpenAI.",
        character_history_file="test_histories/gpt_history.txt"
    )
    
    director_config = DirectorConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )
    
    # Create character manager
    characters = {
        "Claude": claude_config,
        "GPT": gpt_config
    }
    
    manager = CharacterManager(characters, director_config)
    
    # Load character histories
    manager.load_character_histories()
    
    # Test formatting messages for each character
    test_conversation = [
        {"role": "user", "content": "Tell me about your experience with coding."}
    ]
    
    print("\n" + "="*60)
    print("Testing character-specific histories")
    print("="*60)
    
    for char_name in ["Claude", "GPT"]:
        print(f"\n📚 Formatting messages for {char_name}:")
        messages = manager.format_messages_for_character(char_name, test_conversation)
        
        print(f"Total messages: {len(messages)}")
        
        # Show the character's history that was prepended
        if char_name in manager._character_histories:
            history_count = len(manager._character_histories[char_name])
            print(f"Character history messages: {history_count}")
            
            # Show first few history messages
            for i, msg in enumerate(manager._character_histories[char_name][:2]):
                print(f"\n  History message {i+1}:")
                print(f"    Role: {msg['role']}")
                print(f"    Content: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
        
        print(f"\nCurrent conversation appended: {len(test_conversation)} messages")
    
    print("\n✅ Character history test complete!")

if __name__ == "__main__":
    asyncio.run(test_character_histories())