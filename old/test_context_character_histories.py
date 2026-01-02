#!/usr/bin/env python3
"""
Test script to demonstrate context-specific character histories
"""

import os
from pathlib import Path
from datetime import datetime
from context_manager import ContextConfig, ConversationContext, ContextManager, ConversationTurn

def create_test_histories():
    """Create test history files for demonstration."""
    
    # Create directory structure
    os.makedirs("test_histories/coding_context", exist_ok=True)
    os.makedirs("test_histories/creative_context", exist_ok=True)
    
    # Common history files for each context
    with open("test_histories/coding_conversation.txt", "w") as f:
        f.write("""H: Let's talk about software architecture.

Claude 3 Opus: I'd be happy to discuss software architecture! This is a broad topic that encompasses many important concepts.""")
    
    with open("test_histories/creative_conversation.txt", "w") as f:
        f.write("""H: I want to write a story about time travel.

Claude 3 Opus: Time travel stories offer fascinating narrative possibilities! Let's explore some creative approaches.""")
    
    # Character-specific histories for coding context
    with open("test_histories/coding_context/claude_coding_history.txt", "w") as f:
        f.write("""User: What's your approach to clean code?

Claude: I believe clean code should prioritize readability and maintainability. Key principles include:
- Meaningful variable and function names
- Single responsibility principle
- DRY (Don't Repeat Yourself)
- Clear comments for complex logic
- Consistent formatting

User: Can you show an example?

Claude: Here's a Python example demonstrating clean code principles:

def calculate_user_discount(user, purchase_amount):
    \"\"\"Calculate discount based on user loyalty status.\"\"\"
    LOYALTY_DISCOUNT_RATE = 0.10
    BULK_DISCOUNT_THRESHOLD = 100
    BULK_DISCOUNT_RATE = 0.05
    
    discount = 0
    
    if user.is_loyalty_member:
        discount += purchase_amount * LOYALTY_DISCOUNT_RATE
        
    if purchase_amount > BULK_DISCOUNT_THRESHOLD:
        discount += purchase_amount * BULK_DISCOUNT_RATE
        
    return discount""")
    
    with open("test_histories/coding_context/gpt_coding_history.txt", "w") as f:
        f.write("""User: What are your thoughts on microservices?

GPT: Microservices architecture is powerful but comes with trade-offs. Benefits include:
- Independent deployment and scaling
- Technology diversity
- Fault isolation
- Team autonomy

However, challenges include:
- Network complexity
- Distributed system debugging
- Data consistency
- Operational overhead

User: When would you recommend microservices?

GPT: I'd recommend microservices when:
1. You have a large team that needs to work independently
2. Different parts of your system have vastly different scaling needs
3. You need to use different tech stacks for different components
4. Your organization has mature DevOps practices

Start with a monolith and extract microservices as needed.""")
    
    # Character-specific histories for creative context
    with open("test_histories/creative_context/claude_creative_history.txt", "w") as f:
        f.write("""User: How do you approach creative writing?

Claude: My approach to creative writing focuses on:
- Building vivid, sensory-rich descriptions
- Creating complex, relatable characters
- Weaving themes naturally into the narrative
- Maintaining consistent voice and tone

I particularly enjoy exploring philosophical questions through storytelling.

User: Can you give an example of vivid description?

Claude: Here's a descriptive passage:

The autumn leaves crunched beneath her boots like ancient parchment, each step releasing the earthy scent of decay and renewal. The setting sun painted the forest in shades of amber and rust, while shadows stretched like dark fingers between the trees, reaching for something just beyond sight.""")
    
    with open("test_histories/creative_context/gpt_creative_history.txt", "w") as f:
        f.write("""User: What's your creative writing style?

GPT: My creative writing style tends to be:
- Character-driven with strong dialogue
- Plot-focused with unexpected twists
- Rich in metaphor and symbolism
- Influenced by classic and contemporary literature

I enjoy experimenting with narrative structure and perspective.

User: How do you create memorable characters?

GPT: I create memorable characters by:
1. Giving them contradictions - nobody is purely good or evil
2. Developing unique speech patterns and mannerisms
3. Creating compelling backstories that inform their actions
4. Ensuring they have clear motivations and fears
5. Making them change throughout the story""")
    
    print("✅ Created test history files")

def test_context_character_histories():
    """Test the context-specific character history functionality."""
    
    # Create test files
    create_test_histories()
    
    print("\n" + "="*60)
    print("Testing Context-Specific Character Histories")
    print("="*60)
    
    # Create contexts with character-specific histories
    coding_context = ContextConfig(
        name="coding_context",
        history_file="test_histories/coding_conversation.txt",
        description="Technical coding discussions",
        character_histories={
            "Claude": "test_histories/coding_context/claude_coding_history.txt",
            "GPT": "test_histories/coding_context/gpt_coding_history.txt"
        }
    )
    
    creative_context = ContextConfig(
        name="creative_context",
        history_file="test_histories/creative_conversation.txt",
        description="Creative writing discussions",
        character_histories={
            "Claude": "test_histories/creative_context/claude_creative_history.txt",
            "GPT": "test_histories/creative_context/gpt_creative_history.txt"
        }
    )
    
    general_context = ContextConfig(
        name="general_context",
        history_file="test_histories/general_conversation.txt",
        description="General discussions",
        # No character-specific histories for this context
    )
    
    # Create context manager
    # Need to properly convert configs to dict format
    contexts_list = []
    for cfg in [coding_context, creative_context, general_context]:
        ctx_dict = {
            'name': cfg.name,
            'history_file': cfg.history_file,
            'description': cfg.description
        }
        if cfg.character_histories:
            ctx_dict['character_histories'] = cfg.character_histories
        contexts_list.append(ctx_dict)
    
    manager = ContextManager(
        contexts_config=contexts_list,
        state_dir="./test_context_states"
    )
    
    # Simulate loading histories
    def mock_parse_history(file_path):
        """Mock history parser for testing."""
        messages = []
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                # Simple parsing for demo
                parts = content.split('\n\n')
                for part in parts:
                    if ': ' in part:
                        speaker, text = part.split(': ', 1)
                        role = "user" if speaker == "H" else "assistant"
                        messages.append({
                            "role": role,
                            "content": part,
                            "_speaker_name": speaker if role == "assistant" else None
                        })
        except:
            pass
        return messages
    
    # Debug: print configs
    for ctx in manager.contexts.values():
        print(f"\nContext {ctx.config.name} config:")
        print(f"  history_file: {ctx.config.history_file}")
        print(f"  character_histories: {ctx.config.character_histories}")
    
    # Load histories
    manager.load_original_histories(mock_parse_history)
    
    # Test each context
    for context_name in ["coding_context", "creative_context", "general_context"]:
        print(f"\n{'='*40}")
        print(f"Context: {context_name}")
        print(f"{'='*40}")
        
        manager.switch_context(context_name)
        context = manager.get_active_context()
        
        if context:
            print(f"\nCommon history: {len(context.original_history)} turns")
            
            if context.character_histories:
                print(f"\nCharacter-specific histories:")
                for char_name, char_history in context.character_histories.items():
                    print(f"  {char_name}: {len(char_history)} turns")
                    
                    # Show first turn from character history
                    if char_history:
                        first_turn = char_history[0]
                        print(f"    First turn ({first_turn.role}): {first_turn.content[:80]}...")
            else:
                print("\nNo character-specific histories for this context")
    
    print("\n" + "="*60)
    print("✅ Test complete!")
    print("\nKey features demonstrated:")
    print("1. Each context has its own common history file")
    print("2. Each context can have character-specific history files")
    print("3. Characters get different 'memories' in different contexts")
    print("4. This allows characters to maintain context-appropriate knowledge")

if __name__ == "__main__":
    test_context_character_histories()