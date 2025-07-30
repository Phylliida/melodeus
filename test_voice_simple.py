#!/usr/bin/env python3
"""Simple test of voice system."""
import asyncio
import sys
import os

# Set preset
os.environ['VOICE_AI_PRESET'] = 'opus_and_36'

async def main():
    """Run voice conversation."""
    print("🚀 Starting Voice Conversation")
    print("=" * 50)
    
    from config_loader import load_config
    from unified_voice_conversation_config import UnifiedVoiceConversation
    
    try:
        # Load configuration
        print("📁 Loading configuration...")
        config = load_config()
        print("✅ Configuration loaded successfully!")
        
        # Create conversation system
        print("🔧 Creating conversation system...")
        conversation = UnifiedVoiceConversation(config)
        print("✅ Conversation system created!")
        
        # Start conversation
        print("🎙️ Starting conversation...")
        success = await conversation.start_conversation()
        if not success:
            print("❌ Failed to start conversation")
            return
            
        # Keep running until interrupted
        print("🎯 Conversation is active. Press Ctrl+C to exit.")
        while conversation.state.is_active:
            await asyncio.sleep(0.5)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)