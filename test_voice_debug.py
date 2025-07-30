#!/usr/bin/env python3
"""Debug script for voice system."""
import asyncio
import sys
import os

async def main():
    """Run voice conversation with debug output."""
    print("🚀 Starting Voice Conversation Debug")
    print("=" * 50)
    
    # Set preset
    os.environ['VOICE_AI_PRESET'] = 'opus_and_36'
    
    try:
        # Import after setting env
        from unified_voice_conversation_config import main as voice_main
        
        # Run voice conversation
        await voice_main()
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