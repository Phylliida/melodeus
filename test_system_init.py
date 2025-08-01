#!/usr/bin/env python3
"""
Test the full system initialization with echo cancellation
"""

import asyncio
from config_loader import load_config
from unified_voice_conversation_config import UnifiedVoiceConversation

async def test_system_init():
    print("🔍 Testing Full System Initialization")
    print("=" * 50)
    
    try:
        # Load config
        config = load_config(preset="opus_and_36")
        print("✅ Configuration loaded")
        
        # Create conversation system
        print("\n🔧 Creating conversation system...")
        conversation = UnifiedVoiceConversation(config)
        
        # Check echo cancellation
        print("\n📊 Echo Cancellation Status:")
        print(f"   Config enabled: {config.conversation.enable_echo_cancellation}")
        print(f"   STT has echo_canceller: {hasattr(conversation.stt, 'echo_canceller')}")
        if hasattr(conversation.stt, 'echo_canceller'):
            print(f"   Echo canceller object: {conversation.stt.echo_canceller}")
        
        # Check if callback is set
        print(f"   TTS echo callback set: {conversation.tts.echo_cancellation_callback is not None}")
        
        print("\n✅ System initialization test completed!")
        
    except Exception as e:
        print(f"\n❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_system_init())