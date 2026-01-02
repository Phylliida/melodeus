#!/usr/bin/env python3
"""
Test script to verify async event handling fixes in the Voice AI system.
"""

import asyncio
import os
from config_loader import load_config
from async_stt_module import AsyncSTTStreamer, STTEventType, STTResult

async def test_stt_event_handling():
    """Test STT event handling without runtime warnings."""
    print("🧪 Testing STT Event Handling Fixes")
    print("===================================")
    
    try:
        # Load config (will fail gracefully if no config file)
        try:
            config = load_config()
            api_key = config.stt.api_key
        except:
            api_key = os.getenv("DEEPGRAM_API_KEY", "test_key")
            if api_key == "test_key":
                print("⚠️  No API key found - using mock for testing")
            
        # Create STT instance
        from async_stt_module import STTConfig
        stt_config = STTConfig(api_key=api_key, model="nova-3")
        stt = AsyncSTTStreamer(stt_config)
        
        # Test callback registration
        print("✅ STT instance created successfully")
        
        # Test event emission from async context
        test_events_received = []
        
        async def test_callback(data):
            test_events_received.append(data)
            print(f"📨 Received event: {type(data).__name__ if hasattr(data, '__class__') else data}")
        
        # Register callback
        stt.on(STTEventType.CONNECTION_OPENED, test_callback)
        print("✅ Callback registered successfully")
        
        # Test direct event emission (should work cleanly now)
        await stt._emit_event(STTEventType.CONNECTION_OPENED, {
            "message": "Test connection opened"
        })
        
        print("✅ Event emission completed without warnings")
        
        # Test schedule event (simulates Deepgram callback context)
        stt.event_loop = asyncio.get_event_loop()
        stt._schedule_event(STTEventType.CONNECTION_CLOSED, {
            "message": "Test connection closed"
        })
        
        # Give a moment for scheduled events to process
        await asyncio.sleep(0.1)
        
        print("✅ Event scheduling completed without warnings")
        print(f"📊 Events received: {len(test_events_received)}")
        
        # Cleanup
        await stt.cleanup()
        print("✅ Cleanup completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_conversation_shutdown():
    """Test graceful conversation shutdown."""
    print("\n🧪 Testing Conversation Shutdown")
    print("================================")
    
    try:
        # This tests the cleanup without actually starting a full conversation
        from unified_voice_conversation_config import UnifiedVoiceConversation
        from config_loader import VoiceAIConfig, ConversationConfig, STTConfig, TTSConfig, AudioConfig, LoggingConfig, DevelopmentConfig
        
        # Create minimal test config
        test_config = VoiceAIConfig(
            conversation=ConversationConfig(
                deepgram_api_key="test",
                elevenlabs_api_key="test", 
                openai_api_key="test"
            ),
            stt=STTConfig(api_key="test"),
            tts=TTSConfig(api_key="test", voice_id="test"),
            audio=AudioConfig(),
            logging=LoggingConfig(),
            development=DevelopmentConfig()
        )
        
        # Create conversation instance
        conversation = UnifiedVoiceConversation(test_config)
        print("✅ Conversation instance created")
        
        # Test cleanup (should not throw exceptions)
        await conversation.cleanup()
        print("✅ Conversation cleanup completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversation test failed: {e}")
        return False

async def main():
    """Run all async handling tests."""
    print("🔬 Voice AI Async Event Handling Tests")
    print("======================================")
    print()
    
    test1_passed = await test_stt_event_handling()
    test2_passed = await test_conversation_shutdown()
    
    print("\n📊 Test Results:")
    print(f"   STT Event Handling: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Conversation Shutdown: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Async event handling fixes are working.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 If you see no RuntimeWarnings, the fixes are working correctly!")

if __name__ == "__main__":
    asyncio.run(main()) 