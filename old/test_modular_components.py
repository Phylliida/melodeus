#!/usr/bin/env python3
"""
Test script demonstrating modular STT and TTS components.
Shows how to use each module independently.
"""

import asyncio
from async_stt_module import AsyncSTTStreamer, STTConfig, STTEventType, STTResult
from async_tts_module import AsyncTTSStreamer, TTSConfig

async def test_stt_only():
    """Test STT module independently."""
    print("🎤 Testing STT Module")
    print("=" * 30)
    
    api_key = input("Enter Deepgram API key: ").strip()
    if not api_key:
        print("❌ API key required!")
        return
    
    config = STTConfig(api_key=api_key, interim_results=True)
    stt = AsyncSTTStreamer(config)
    
    # Set up callbacks
    async def on_utterance(result: STTResult):
        print(f"🎯 Utterance: {result.text} (confidence: {result.confidence:.2f})")
    
    def on_interim(result: STTResult):
        print(f"💭 Interim: {result.text}")
    
    async def on_speech_start(data):
        print("🎤 Speech started")
    
    async def on_speech_end(data):
        print("🔇 Speech ended")
    
    # Register callbacks
    stt.on(STTEventType.UTTERANCE_COMPLETE, on_utterance)
    stt.on(STTEventType.INTERIM_RESULT, on_interim)
    stt.on(STTEventType.SPEECH_STARTED, on_speech_start)
    stt.on(STTEventType.SPEECH_ENDED, on_speech_end)
    
    try:
        if await stt.start_listening():
            print("✅ STT started. Speak for 10 seconds...")
            await asyncio.sleep(10)
        else:
            print("❌ Failed to start STT")
    finally:
        await stt.cleanup()

async def test_tts_only():
    """Test TTS module independently."""
    print("\n🔊 Testing TTS Module")
    print("=" * 30)
    
    api_key = input("Enter ElevenLabs API key: ").strip()
    if not api_key:
        print("❌ API key required!")
        return
    
    voice_id = input("Enter voice ID (or press Enter for default): ").strip()
    if not voice_id:
        voice_id = "T2KZm9rWPG5TgXTyjt7E"
    
    config = TTSConfig(api_key=api_key, voice_id=voice_id)
    tts = AsyncTTSStreamer(config)
    
    try:
        # Test simple text
        print("🔊 Speaking simple text...")
        result = await tts.speak_text("Hello! This is a test of the TTS module.")
        print(f"✅ Simple text completed: {result}")
        
        await asyncio.sleep(1)
        
        # Test streaming text
        print("🔊 Speaking streaming text...")
        
        async def text_generator():
            words = ["This", "is", "streaming", "text", "being", "generated", "word", "by", "word."]
            for word in words:
                yield word + " "
                await asyncio.sleep(0.3)  # Simulate slow generation
        
        result = await tts.speak_stream(text_generator())
        print(f"✅ Streaming text completed: {result}")
        
        await asyncio.sleep(1)
        
        # Test interruption
        print("🔊 Testing interruption...")
        task = asyncio.create_task(
            tts.speak_text("This is a longer message that we will interrupt before it finishes completely.")
        )
        
        # Interrupt after 2 seconds
        await asyncio.sleep(2)
        await tts.stop()
        
        result = await task
        print(f"✅ Interruption test completed: {result}")
        
    finally:
        await tts.cleanup()

async def test_combined_simple():
    """Test STT and TTS together in a simple loop."""
    print("\n🔄 Testing Combined STT + TTS")
    print("=" * 40)
    
    deepgram_key = input("Enter Deepgram API key: ").strip()
    elevenlabs_key = input("Enter ElevenLabs API key: ").strip()
    
    if not all([deepgram_key, elevenlabs_key]):
        print("❌ Both API keys required!")
        return
    
    voice_id = input("Enter voice ID (or press Enter for default): ").strip()
    if not voice_id:
        voice_id = "T2KZm9rWPG5TgXTyjt7E"
    
    # Setup modules
    stt_config = STTConfig(api_key=deepgram_key, interim_results=False)
    tts_config = TTSConfig(api_key=elevenlabs_key, voice_id=voice_id)
    
    stt = AsyncSTTStreamer(stt_config)
    tts = AsyncTTSStreamer(tts_config)
    
    # Simple echo system
    async def echo_utterance(result: STTResult):
        print(f"🎯 Heard: {result.text}")
        print(f"🔊 Echoing back...")
        await tts.speak_text(f"You said: {result.text}")
    
    stt.on(STTEventType.UTTERANCE_COMPLETE, echo_utterance)
    
    try:
        if await stt.start_listening():
            print("✅ Echo system started!")
            print("💡 Say something and it will be echoed back.")
            print("⏰ Running for 30 seconds...")
            
            # Run for 30 seconds
            for i in range(30):
                await asyncio.sleep(1)
                if i % 10 == 9:
                    print(f"⏰ {30 - i - 1} seconds remaining...")
        else:
            print("❌ Failed to start STT")
    finally:
        await stt.cleanup()
        await tts.cleanup()

async def main():
    """Main test menu."""
    print("🧪 Modular Components Test Suite")
    print("================================")
    print()
    print("Choose a test:")
    print("1. STT Only")
    print("2. TTS Only") 
    print("3. Combined Echo System")
    print("4. All Tests")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        await test_stt_only()
    elif choice == "2":
        await test_tts_only()
    elif choice == "3":
        await test_combined_simple()
    elif choice == "4":
        await test_stt_only()
        await test_tts_only()
        await test_combined_simple()
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}") 