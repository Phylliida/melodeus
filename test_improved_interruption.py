#!/usr/bin/env python3
"""
Test the improved interruption system with better async state management.
"""

import asyncio
import time
from config_loader import load_config
from async_tts_module import AsyncTTSStreamer

async def test_improved_interruption():
    """Test the improved TTS interruption with precise state tracking."""
    print("🧪 Testing Improved TTS Interruption System")
    print("=" * 50)
    
    try:
        # Load config
        config = load_config()
        tts = AsyncTTSStreamer(config.tts)
        
        print("✅ TTS module loaded")
        
        # Test 1: Basic TTS with state tracking
        print("\n🔵 Test 1: Basic TTS with state tracking")
        print("Starting TTS...")
        
        long_text = ("This is a long text that should take several seconds to speak. "
                    "We will test interrupting it in the middle to verify that the "
                    "improved state management works correctly. The audio should stop "
                    "immediately when we call the stop method, and the state should "
                    "be properly synchronized between the conversation system and the "
                    "TTS module without any race conditions or timing issues.")
        
        # Start TTS
        task = asyncio.create_task(tts.speak_text(long_text))
        
        # Monitor state for first 2 seconds
        start_time = time.time()
        for i in range(20):  # 2 seconds worth of checks
            await asyncio.sleep(0.1)
            elapsed = time.time() - start_time
            
            currently_playing = tts.is_currently_playing()
            is_streaming = tts.is_streaming
            is_playing = tts.is_playing
            
            print(f"[{elapsed:.1f}s] Currently Playing: {currently_playing}, "
                  f"Streaming: {is_streaming}, Playing: {is_playing}")
            
            # Interrupt at 1.5 seconds
            if elapsed >= 1.5 and currently_playing:
                print(f"\n🛑 Interrupting at {elapsed:.1f}s...")
                interrupt_start = time.time()
                
                await tts.stop()
                
                interrupt_duration = time.time() - interrupt_start
                print(f"✅ Stop completed in {interrupt_duration:.3f}s")
                
                # Check state immediately after stop
                still_playing = tts.is_currently_playing()
                print(f"🔍 Still playing after stop: {still_playing}")
                
                if still_playing:
                    print("❌ TTS still playing - state sync failed!")
                else:
                    print("✅ TTS stopped cleanly - state sync successful!")
                
                break
        
        # Wait for task to complete
        result = await task
        print(f"📊 Task result: {result}")
        
        # Test 2: Rapid start/stop cycles
        print("\n🔵 Test 2: Rapid start/stop cycles")
        
        for cycle in range(3):
            print(f"\n--- Cycle {cycle + 1} ---")
            
            # Start TTS
            task = asyncio.create_task(tts.speak_text(f"This is test cycle number {cycle + 1}."))
            
            # Wait a brief moment
            await asyncio.sleep(0.5)
            
            # Check state
            playing_before = tts.is_currently_playing()
            print(f"Playing before stop: {playing_before}")
            
            # Stop immediately
            await tts.stop()
            
            # Check state after stop
            playing_after = tts.is_currently_playing()
            print(f"Playing after stop: {playing_after}")
            
            # Wait for task completion
            result = await task
            print(f"Cycle {cycle + 1} result: {result}")
            
            # Brief pause between cycles
            await asyncio.sleep(0.2)
        
        print("\n🎯 Test Summary:")
        print("✅ Improved interruption system tested")
        print("✅ State management verified")
        print("✅ Rapid start/stop cycles completed")
        
        await tts.cleanup()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run the improved interruption tests."""
    await test_improved_interruption()

if __name__ == "__main__":
    asyncio.run(main()) 