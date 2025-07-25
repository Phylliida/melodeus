#!/usr/bin/env python3
"""
Test script to verify STT connection handling and recovery.
"""

import asyncio
import os
from config_loader import load_config
from async_stt_module import AsyncSTTStreamer, STTEventType, STTResult

async def test_connection_handling():
    """Test STT connection handling and error recovery."""
    print("🧪 Testing STT Connection Handling")
    print("==================================")
    
    try:
        # Load config
        try:
            config = load_config()
            api_key = config.stt.api_key
            print("✅ Loaded API key from config")
        except:
            print("⚠️  No config file - this test requires a valid Deepgram API key")
            return
            
        # Create STT instance
        from async_stt_module import STTConfig
        stt_config = STTConfig(
            api_key=api_key, 
            model="nova-3",
            chunk_size=4000,  # Smaller chunks for testing
            sample_rate=16000
        )
        stt = AsyncSTTStreamer(stt_config)
        
        # Track events
        events_received = []
        connection_states = []
        
        async def on_connection_opened(data):
            events_received.append("opened")
            connection_states.append("opened")
            print("📡 Connection opened event received")
        
        async def on_connection_closed(data):
            events_received.append("closed")
            connection_states.append("closed")
            print(f"🔌 Connection closed event received: {data.get('message', 'Unknown')}")
        
        async def on_error(data):
            events_received.append("error")
            print(f"❌ Error event received: {data.get('error', 'Unknown')}")
        
        async def on_utterance(result: STTResult):
            print(f"🎯 Utterance: {result.text}")
        
        # Register callbacks
        stt.on(STTEventType.CONNECTION_OPENED, on_connection_opened)
        stt.on(STTEventType.CONNECTION_CLOSED, on_connection_closed)
        stt.on(STTEventType.ERROR, on_error)
        stt.on(STTEventType.UTTERANCE_COMPLETE, on_utterance)
        
        print("✅ Event callbacks registered")
        
        # Test 1: Basic connection
        print("\n🔵 Test 1: Basic Connection Test")
        print("Attempting to start STT connection...")
        
        success = await stt.start_listening()
        if success:
            print("✅ STT started successfully")
            print(f"🔗 Connection alive: {stt.connection_alive}")
            print(f"🎤 Is listening: {stt.is_listening}")
            
            # Let it run for a few seconds
            print("⏳ Running for 10 seconds...")
            for i in range(10):
                await asyncio.sleep(1)
                if not stt.connection_alive:
                    print(f"⚠️  Connection died after {i+1} seconds")
                    break
                print(f"⏱️  Second {i+1}: Connection alive: {stt.connection_alive}")
            
            print("🛑 Stopping STT...")
            await stt.stop_listening()
            print("✅ STT stopped")
            
        else:
            print("❌ Failed to start STT")
            return
        
        print(f"\n📊 Events received: {events_received}")
        print(f"📊 Connection states: {connection_states}")
        
        # Test 2: Check cleanup
        print("\n🔵 Test 2: Cleanup Test")
        await stt.cleanup()
        print("✅ Cleanup completed")
        
        # Summary
        print("\n📋 Test Summary:")
        if "opened" in events_received:
            print("   ✅ Connection open event received")
        else:
            print("   ❌ Connection open event missing")
            
        if "closed" in events_received:
            print("   ✅ Connection close event received")
        else:
            print("   ❌ Connection close event missing")
        
        if "error" in events_received:
            print("   ⚠️  Error events were received (check details above)")
        else:
            print("   ✅ No error events (good!)")
        
        return len(events_received) > 0
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run connection handling tests."""
    print("🔬 STT Connection Handling Test")
    print("===============================")
    print()
    
    success = await test_connection_handling()
    
    print("\n🎯 Result:")
    if success:
        print("✅ Connection handling test completed successfully!")
        print("💡 Check the output above for any connection issues or errors.")
    else:
        print("❌ Connection handling test failed.")
        print("💡 Check your Deepgram API key and internet connection.")

if __name__ == "__main__":
    asyncio.run(main()) 