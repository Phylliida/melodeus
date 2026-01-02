#!/usr/bin/env python3
"""
Test the improved async STT module to verify connection and audio streaming works.
"""

import asyncio
import time
from config_loader import load_config
from async_stt_module import AsyncSTTStreamer, STTEventType, STTResult

async def test_async_stt_connection():
    """Test the improved STT connection with async fixes."""
    print("🧪 Testing Improved Async STT Connection")
    print("=" * 50)
    
    try:
        # Load config
        config = load_config()
        stt = AsyncSTTStreamer(config.stt)
        
        print("✅ STT module loaded")
        
        # Setup event handlers
        connection_opened = False
        audio_chunks_received = 0
        
        async def on_connection_opened(data):
            nonlocal connection_opened
            connection_opened = True
            print(f"✅ Connection opened: {data['message']}")
        
        async def on_interim_result(result: STTResult):
            nonlocal audio_chunks_received
            audio_chunks_received += 1
            print(f"💭 Interim #{audio_chunks_received}: {result.text} (conf: {result.confidence:.2f})")
        
        async def on_utterance_complete(result: STTResult):
            print(f"🎯 Final: {result.text} (conf: {result.confidence:.2f})")
        
        async def on_error(data):
            print(f"❌ Error: {data['error']}")
        
        async def on_connection_closed(data):
            print(f"🔌 Connection closed: {data['message']}")
        
        # Register callbacks
        stt.on(STTEventType.CONNECTION_OPENED, on_connection_opened)
        stt.on(STTEventType.INTERIM_RESULT, on_interim_result)
        stt.on(STTEventType.UTTERANCE_COMPLETE, on_utterance_complete)
        stt.on(STTEventType.ERROR, on_error)
        stt.on(STTEventType.CONNECTION_CLOSED, on_connection_closed)
        
        print("🔄 Starting STT connection...")
        start_time = time.time()
        
        # Test connection
        success = await stt.start_listening()
        connection_time = time.time() - start_time
        
        print(f"📊 Connection attempt took {connection_time:.2f}s")
        
        if success:
            print("✅ STT started successfully!")
            
            # Wait for connection to be confirmed
            for i in range(50):  # 5 seconds
                await asyncio.sleep(0.1)
                if connection_opened:
                    break
            
            if connection_opened:
                print("✅ Connection confirmed opened")
                
                # Monitor for 10 seconds
                print("👂 Listening for 10 seconds...")
                for i in range(100):  # 10 seconds
                    await asyncio.sleep(0.1)
                    
                    # Check if still connected
                    if not stt.is_currently_listening():
                        print("❌ Lost connection during test")
                        break
                    
                    # Show progress every 2 seconds
                    if i % 20 == 0 and i > 0:
                        elapsed = i / 10
                        print(f"⏱️  {elapsed:.0f}s elapsed, listening={stt.is_currently_listening()}, "
                              f"connection_alive={stt.connection_alive}")
                
                print(f"📊 Test completed. Received {audio_chunks_received} audio events")
                
            else:
                print("❌ Connection not confirmed within 5 seconds")
        else:
            print("❌ Failed to start STT")
        
        # Cleanup
        print("🧹 Cleaning up...")
        await stt.cleanup()
        print("✅ Cleanup complete")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run the async STT connection test."""
    await test_async_stt_connection()

if __name__ == "__main__":
    asyncio.run(main()) 