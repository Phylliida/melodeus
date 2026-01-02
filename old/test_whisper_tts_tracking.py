#!/usr/bin/env python3
"""
Test script for Whisper TTS Tracking functionality.
Tests the accuracy and reliability of Whisper-based spoken content detection.
"""

import asyncio
import time
from config_loader import load_config
from async_tts_module import AsyncTTSStreamer, TTSConfig

async def test_whisper_tracking():
    """Test Whisper tracking with various scenarios."""
    print("🧪 Testing Whisper TTS Tracking")
    print("=" * 50)
    
    # Load config
    config = load_config()
    
    # Setup TTS
    tts_config = TTSConfig(
        api_key=config.conversation.elevenlabs_api_key,
        voice_id=config.conversation.voice_id,
        model_id=config.conversation.tts_model,
        sample_rate=config.tts.sample_rate,
        speed=config.conversation.tts_speed,
        stability=config.conversation.tts_stability,
        similarity_boost=config.conversation.tts_similarity_boost
    )
    
    tts = AsyncTTSStreamer(tts_config)
    
    # Test scenarios
    test_cases = [
        {
            "name": "Short phrase",
            "text": "Hello, how are you today?",
            "interrupt_after": None
        },
        {
            "name": "Medium sentence", 
            "text": "This is a longer sentence that should give Whisper more content to work with for accurate transcription.",
            "interrupt_after": None
        },
        {
            "name": "Interrupted speech",
            "text": "This is a very long sentence that we will interrupt in the middle to test how well Whisper captures partial content when the TTS is stopped before completion.",
            "interrupt_after": 3.0  # Interrupt after 3 seconds
        },
        {
            "name": "Quick interruption",
            "text": "Another test where we interrupt very quickly to see minimal spoken content.",
            "interrupt_after": 1.0  # Interrupt after 1 second
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"Input: '{test_case['text']}'")
        
        start_time = time.time()
        
        try:
            if test_case['interrupt_after']:
                # Test with interruption
                print(f"⏱️ Will interrupt after {test_case['interrupt_after']}s")
                
                # Start speaking in background
                speak_task = asyncio.create_task(tts.speak_text(test_case['text']))
                
                # Wait and then interrupt
                await asyncio.sleep(test_case['interrupt_after'])
                print("🛑 Interrupting...")
                await tts.stop()
                
                # Wait for completion
                result = await speak_task
                
            else:
                # Test complete speech
                result = await tts.speak_text(test_case['text'])
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Get Whisper results
            spoken_content = tts.get_spoken_content()
            spoken_text = tts.get_spoken_text()
            spoken_heuristic = tts.get_spoken_text_heuristic()
            generated_vs_spoken = tts.get_generated_vs_spoken()
            was_fully_spoken = tts.was_fully_spoken()
            
            # Analyze results
            test_result = {
                "test_name": test_case['name'],
                "generated_text": test_case['text'],
                "spoken_text": spoken_text,
                "spoken_heuristic": spoken_heuristic,
                "duration": duration,
                "completed": result,
                "was_fully_spoken": was_fully_spoken,
                "num_segments": len(spoken_content),
                "whisper_accuracy_ratio": len(spoken_text) / len(test_case['text']) if test_case['text'] else 0,
                "heuristic_accuracy_ratio": len(spoken_heuristic) / len(test_case['text']) if test_case['text'] else 0
            }
            
            results.append(test_result)
            
            # Print results
            print(f"✅ Completed: {result}")
            print(f"⏱️ Duration: {duration:.2f}s")
            print(f"📝 Generated: {len(test_case['text'])} chars")
            print(f"🎙️ Whisper: {len(spoken_text)} chars")
            print(f"🧮 Heuristic: {len(spoken_heuristic)} chars")
            print(f"📊 Whisper accuracy: {test_result['whisper_accuracy_ratio']:.2%}")
            print(f"📊 Heuristic accuracy: {test_result['heuristic_accuracy_ratio']:.2%}")
            print(f"🔍 Fully spoken: {was_fully_spoken}")
            print(f"📜 Whisper segments: {len(spoken_content)}")
            
            if spoken_text:
                print(f"💬 Whisper text: '{spoken_text}'")
                print(f"🧮 Heuristic text: '{spoken_heuristic}'")
                
                # Compare heuristic similarity
                if spoken_heuristic.lower().strip() == test_case['text'].lower().strip():
                    print("✅ Heuristic: Perfect match!")
                elif spoken_heuristic.lower() in test_case['text'].lower():
                    print("🔄 Heuristic: Partial match (subset)")
                elif any(word in spoken_heuristic.lower() for word in test_case['text'].lower().split()[:3]):
                    print("⚠️ Heuristic: Some words match")
                else:
                    print("❌ Heuristic: No clear match")
            else:
                print("❌ No spoken text captured")
            
            # Wait between tests
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 WHISPER TRACKING TEST SUMMARY")
    print("=" * 50)
    
    for result in results:
        print(f"\n🧪 {result['test_name']}:")
        print(f"   Completed: {result['completed']}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Whisper accuracy: {result['whisper_accuracy_ratio']:.2%}")
        print(f"   Heuristic accuracy: {result['heuristic_accuracy_ratio']:.2%}")
        print(f"   Segments: {result['num_segments']}")
        print(f"   Fully spoken: {result['was_fully_spoken']}")
    
    # Overall stats
    completed_tests = sum(1 for r in results if r['completed'])
    avg_whisper_accuracy = sum(r['whisper_accuracy_ratio'] for r in results) / len(results) if results else 0
    avg_heuristic_accuracy = sum(r['heuristic_accuracy_ratio'] for r in results) / len(results) if results else 0
    
    print(f"\n🎯 Overall Results:")
    print(f"   Tests completed: {completed_tests}/{len(results)}")
    print(f"   Average Whisper accuracy: {avg_whisper_accuracy:.2%}")
    print(f"   Average Heuristic accuracy: {avg_heuristic_accuracy:.2%}")
    print(f"   Heuristic tracking: {'✅ Working' if avg_heuristic_accuracy > 0.5 else '❌ Needs work'}")
    
    # Cleanup
    await tts.cleanup()
    
    return results

async def main():
    """Main test function."""
    try:
        results = await test_whisper_tracking()
        print("\n🏁 Testing complete!")
        
    except KeyboardInterrupt:
        print("\n👋 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 