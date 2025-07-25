#!/usr/bin/env python3
"""
Debug script to diagnose Deepgram connection issues.
Tests different configurations to find what's causing immediate connection closure.
"""

import asyncio
import pyaudio
import struct
import time
from config_loader import load_config
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

class DeepgramConnectionDebugger:
    """Debug Deepgram connection issues."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.deepgram = DeepgramClient(api_key)
        self.p = pyaudio.PyAudio()
        self.connection_events = []
        
    def log_event(self, event_type: str, details: str = ""):
        timestamp = time.time()
        event = f"[{timestamp:.2f}] {event_type}: {details}"
        print(event)
        self.connection_events.append(event)
    
    async def test_configuration(self, config_name: str, options: LiveOptions) -> dict:
        """Test a specific Deepgram configuration."""
        print(f"\n🧪 Testing Configuration: {config_name}")
        print("=" * 50)
        
        self.connection_events.clear()
        connection = None
        microphone = None
        connection_alive = False
        start_time = time.time()
        
        try:
            # Setup microphone
            self.log_event("SETUP", "Creating microphone stream")
            microphone = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000
            )
            
            # Create connection
            self.log_event("SETUP", "Creating Deepgram connection")
            connection = self.deepgram.listen.websocket.v("1")
            
            # Set up event handlers
            def on_open(*args, **kwargs):
                nonlocal connection_alive
                connection_alive = True
                elapsed = time.time() - start_time
                self.log_event("OPEN", f"Connection opened after {elapsed:.2f}s")
            
            def on_transcript(*args, **kwargs):
                result = kwargs.get('result')
                if result and hasattr(result, 'channel'):
                    try:
                        transcript = result.channel.alternatives[0].transcript.strip()
                        if transcript:
                            self.log_event("TRANSCRIPT", f"'{transcript}' (final: {result.is_final})")
                    except:
                        self.log_event("TRANSCRIPT", "Received transcript (parsing failed)")
            
            def on_error(*args, **kwargs):
                error = kwargs.get('error', 'Unknown error')
                self.log_event("ERROR", str(error))
            
            def on_close(*args, **kwargs):
                nonlocal connection_alive
                connection_alive = False
                elapsed = time.time() - start_time
                self.log_event("CLOSE", f"Connection closed after {elapsed:.2f}s")
            
            def on_utterance_end(*args, **kwargs):
                self.log_event("UTTERANCE_END", "Utterance ended")
            
            def on_speech_started(*args, **kwargs):
                self.log_event("SPEECH_START", "Speech detected")
            
            # Register handlers
            connection.on(LiveTranscriptionEvents.Open, on_open)
            connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
            connection.on(LiveTranscriptionEvents.Error, on_error)
            connection.on(LiveTranscriptionEvents.Close, on_close)
            connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
            connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
            
            # Start connection
            self.log_event("START", f"Starting connection with options: {options}")
            if connection.start(options):
                self.log_event("START", "Connection.start() returned True")
                
                # Stream audio for 15 seconds or until connection dies
                chunks_sent = 0
                silent_chunks = 0
                
                for i in range(150):  # 15 seconds with 0.1s intervals
                    if not connection_alive:
                        self.log_event("LOOP", f"Connection died, stopping after {i/10:.1f}s")
                        break
                    
                    try:
                        # Read audio data
                        data = microphone.read(1600, exception_on_overflow=False)  # 0.1s worth
                        chunks_sent += 1
                        
                        # Check if audio is mostly silent
                        audio_samples = struct.unpack(f'<{len(data)//2}h', data)
                        max_amplitude = max(abs(sample) for sample in audio_samples)
                        
                        if max_amplitude < 500:  # Mostly silent
                            silent_chunks += 1
                        
                        # Send to Deepgram
                        connection.send(data)
                        
                        if i % 50 == 0:  # Every 5 seconds
                            self.log_event("AUDIO", f"Sent {chunks_sent} chunks, {silent_chunks} silent, max_amp: {max_amplitude}")
                        
                    except Exception as e:
                        self.log_event("AUDIO_ERROR", str(e))
                        break
                    
                    await asyncio.sleep(0.1)
                
                # Final stats
                final_time = time.time() - start_time
                self.log_event("FINAL", f"Ran for {final_time:.1f}s, sent {chunks_sent} chunks ({silent_chunks} silent)")
                
            else:
                self.log_event("START", "Connection.start() returned False")
                
        except Exception as e:
            self.log_event("EXCEPTION", str(e))
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if connection:
                try:
                    connection.finish()
                except:
                    pass
            if microphone:
                try:
                    microphone.stop_stream()
                    microphone.close()
                except:
                    pass
        
        # Return results
        total_time = time.time() - start_time
        return {
            "config_name": config_name,
            "success": connection_alive,
            "duration": total_time,
            "events": self.connection_events.copy()
        }
    
    async def run_all_tests(self):
        """Run multiple configuration tests."""
        print("🔬 Deepgram Connection Diagnostics")
        print("==================================")
        
        # Test configurations from simple to complex
        test_configs = [
            ("Basic STT", LiveOptions(
                model="nova-3",
                language="en-US",
                encoding="linear16",
                sample_rate=16000,
                channels=1
            )),
            
            ("With Smart Format", LiveOptions(
                model="nova-3", 
                language="en-US",
                encoding="linear16",
                sample_rate=16000,
                channels=1,
                smart_format=True
            )),
            
            ("With Interim Results", LiveOptions(
                model="nova-3",
                language="en-US", 
                encoding="linear16",
                sample_rate=16000,
                channels=1,
                smart_format=True,
                interim_results=True
            )),
            
            ("With Punctuation", LiveOptions(
                model="nova-3",
                language="en-US",
                encoding="linear16", 
                sample_rate=16000,
                channels=1,
                smart_format=True,
                interim_results=True,
                punctuate=True
            )),
            
            ("With VAD Events", LiveOptions(
                model="nova-3",
                language="en-US",
                encoding="linear16",
                sample_rate=16000, 
                channels=1,
                smart_format=True,
                interim_results=True,
                punctuate=True,
                vad_events=True
            )),
            
            ("With Diarization", LiveOptions(
                model="nova-3",
                language="en-US",
                encoding="linear16",
                sample_rate=16000,
                channels=1,
                smart_format=True,
                interim_results=True,
                punctuate=True,
                vad_events=True,
                diarize=True
            )),
            
            ("Full Config (Current)", LiveOptions(
                model="nova-3",
                language="en-US",
                encoding="linear16",
                sample_rate=16000,
                channels=1,
                smart_format=True,
                interim_results=True,
                punctuate=True,
                diarize=True,
                utterance_end_ms=1000,
                vad_events=True
            )),
        ]
        
        results = []
        
        for config_name, options in test_configs:
            result = await self.test_configuration(config_name, options)
            results.append(result)
            
            # Short break between tests
            await asyncio.sleep(2)
        
        # Summary
        print("\n📊 Test Results Summary")
        print("=" * 50)
        
        for result in results:
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            duration = result["duration"]
            print(f"{status} {result['config_name']:<20} Duration: {duration:.1f}s")
        
        # Find the most complex working config
        working_configs = [r for r in results if r["success"]]
        if working_configs:
            best = max(working_configs, key=lambda x: len(x["config_name"]))
            print(f"\n🎯 Best working config: {best['config_name']}")
        else:
            print("\n❌ No configurations worked! Check API key and network.")
        
        return results

    def cleanup(self):
        """Clean up PyAudio."""
        if self.p:
            self.p.terminate()

async def main():
    """Run Deepgram connection diagnostics."""
    try:
        # Load API key
        config = load_config()
        api_key = config.stt.api_key
        print(f"✅ Loaded API key: {api_key[:8]}...")
        
    except Exception as e:
        print(f"❌ Failed to load API key: {e}")
        return
    
    debugger = DeepgramConnectionDebugger(api_key)
    
    try:
        results = await debugger.run_all_tests()
        
        print("\n💡 Recommendations:")
        
        # Check for patterns
        working = [r for r in results if r["success"]]
        failing = [r for r in results if not r["success"]]
        
        if not working:
            print("   - Check your Deepgram API key tier and permissions")
            print("   - Verify your internet connection") 
            print("   - Try a different model (e.g., 'base' instead of 'nova-3')")
        elif len(working) < len(results):
            failed_features = []
            if any("Diarization" in r["config_name"] for r in failing):
                failed_features.append("diarization")
            if any("VAD" in r["config_name"] for r in failing):
                failed_features.append("VAD events")
            
            if failed_features:
                print(f"   - Disable these features: {', '.join(failed_features)}")
            print(f"   - Use the '{working[-1]['config_name']}' configuration")
        else:
            print("   - All configurations work! The issue might be elsewhere.")
            print("   - Check audio device permissions and microphone access")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        debugger.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 