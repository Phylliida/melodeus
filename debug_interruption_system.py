#!/usr/bin/env python3
"""
Debug script to analyze interruption system timing and async state issues.
"""

import asyncio
import time
from datetime import datetime
from config_loader import load_config
from async_stt_module import AsyncSTTStreamer, STTEventType, STTResult
from async_tts_module import AsyncTTSStreamer

class InterruptionDebugger:
    """Debug interruption timing and TTS state management."""
    
    def __init__(self, config):
        self.config = config
        self.stt = AsyncSTTStreamer(config.stt)
        self.tts = AsyncTTSStreamer(config.tts)
        
        # State tracking
        self.conversation_thinks_speaking = False
        self.last_tts_start = None
        self.last_tts_stop = None
        self.events = []
        
        # Interruption tracking
        self.interruption_attempts = 0
        self.successful_interruptions = 0
        self.failed_interruptions = 0
        
    def log_event(self, event_type: str, details: str = "", include_states=True):
        """Log an event with timestamps and state info."""
        timestamp = time.time()
        event = {
            "time": timestamp,
            "datetime": datetime.now().isoformat(),
            "type": event_type,
            "details": details
        }
        
        if include_states:
            event.update({
                "conversation_speaking": self.conversation_thinks_speaking,
                "tts_is_playing": getattr(self.tts, 'is_playing', False),
                "tts_websocket_open": getattr(self.tts, 'websocket', None) is not None,
                "audio_queue_size": getattr(self.tts, 'audio_queue', {}).qsize() if hasattr(getattr(self.tts, 'audio_queue', None), 'qsize') else 0
            })
        
        self.events.append(event)
        
        # Print with color coding
        color = {
            "TTS_START": "\033[92m",  # Green
            "TTS_STOP": "\033[91m",   # Red
            "SPEECH_DETECTED": "\033[93m",  # Yellow
            "INTERRUPTION_ATTEMPT": "\033[94m",  # Blue
            "INTERRUPTION_SUCCESS": "\033[95m",  # Magenta
            "STATE_MISMATCH": "\033[91m",  # Red
            "INFO": "\033[96m",  # Cyan
        }.get(event_type, "\033[0m")  # Default
        
        print(f"{color}[{timestamp:.3f}] {event_type}: {details}\033[0m")
        
        if include_states:
            print(f"  States: Conv={self.conversation_thinks_speaking}, TTS={getattr(self.tts, 'is_playing', 'Unknown')}")
    
    async def start_debugging(self):
        """Start the debugging session."""
        print("🔍 Interruption System Debugging")
        print("=" * 50)
        
        # Setup STT callbacks with debugging
        self.stt.on(STTEventType.INTERIM_RESULT, self._debug_interim_result)
        self.stt.on(STTEventType.UTTERANCE_COMPLETE, self._debug_utterance_complete)
        self.stt.on(STTEventType.SPEECH_STARTED, self._debug_speech_started)
        self.stt.on(STTEventType.SPEECH_ENDED, self._debug_speech_ended)
        
        # Start STT
        self.log_event("STT_START", "Starting speech recognition")
        if not await self.stt.start_listening():
            print("❌ Failed to start STT")
            return
        
        print("✅ Debugging active! Try the following:")
        print("1. Say something to trigger TTS")
        print("2. Try to interrupt while AI is speaking")
        print("3. Watch for state mismatches")
        print("4. Press Ctrl+C when done")
        print()
        
        # Run test scenarios
        await self._run_debug_scenarios()
    
    async def _run_debug_scenarios(self):
        """Run specific debug scenarios."""
        try:
            # Wait for user interaction
            while True:
                await asyncio.sleep(0.1)
                
                # Check for state mismatches
                await self._check_state_consistency()
                
        except KeyboardInterrupt:
            print("\n🛑 Debugging stopped by user")
    
    async def _check_state_consistency(self):
        """Check if TTS and conversation states are consistent."""
        tts_playing = getattr(self.tts, 'is_playing', False)
        
        if self.conversation_thinks_speaking != tts_playing:
            self.log_event("STATE_MISMATCH", 
                f"Conversation thinks speaking: {self.conversation_thinks_speaking}, "
                f"TTS actually playing: {tts_playing}")
    
    async def _debug_interim_result(self, result: STTResult):
        """Debug interim results and interruption logic."""
        self.log_event("SPEECH_DETECTED", 
            f"'{result.text}' (conf: {result.confidence:.2f}, words: {len(result.text.split())})")
        
        # Check interruption criteria
        if self.conversation_thinks_speaking:
            self.log_event("INTERRUPTION_ATTEMPT", 
                f"Checking: conf={result.confidence:.2f} > 0.8? words={len(result.text.split())} >= 2?")
            
            self.interruption_attempts += 1
            
            if (result.confidence > 0.8 and len(result.text.split()) >= 2):
                self.log_event("INTERRUPTION_SUCCESS", f"Attempting to stop TTS with: '{result.text}'")
                
                # Try to stop TTS
                old_speaking_state = self.conversation_thinks_speaking
                self.conversation_thinks_speaking = False
                
                try:
                    await self.tts.stop()
                    self.successful_interruptions += 1
                    self.log_event("TTS_STOP", "TTS stop called successfully")
                    
                    # Wait a bit and check if it actually stopped
                    await asyncio.sleep(0.1)
                    actual_tts_state = getattr(self.tts, 'is_playing', False)
                    
                    if actual_tts_state:
                        self.log_event("STATE_MISMATCH", "TTS still playing after stop() call")
                    else:
                        self.log_event("INFO", "TTS successfully stopped")
                        
                except Exception as e:
                    self.log_event("INTERRUPTION_FAILURE", f"Failed to stop TTS: {e}")
                    self.failed_interruptions += 1
                    self.conversation_thinks_speaking = old_speaking_state
            else:
                self.log_event("INFO", "Interruption criteria not met")
    
    async def _debug_utterance_complete(self, result: STTResult):
        """Debug completed utterances."""
        self.log_event("UTTERANCE_COMPLETE", f"'{result.text}'")
        
        # Simulate LLM response
        await self._debug_tts_response(f"I heard you say: {result.text}")
    
    async def _debug_speech_started(self, data):
        """Debug speech start events."""
        self.log_event("SPEECH_START", "User speech detected")
    
    async def _debug_speech_ended(self, data):
        """Debug speech end events."""
        self.log_event("SPEECH_END", "User speech ended")
    
    async def _debug_tts_response(self, text: str):
        """Debug TTS response with timing tracking."""
        if self.conversation_thinks_speaking:
            self.log_event("INFO", "Skipping TTS - already speaking")
            return
        
        self.log_event("TTS_START", f"Starting TTS: '{text[:50]}...'")
        self.conversation_thinks_speaking = True
        self.last_tts_start = time.time()
        
        try:
            # Use TTS to speak
            result = await self.tts.speak_text(text)
            
            if result:
                self.log_event("TTS_COMPLETE", "TTS completed successfully")
            else:
                self.log_event("TTS_INTERRUPTED", "TTS was interrupted")
                
        except Exception as e:
            self.log_event("TTS_ERROR", f"TTS error: {e}")
        finally:
            self.conversation_thinks_speaking = False
            self.last_tts_stop = time.time()
            
            if self.last_tts_start:
                duration = self.last_tts_stop - self.last_tts_start
                self.log_event("TTS_DURATION", f"TTS took {duration:.2f} seconds")
    
    def print_summary(self):
        """Print debugging summary."""
        print("\n" + "="*50)
        print("🔍 DEBUGGING SUMMARY")
        print("="*50)
        
        print(f"📊 Interruption Stats:")
        print(f"   Total Attempts: {self.interruption_attempts}")
        print(f"   Successful: {self.successful_interruptions}")
        print(f"   Failed: {self.failed_interruptions}")
        
        if self.interruption_attempts > 0:
            success_rate = (self.successful_interruptions / self.interruption_attempts) * 100
            print(f"   Success Rate: {success_rate:.1f}%")
        
        # Analyze timing patterns
        print(f"\n🕐 Timing Analysis:")
        speech_events = [e for e in self.events if e['type'] in ['SPEECH_DETECTED', 'TTS_START', 'TTS_STOP']]
        
        if len(speech_events) > 1:
            print(f"   Total Events: {len(speech_events)}")
            
            # Find state mismatches
            mismatches = [e for e in self.events if e['type'] == 'STATE_MISMATCH']
            print(f"   State Mismatches: {len(mismatches)}")
            
            for mismatch in mismatches[-3:]:  # Show last 3
                print(f"     {mismatch['datetime']}: {mismatch['details']}")
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.stt.cleanup()
            await self.tts.cleanup()
        except Exception as e:
            print(f"Cleanup error: {e}")

async def main():
    """Run interruption debugging."""
    try:
        # Load config
        config = load_config()
        print("✅ Configuration loaded")
        
        debugger = InterruptionDebugger(config)
        
        try:
            await debugger.start_debugging()
        except Exception as e:
            print(f"❌ Debugging error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            debugger.print_summary()
            await debugger.cleanup()
            
    except Exception as e:
        print(f"❌ Setup error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 