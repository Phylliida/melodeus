#!/usr/bin/env python3
"""
Async STT Module for Deepgram Real-time Speech-to-Text
Provides callback-based speech recognition with speaker identification.
"""

import asyncio
import pyaudio
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
from enum import Enum

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

class STTEventType(Enum):
    """Types of STT events."""
    UTTERANCE_COMPLETE = "utterance_complete"
    INTERIM_RESULT = "interim_result"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    SPEAKER_CHANGE = "speaker_change"
    CONNECTION_OPENED = "connection_opened"
    CONNECTION_CLOSED = "connection_closed"
    ERROR = "error"

@dataclass
class STTResult:
    """Result from speech-to-text recognition."""
    text: str
    confidence: float
    is_final: bool
    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Optional[Dict[str, Any]] = None

@dataclass
class STTConfig:
    """Configuration for STT settings."""
    api_key: str
    model: str = "nova-3"
    language: str = "en-US"
    sample_rate: int = 16000
    chunk_size: int = 8000
    channels: int = 1
    smart_format: bool = True
    interim_results: bool = True
    punctuate: bool = True
    diarize: bool = True
    utterance_end_ms: int = 1000
    vad_events: bool = True
    # Speaker identification settings
    enable_speaker_id: bool = False
    speaker_profiles_path: Optional[str] = None

class AsyncSTTStreamer:
    """Async STT Streamer with callback support."""
    
    def __init__(self, config: STTConfig):
        self.config = config
        self.is_listening = False
        self.connection = None
        self.microphone = None
        self.event_loop = None
        self.audio_task = None
        self.connection_alive = False
        
        # Audio setup
        self.p = pyaudio.PyAudio()
        
        # Deepgram client
        self.deepgram = DeepgramClient(config.api_key)
        
        # Callback registry
        self.callbacks: Dict[STTEventType, List[Callable]] = {
            event_type: [] for event_type in STTEventType
        }
        
        # State tracking
        self.current_speaker = None
        self.last_utterance_time = None
        self.session_speakers = {}  # Maps session speaker IDs to names
        
        # Speaker identification (optional)
        self.speaker_identifier = None
        if config.enable_speaker_id:
            self._setup_speaker_identification()
    
    def on(self, event_type: STTEventType, callback: Callable):
        """Register a callback for an event type."""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def off(self, event_type: STTEventType, callback: Callable):
        """Unregister a callback for an event type."""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    async def start_listening(self) -> bool:
        """Start listening for speech."""
        if self.is_listening:
            await self.stop_listening()
        
        # Store the current event loop for scheduling tasks from sync callbacks
        self.event_loop = asyncio.get_event_loop()
        
        try:
            # Setup microphone in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self.microphone = await loop.run_in_executor(
                None,
                lambda: self.p.open(
                    format=pyaudio.paInt16,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    frames_per_buffer=self.config.chunk_size
                )
            )
            
            # Setup Deepgram connection
            self.connection = self.deepgram.listen.websocket.v("1")
            
            # Configure options with explicit audio format
            options = LiveOptions(
                model=self.config.model,
                language=self.config.language,
                encoding="linear16",  # Critical: Explicit audio encoding
                sample_rate=self.config.sample_rate,  # Explicit sample rate
                channels=self.config.channels,  # Explicit channel count
                smart_format=self.config.smart_format,
                interim_results=self.config.interim_results,
                punctuate=self.config.punctuate,
                diarize=self.config.diarize,
                utterance_end_ms=self.config.utterance_end_ms,
                vad_events=self.config.vad_events
            )
            
            # Set up event handlers
            self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self.connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
            self.connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
            self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self._on_close)
            
            # Start connection with proper async handling
            connection_success = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.connection.start(options)
            )
            
            if connection_success:
                self.is_listening = True
                # Don't set connection_alive yet - wait for _on_open callback
                
                # Minimal wait for connection to stabilize
                await asyncio.sleep(0.01)
                
                # Verify connection is actually alive before starting audio
                if self.connection_alive:
                    # Start audio streaming task and track it
                    self.audio_task = asyncio.create_task(self._audio_streaming_loop())
                    
                    await self._emit_event(STTEventType.CONNECTION_OPENED, {
                        "message": "STT connection established"
                    })
                    
                    return True
                else:
                    await self._emit_event(STTEventType.ERROR, {
                        "error": "Connection opened but not confirmed alive"
                    })
                    return False
            else:
                await self._emit_event(STTEventType.ERROR, {
                    "error": "Failed to start Deepgram connection"
                })
                return False
                
        except Exception as e:
            await self._emit_event(STTEventType.ERROR, {
                "error": f"Failed to start listening: {e}"
            })
            return False
    
    async def stop_listening(self):
        """Stop listening for speech."""
        print("🛑 STT stop requested")
        self.is_listening = False
        self.connection_alive = False
        
        # Cancel audio streaming task if it's running
        if self.audio_task and not self.audio_task.done():
            print("🔄 Cancelling audio streaming task...")
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                print("✅ Audio streaming task cancelled")
            except Exception as e:
                print(f"Error cancelling audio task: {e}")
            self.audio_task = None
        
        # Close microphone
        if self.microphone:
            try:
                # Use executor to avoid blocking on microphone operations
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: (
                        self.microphone.stop_stream(),
                        self.microphone.close()
                    )
                )
            except Exception as e:
                print(f"Error closing microphone: {e}")
            self.microphone = None
        
        # Close Deepgram connection
        if self.connection:
            try:
                # Use executor to avoid blocking on connection close
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.connection.finish)
            except Exception as e:
                print(f"Error closing Deepgram connection: {e}")
            self.connection = None
        
        await self._emit_event(STTEventType.CONNECTION_CLOSED, {
            "message": "STT connection closed"
        })
        
        print("✅ STT stopped")
    
    def is_currently_listening(self) -> bool:
        """Check if STT is currently listening."""
        return self.is_listening
    
    def get_session_speakers(self) -> Dict[int, str]:
        """Get the mapping of session speaker IDs to names."""
        return self.session_speakers.copy()
    
    def register_speaker(self, session_speaker_id: int, name: str):
        """Register a name for a session speaker ID."""
        self.session_speakers[session_speaker_id] = name
        print(f"📝 Registered speaker {session_speaker_id} as '{name}'")
    
    async def _audio_streaming_loop(self):
        """Main audio streaming loop with proper async handling."""
        print("🎵 Starting audio streaming loop...")
        chunk_count = 0
        
        # Minimal wait for connection to be fully established
        await asyncio.sleep(0.001)
        
        try:
            while self.is_listening and self.microphone and self.connection and self.connection_alive:
                try:
                    # Check if we should still be running
                    if not self.connection_alive:
                        print("🔌 Connection no longer alive, stopping audio stream")
                        break
                    
                    # Read audio data in executor to avoid blocking
                    try:
                        loop = asyncio.get_event_loop()
                        data = await loop.run_in_executor(
                            None, 
                            lambda: self.microphone.read(
                                self.config.chunk_size, 
                                exception_on_overflow=False
                            )
                        )
                    except Exception as read_error:
                        print(f"❌ Failed to read audio data: {read_error}")
                        break
                    
                    # Try to send data to Deepgram
                    try:
                        self.connection.send(data)
                        chunk_count += 1
                        
                        # Less frequent debug output
                        if chunk_count % 100 == 0:
                            print(f"📊 Sent {chunk_count} audio chunks")
                            
                    except Exception as send_error:
                        print(f"❌ Failed to send audio data to Deepgram: {send_error}")
                        print(f"📊 Sent {chunk_count} chunks before connection failed")
                        # Connection is likely closed, mark as not alive
                        self.connection_alive = False
                        await self._emit_event(STTEventType.ERROR, {
                            "error": f"Connection send failed: {send_error}"
                        })
                        break
                    
                    # Brief async sleep to yield control
                    await asyncio.sleep(0.001)  # 1ms delay
                    
                except Exception as e:
                    print(f"❌ Audio streaming error: {e}")
                    await self._emit_event(STTEventType.ERROR, {
                        "error": f"Audio streaming error: {e}"
                    })
                    break
                    
        except asyncio.CancelledError:
            print("🔄 Audio streaming loop cancelled")
            raise
        except Exception as e:
            print(f"❌ Audio streaming loop fatal error: {e}")
            await self._emit_event(STTEventType.ERROR, {
                "error": f"Audio streaming loop error: {e}"
            })
        finally:
            print(f"🏁 Audio streaming loop ended. Sent {chunk_count} chunks total.")
    
    def _on_open(self, *args, **kwargs):
        """Handle Deepgram connection open."""
        print("🔗 Deepgram STT connection opened")
        self.connection_alive = True
    
    def _on_transcript(self, *args, **kwargs):
        """Handle incoming transcripts."""
        result = kwargs.get('result')
        if not result or not hasattr(result, 'channel'):
            return
        
        try:
            channel = result.channel
            alternative = channel.alternatives[0]
            transcript = alternative.transcript.strip()
            confidence = alternative.confidence
            is_final = result.is_final
            
            if not transcript:
                return
            
            # Extract speaker information
            speaker_id = None
            speaker_name = None
            
            if hasattr(channel, 'metadata') and channel.metadata:
                speaker_id = getattr(channel.metadata, 'speaker', None)
                if speaker_id is not None:
                    speaker_name = self.session_speakers.get(speaker_id)
            
            # Create STT result
            stt_result = STTResult(
                text=transcript,
                confidence=confidence,
                is_final=is_final,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                timestamp=datetime.now(),
                raw_data=kwargs
            )
            
            # Apply speaker identification if enabled
            if self.config.enable_speaker_id and self.speaker_identifier:
                stt_result = self._apply_speaker_identification(stt_result)
            
            # Emit appropriate event
            if is_final:
                self._schedule_event(
                    STTEventType.UTTERANCE_COMPLETE, 
                    stt_result
                )
                
                # Check for speaker change
                if speaker_id is not None and speaker_id != self.current_speaker:
                    self.current_speaker = speaker_id
                    self._schedule_event(
                        STTEventType.SPEAKER_CHANGE,
                        {
                            "speaker_id": speaker_id,
                            "speaker_name": speaker_name,
                            "previous_speaker": self.current_speaker
                        }
                    )
                
                self.last_utterance_time = datetime.now()
                
            else:
                self._schedule_event(
                    STTEventType.INTERIM_RESULT,
                    stt_result
                )
                
        except Exception as e:
            self._schedule_event(STTEventType.ERROR, {
                "error": f"Transcript processing error: {e}"
            })
    
    def _on_utterance_end(self, *args, **kwargs):
        """Handle utterance end events."""
        self._schedule_event(STTEventType.SPEECH_ENDED, {
            "timestamp": datetime.now()
        })
    
    def _on_speech_started(self, *args, **kwargs):
        """Handle speech start events."""
        self._schedule_event(STTEventType.SPEECH_STARTED, {
            "timestamp": datetime.now()
        })
    
    def _on_error(self, *args, **kwargs):
        """Handle STT errors."""
        error = kwargs.get('error', 'Unknown error')
        self._schedule_event(STTEventType.ERROR, {
            "error": f"Deepgram error: {error}"
        })
    
    def _on_close(self, *args, **kwargs):
        """Handle connection close."""
        print("🔌 Deepgram connection closed by server")
        self.is_listening = False
        self.connection_alive = False
        self._schedule_event(STTEventType.CONNECTION_CLOSED, {
            "message": "Deepgram connection closed by server"
        })
    
    def _setup_speaker_identification(self):
        """Setup speaker identification if enabled."""
        if self.config.speaker_profiles_path:
            try:
                # Import speaker identification module if available
                from improved_speaker_identification import SpeakerIdentifier
                self.speaker_identifier = SpeakerIdentifier(
                    profiles_file=self.config.speaker_profiles_path
                )
                print("✅ Speaker identification enabled")
            except ImportError:
                print("⚠️ Speaker identification module not found")
                self.speaker_identifier = None
    
    def _apply_speaker_identification(self, stt_result: STTResult) -> STTResult:
        """Apply speaker identification to the result."""
        if self.speaker_identifier and stt_result.raw_data:
            try:
                # Extract audio features and identify speaker
                # This would need to be implemented based on your speaker ID system
                identified_name = self.speaker_identifier.identify_speaker(
                    # Pass appropriate audio data
                )
                if identified_name:
                    stt_result.speaker_name = identified_name
            except Exception as e:
                print(f"Speaker identification error: {e}")
        
        return stt_result
    
    async def _emit_event(self, event_type: STTEventType, data: Any):
        """Emit an event to all registered callbacks."""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    print(f"Callback error for {event_type}: {e}")
    
    def _schedule_event(self, event_type: STTEventType, data: Any):
        """Schedule an event emission from a synchronous context."""
        try:
            # Try to get the current event loop
            current_loop = None
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = self.event_loop
            
            if current_loop and current_loop.is_running():
                if current_loop == self.event_loop:
                    # We're in the same loop, schedule normally
                    future = asyncio.run_coroutine_threadsafe(
                        self._emit_event(event_type, data), 
                        current_loop
                    )
                    # Don't wait for result to avoid blocking
                else:
                    # Different loop, use stored event loop
                    if self.event_loop and self.event_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._emit_event(event_type, data), 
                            self.event_loop
                        )
                    else:
                        # Fallback to sync emission
                        self._emit_event_sync(event_type, data)
            else:
                # No running loop, emit synchronously
                self._emit_event_sync(event_type, data)
                
        except Exception as e:
            print(f"Error scheduling event {event_type}: {e}")
            # Final fallback
            self._emit_event_sync(event_type, data)
    
    def _emit_event_sync(self, event_type: STTEventType, data: Any):
        """Emit an event synchronously (fallback for when no event loop is available)."""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if not asyncio.iscoroutinefunction(callback):
                        callback(data)
                    # Skip async callbacks in sync context
                except Exception as e:
                    print(f"Sync callback error for {event_type}: {e}")
    
    async def cleanup(self):
        """Clean up all resources."""
        await self.stop_listening()
        
        # Clear event loop reference
        self.event_loop = None
        
        # Clean up PyAudio in executor to avoid segfault
        if self.p:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._cleanup_pyaudio)
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
            finally:
                self.p = None
    
    def _cleanup_pyaudio(self):
        """Clean up PyAudio in a separate thread context."""
        try:
            if self.p:
                self.p.terminate()
        except Exception as e:
            print(f"PyAudio cleanup error: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        # Don't call terminate in destructor - too risky for segfaults
        pass

# Convenience functions for quick usage
async def start_listening(api_key: str, on_utterance: Callable[[STTResult], None]) -> AsyncSTTStreamer:
    """
    Convenience function to start listening with default settings.
    
    Args:
        api_key: Deepgram API key
        on_utterance: Callback for completed utterances
        
    Returns:
        AsyncSTTStreamer: The STT streamer instance
    """
    config = STTConfig(api_key=api_key)
    stt = AsyncSTTStreamer(config)
    stt.on(STTEventType.UTTERANCE_COMPLETE, on_utterance)
    
    success = await stt.start_listening()
    if success:
        return stt
    else:
        await stt.cleanup()
        raise Exception("Failed to start STT")

# Example usage
async def main():
    """Example usage of the STT module."""
    api_key = input("Enter your Deepgram API key: ").strip()
    
    config = STTConfig(        api_key=api_key,
        interim_results=True,
        diarize=True
    )
    stt = AsyncSTTStreamer(config)
    
    # Set up callbacks
    async def on_utterance_complete(result: STTResult):
        speaker_info = f" (Speaker {result.speaker_id})" if result.speaker_id is not None else ""
        print(f"🎯 Final{speaker_info}: {result.text} (confidence: {result.confidence:.2f})")
    
    def on_interim_result(result: STTResult):
        print(f"💭 Interim: {result.text}")
    
    async def on_speech_started(data):
        print("🎤 Speech started")
    
    async def on_speech_ended(data):
        print("🔇 Speech ended")
    
    async def on_error(data):
        print(f"❌ Error: {data['error']}")
    
    # Register callbacks
    stt.on(STTEventType.UTTERANCE_COMPLETE, on_utterance_complete)
    stt.on(STTEventType.INTERIM_RESULT, on_interim_result)
    stt.on(STTEventType.SPEECH_STARTED, on_speech_started)
    stt.on(STTEventType.SPEECH_ENDED, on_speech_ended)
    stt.on(STTEventType.ERROR, on_error)
    
    try:
        print("🎤 Starting STT test...")
        success = await stt.start_listening()
        
        if success:
            print("✅ STT started successfully!")
            print("💡 Speak into your microphone. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while stt.is_currently_listening():
                await asyncio.sleep(1)
        else:
            print("❌ Failed to start STT")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        await stt.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 