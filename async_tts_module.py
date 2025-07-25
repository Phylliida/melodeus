#!/usr/bin/env python3
"""
Async TTS Module for ElevenLabs WebSocket Streaming
Provides interruptible text-to-speech with real-time audio playback.
"""

import asyncio
import websockets
import json
import base64
import pyaudio
import threading
import queue
import time
from typing import Optional, AsyncGenerator, Dict, Any
from dataclasses import dataclass

@dataclass
class TTSConfig:
    """Configuration for TTS settings."""
    api_key: str
    voice_id: str = "T2KZm9rWPG5TgXTyjt7E"  # Catalyst voice
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "pcm_22050"
    sample_rate: int = 22050
    speed: float = 1.0
    stability: float = 0.5
    similarity_boost: float = 0.8
    chunk_size: int = 1024
    buffer_size: int = 2048

class AsyncTTSStreamer:
    """Async TTS Streamer with interruption capabilities."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.is_streaming = False
        self.playback_thread = None
        self.websocket = None
        self.audio_task = None
        
        # Audio setup
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Control flags
        self._stop_requested = False
        self._interrupted = False
        
    async def speak_text(self, text: str) -> bool:
        """
        Speak the given text with interruption support.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
        if self.is_streaming:
            await self.stop()
            
        try:
            self._stop_requested = False
            self._interrupted = False
            
            await self._start_streaming(text)
            return not self._interrupted
            
        except Exception as e:
            print(f"TTS error: {e}")
            return False
        finally:
            await self._cleanup_stream()
    
    async def speak_stream(self, text_generator: AsyncGenerator[str, None]) -> bool:
        """
        Speak streaming text with interruption support.
        
        Args:
            text_generator: Async generator yielding text chunks
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
        if self.is_streaming:
            await self.stop()
            
        try:
            self._stop_requested = False
            self._interrupted = False
            
            await self._start_streaming_generator(text_generator)
            return not self._interrupted
            
        except Exception as e:
            print(f"TTS streaming error: {e}")
            return False
        finally:
            await self._cleanup_stream()
    
    async def stop(self):
        """Stop current TTS playback immediately and wait for complete shutdown."""
        print("🛑 TTS stop requested")
        self._stop_requested = True
        self._interrupted = True
        
        # Clear audio queue immediately
        self._clear_audio_queue()
        
        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                print(f"Error closing TTS WebSocket: {e}")
            self.websocket = None
        
        # Cancel audio task and wait for it to finish
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
        
        # Stop audio playback and wait for thread to exit
        await self._stop_audio_playback_async()
        
        self.is_streaming = False
        print("✅ TTS stopped")
    
    def is_currently_playing(self) -> bool:
        """Check if TTS is currently playing (both streaming and audio output)."""
        return (self.is_streaming and 
                self.is_playing and 
                self.playback_thread and 
                self.playback_thread.is_alive() and
                not self._stop_requested)
    
    async def _start_streaming(self, text: str):
        """Start streaming a single text."""
        await self._setup_websocket()
        
        # Send text
        if not self._stop_requested:
            message = {
                "text": text,
                "try_trigger_generation": True
            }
            await self.websocket.send(json.dumps(message))
            print(f"🔊 TTS: {text}")
        
        # Signal completion
        if not self._stop_requested:
            await self.websocket.send(json.dumps({"text": ""}))
        
        # Wait for completion
        await self._wait_for_completion()
    
    async def _start_streaming_generator(self, text_generator: AsyncGenerator[str, None]):
        """Start streaming from a text generator."""
        await self._setup_websocket()
        
        text_buffer = ""
        
        async for text_chunk in text_generator:
            if self._stop_requested:
                break
                
            text_buffer += text_chunk
            
            # Send chunks at natural breaks
            if any(punct in text_buffer for punct in ['.', '!', '?', ',', ';']) or len(text_buffer) > 40:
                if not self._stop_requested:
                    message = {
                        "text": text_buffer,
                        "try_trigger_generation": True
                    }
                    await self.websocket.send(json.dumps(message))
                    print(f"🔊 TTS: {text_buffer.strip()}")
                    text_buffer = ""
        
        # Send remaining text
        if text_buffer.strip() and not self._stop_requested:
            message = {
                "text": text_buffer,
                "try_trigger_generation": True
            }
            await self.websocket.send(json.dumps(message))
            print(f"🔊 TTS Final: {text_buffer.strip()}")
        
        # Signal completion
        if not self._stop_requested:
            await self.websocket.send(json.dumps({"text": ""}))
        
        # Wait for completion
        await self._wait_for_completion()
    
    async def _setup_websocket(self):
        """Setup WebSocket connection and audio playback."""
        # Clear any leftover audio
        self._clear_audio_queue()
        
        # Start audio playback
        self._start_audio_playback()
        
        # Connect to WebSocket
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.config.voice_id}/stream-input"
        params = f"?model_id={self.config.model_id}&output_format={self.config.output_format}"
        
        self.websocket = await websockets.connect(uri + params)
        self.is_streaming = True
        
        # Send initial configuration
        initial_message = {
            "text": " ",
            "voice_settings": {
                "speed": self.config.speed,
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost
            },
            "xi_api_key": self.config.api_key
        }
        await self.websocket.send(json.dumps(initial_message))
        
        # Start audio response handler
        self.audio_task = asyncio.create_task(self._handle_audio_responses())
    
    async def _handle_audio_responses(self):
        """Handle incoming audio responses from ElevenLabs."""
        try:
            async for message in self.websocket:
                if self._stop_requested:
                    break
                
                if message is None:
                    continue
                
                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    continue
                
                if "audio" in data and data["audio"] and not self._stop_requested:
                    try:
                        audio_data = base64.b64decode(data["audio"])
                        if len(audio_data) > 0:
                            self.audio_queue.put(audio_data)
                    except Exception as e:
                        print(f"Failed to decode TTS audio: {e}")
                        continue
                
                if data.get("isFinal", False):
                    print("🏁 TTS generation completed")
                    break
                    
        except asyncio.CancelledError:
            print("🔇 TTS audio handler cancelled")
        except Exception as e:
            if not self._stop_requested:
                print(f"TTS audio handling error: {e}")
    
    async def _wait_for_completion(self):
        """Wait for audio processing to complete."""
        if self.audio_task and not self._stop_requested:
            try:
                await asyncio.wait_for(self.audio_task, timeout=15.0)
            except asyncio.TimeoutError:
                print("⏰ TTS audio task timed out")
                if self.audio_task:
                    self.audio_task.cancel()
            except asyncio.CancelledError:
                pass
        
        # Wait for queue to drain
        if not self._stop_requested:
            while not self.audio_queue.empty() and not self._stop_requested:
                await asyncio.sleep(0.1)
    
    def _start_audio_playback(self):
        """Start the audio playback thread."""
        if not self.is_playing:
            self.is_playing = True
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                output=True,
                frames_per_buffer=self.config.buffer_size
            )
            
            self.playback_thread = threading.Thread(target=self._audio_playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()
    
    def _stop_audio_playback(self):
        """Stop the audio playback thread (sync version)."""
        if self.is_playing:
            self.is_playing = False
            
            if self.playback_thread:
                self.playback_thread.join(timeout=2.0)
                
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"Error stopping audio stream: {e}")
                self.stream = None
    
    async def _stop_audio_playback_async(self):
        """Stop the audio playback thread and wait for complete shutdown."""
        if not self.is_playing:
            return  # Already stopped
        
        print("🔄 Stopping audio playback...")
        self.is_playing = False
        
        # Wait for playback thread to exit in a non-blocking way
        if self.playback_thread and self.playback_thread.is_alive():
            # Use asyncio to wait for thread without blocking
            max_wait_time = 1.0  # Maximum 1 second wait
            start_time = asyncio.get_event_loop().time()
            
            while (self.playback_thread.is_alive() and 
                   (asyncio.get_event_loop().time() - start_time) < max_wait_time):
                await asyncio.sleep(0.01)  # Check every 10ms
            
            if self.playback_thread.is_alive():
                print("⚠️ Audio thread didn't exit cleanly, forcing...")
                # Thread should exit on its own when is_playing = False
        
        # Clean up audio stream
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                print("🔇 Audio stream closed")
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
            self.stream = None
        
        self.playback_thread = None
        print("✅ Audio playback fully stopped")
    
    def _audio_playback_worker(self):
        """Worker thread for audio playback with responsive stop handling."""
        print("🎵 Audio playback worker started")
        
        while self.is_playing and not self._stop_requested:
            try:
                # Use shorter timeout for more responsive stopping
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Double-check stop status before writing
                if audio_chunk and self.stream and not self._stop_requested and self.is_playing:
                    try:
                        self.stream.write(audio_chunk)
                        self.audio_queue.task_done()
                    except Exception as e:
                        print(f"Audio write error: {e}")
                        self.audio_queue.task_done()
                        break
                elif audio_chunk:
                    # If stopped, mark task as done but don't play
                    self.audio_queue.task_done()
                    
            except queue.Empty:
                # Check stop condition more frequently during silence
                if self._stop_requested or not self.is_playing:
                    break
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
                break
        
        print("🔇 Audio playback worker exited")
    
    def _clear_audio_queue(self):
        """Clear the audio queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
    
    async def _cleanup_stream(self):
        """Clean up streaming resources."""
        self.is_streaming = False
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
        
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
            self.audio_task = None
        
        self._stop_audio_playback()
    
    async def cleanup(self):
        """Clean up all resources."""
        await self.stop()
        
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'p') and self.p:
            try:
                self.p.terminate()
            except Exception:
                pass

# Convenience function for quick usage
async def speak_text(text: str, api_key: str, voice_id: str = "T2KZm9rWPG5TgXTyjt7E") -> bool:
    """
    Convenience function to speak text with default settings.
    
    Args:
        text: Text to speak
        api_key: ElevenLabs API key
        voice_id: Voice ID to use
        
    Returns:
        bool: True if completed successfully, False if interrupted
    """
    config = TTSConfig(api_key=api_key, voice_id=voice_id)
    tts = AsyncTTSStreamer(config)
    
    try:
        result = await tts.speak_text(text)
        return result
    finally:
        await tts.cleanup()

# Example usage
async def main():
    """Example usage of the TTS module."""
    api_key = input("Enter your ElevenLabs API key: ").strip()
    
    config = TTSConfig(api_key=api_key)
    tts = AsyncTTSStreamer(config)
    
    try:
        print("🎤 Starting TTS test...")
        
        # Test single text
        result = await tts.speak_text("Hello! This is a test of the async TTS module.")
        print(f"✅ Completed: {result}")
        
        # Test interruption
        print("\n🛑 Testing interruption...")
        task = asyncio.create_task(
            tts.speak_text("This is a longer message that we will interrupt before it finishes playing completely.")
        )
        
        # Interrupt after 2 seconds
        await asyncio.sleep(2.0)
        await tts.stop()
        
        result = await task
        print(f"✅ Interrupted result: {result}")
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        await tts.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 