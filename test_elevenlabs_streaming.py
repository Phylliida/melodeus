#!/usr/bin/env python3
"""
Test script for ElevenLabs WebSocket streaming TTS with LLM integration.
Streams LLM completions to ElevenLabs and plays audio in real-time.
"""

import asyncio
import websockets
import json
import base64
import pyaudio
import threading
import queue
from openai import OpenAI
import time
import sys
from typing import Optional, AsyncGenerator

class ElevenLabsStreamer:
    def __init__(self, api_key: str, voice_id: str, openai_api_key: str):
        self.api_key = api_key
        self.voice_id = voice_id
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_thread = None
        
        # Audio settings (match ElevenLabs PCM format)
        self.sample_rate = 22050
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        
    def start_audio_playback(self):
        """Start the audio playback thread."""
        if not self.is_playing:
            self.is_playing = True
            # Use larger buffer for more stable playback
            self.stream = self.p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size * 2,  # Larger buffer
                stream_callback=None,
                output_device_index=None
            )
            print(f"🔧 Audio stream opened: {self.sample_rate}Hz, {self.channels} channel(s)")
            self.playback_thread = threading.Thread(target=self._audio_playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()
            
    def stop_audio_playback(self):
        """Stop the audio playback thread."""
        print("🛑 Stopping audio playback...")
        
        # Wait for queue to finish before stopping
        while not self.audio_queue.empty():
            time.sleep(0.1)
        
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join(timeout=3.0)  # Give more time
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        print("🔇 Audio playback stopped")
            
    def _audio_playback_worker(self):
        """Worker thread for audio playback."""
        print("🔊 Audio playback worker started")
        while self.is_playing:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.5)
                if audio_chunk and self.stream:
                    try:
                        self.stream.write(audio_chunk)
                        self.audio_queue.task_done()
                        print(f"🎵 Played audio chunk: {len(audio_chunk)} bytes")
                    except Exception as write_error:
                        print(f"Failed to write audio chunk: {write_error}")
                        self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
                break
        print("🔊 Audio playback worker stopped")
                
    async def get_llm_completion(self, prompt: str) -> AsyncGenerator[str, None]:
        """Get streaming completion from OpenAI."""
        models_to_try = ["chatgpt-4o-latest"]
        
        for model in models_to_try:
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are talking in voice mode. Avoid using complex text formatting or very long messages."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True,
                    max_tokens=500
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # Success, exit the function
                        
            except Exception as e:
                print(f"Failed with model {model}: {e}")
                if model == models_to_try[-1]:  # Last model in list
                    yield f"Sorry, I couldn't connect to any available models. Error: {e}"
                else:
                    continue  # Try next modelex
            
    async def stream_tts(self, prompt: str):
        """Stream LLM completion to ElevenLabs TTS and play audio."""
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id=eleven_multilingual_v2&output_format=pcm_22050"
        
        # Clear any leftover audio from previous sessions
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        # Start audio playback
        self.start_audio_playback()
        
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to ElevenLabs WebSocket")
                
                # Send initial message with voice settings and API key
                initial_message = {
                    "text": " ",
                    "voice_settings": {
                        "speed": 1.0,
                        "stability": 0.5,
                        "similarity_boost": 0.8
                    },
                    "xi_api_key": self.api_key
                }
                await websocket.send(json.dumps(initial_message))
                print("Sent initial voice settings")
                
                # Start listening for audio responses
                audio_task = asyncio.create_task(self._handle_audio_responses(websocket))
                
                # Stream LLM completion to TTS
                print(f"Getting LLM completion for: {prompt}")
                text_buffer = ""
                
                async for text_chunk in self.get_llm_completion(prompt):
                    print(f"LLM chunk: {repr(text_chunk)}")
                    text_buffer += text_chunk
                    
                    # Send text chunks to ElevenLabs
                    # We can send after accumulating a few words or punctuation
                    if any(punct in text_buffer for punct in ['.', '!', '?', ',']) or len(text_buffer) > 50:
                        message = {
                            "text": text_buffer,
                            "try_trigger_generation": True
                        }
                        await websocket.send(json.dumps(message))
                        print(f"Sent to TTS: {repr(text_buffer)}")
                        text_buffer = ""
                        
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(0.05)
                
                # Send any remaining text
                if text_buffer.strip():
                    message = {
                        "text": text_buffer,
                        "try_trigger_generation": True
                    }
                    await websocket.send(json.dumps(message))
                    print(f"Sent final to TTS: {repr(text_buffer)}")
                
                # Send empty text to signal completion
                await websocket.send(json.dumps({"text": ""}))
                print("Sent completion signal")
                
                # Wait for audio task to complete or timeout
                try:
                    await asyncio.wait_for(audio_task, timeout=15.0)  # Longer timeout
                except asyncio.TimeoutError:
                    print("⏰ Audio task timed out")
                    audio_task.cancel()
                except asyncio.CancelledError:
                    pass
                
                # Give extra time for remaining audio to play
                print("⏳ Waiting for final audio to complete...")
                await asyncio.sleep(2.0)
                
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            # Stop audio playback
            self.stop_audio_playback()
            
    async def _handle_audio_responses(self, websocket):
        """Handle incoming audio responses from ElevenLabs."""
        try:
            async for message in websocket:
                # Check if message is valid
                if message is None:
                    continue
                
                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Failed to parse WebSocket message: {e}")
                    continue
                
                if "audio" in data and data["audio"]:
                    try:
                        # Decode base64 audio and add to playback queue
                        audio_data = base64.b64decode(data["audio"])
                        if len(audio_data) > 0:  # Only queue non-empty audio
                            self.audio_queue.put(audio_data)
                            print(f"📨 Queued audio chunk: {len(audio_data)} bytes, queue size: {self.audio_queue.qsize()}")
                    except Exception as e:
                        print(f"Failed to decode audio data: {e}")
                        continue
                    
                    # Print alignment info if available
                    if "normalizedAlignment" in data and data["normalizedAlignment"]:
                        alignment = data["normalizedAlignment"]
                        if isinstance(alignment, dict) and "chars" in alignment:
                            chars = alignment.get("chars", [])
                            if chars and isinstance(chars, list):
                                print(f"TTS playing: {''.join(chars)}")
                
                if data.get("isFinal", False):
                    print("🏁 TTS generation completed - waiting for queue to drain")
                    # Wait for audio queue to drain before breaking
                    while not self.audio_queue.empty():
                        await asyncio.sleep(0.1)
                    print("🎯 Audio queue drained")
                    break
                    
        except asyncio.CancelledError:
            print("Audio response handler cancelled")
        except Exception as e:
            print(f"Audio response error: {e}")
            
    def cleanup(self):
        """Clean up resources."""
        self.stop_audio_playback()
        self.p.terminate()

async def main():
    print("ElevenLabs Streaming TTS Test")
    print("=" * 40)
    
    # Get API keys (you'll need to provide these)
    elevenlabs_api_key = input("Enter your ElevenLabs API key: ").strip()
    if not elevenlabs_api_key:
        print("ElevenLabs API key is required!")
        return
        
    openai_api_key = input("Enter your OpenAI API key: ").strip()
    if not openai_api_key:
        print("OpenAI API key is required!")
        return
    
    # Default voice ID (Rachel voice)
    voice_id = input("Enter voice ID (or press Enter for default 'pNInz6obpgDQGcFmaJgB'): ").strip()
    if not voice_id:
        voice_id = "T2KZm9rWPG5TgXTyjt7E" # catalyst

        #voice_id = "yjJ45q8TVCrtMhEKurxY"  # Von Fusion voice
    
    # Create streamer
    streamer = ElevenLabsStreamer(elevenlabs_api_key, voice_id, openai_api_key)
    
    try:
        while True:
            prompt = input("\nEnter your prompt (or 'quit' to exit): ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            if prompt:
                print(f"\nProcessing: {prompt}")
                await streamer.stream_tts(prompt)
                print("Completed!")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        streamer.cleanup()
        print("Cleaned up resources")

if __name__ == "__main__":
    asyncio.run(main()) 