#!/usr/bin/env python3
"""
Real-time Speech-to-Text with Live Diarization using Deepgram API
"""

import asyncio
import json
import os
import pyaudio
import threading
import time
from datetime import datetime
from typing import Dict, Optional

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Initialize colorama for colored console output
init(autoreset=True)

# Load environment variables
load_dotenv()

class RealTimeSTTDiarization:
    def __init__(self, api_key: str):
        """
        Initialize the real-time STT with diarization system.
        
        Args:
            api_key (str): Deepgram API key
        """
        self.api_key = api_key
        self.deepgram = DeepgramClient(api_key)
        self.connection = None
        self.microphone = None
        self.is_running = False
        self.speakers: Dict[int, str] = {}  # Track speakers and their colors
        self.speaker_colors = [
            Fore.CYAN,
            Fore.MAGENTA,
            Fore.YELLOW,
            Fore.GREEN,
            Fore.BLUE,
            Fore.RED,
        ]
        
    def get_speaker_color(self, speaker_id: int) -> str:
        """Get a consistent color for a speaker."""
        if speaker_id not in self.speakers:
            color_index = len(self.speakers) % len(self.speaker_colors)
            self.speakers[speaker_id] = self.speaker_colors[color_index]
        return self.speakers[speaker_id]
    
    def setup_connection(self):
        """Set up the WebSocket connection to Deepgram."""
        try:
            # Configure live transcription options
            options = LiveOptions(
                model="nova-3",
                language="en-US",
                smart_format=True,
                interim_results=True,
                utterance_end_ms=1000,
                vad_events=True,
                punctuate=True,
                diarize=True,  # Enable diarization
                multichannel=False,
                alternatives=1,
                # tier="nova",  # Removed - not available for this API key
            )
            
            # Create the connection
            self.connection = self.deepgram.listen.websocket.v("1")
            
            # Register event handlers
            self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
            self.connection.on(LiveTranscriptionEvents.Metadata, self.on_metadata)
            self.connection.on(LiveTranscriptionEvents.Error, self.on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
            
            # Start the connection
            if not self.connection.start(options):
                print(f"{Fore.RED}Failed to start Deepgram connection")
                return False
                
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error setting up connection: {str(e)}")
            return False
    
    def setup_microphone(self):
        """Set up the microphone for audio capture."""
        try:
            # Create microphone instance
            self.microphone = Microphone(self.connection.send)
            return True
        except Exception as e:
            print(f"{Fore.RED}Error setting up microphone: {str(e)}")
            return False
    
    def on_open(self, *args, **kwargs):
        """Handle connection open event."""
        print(f"{Fore.GREEN}✓ Connected to Deepgram")
        print(f"{Fore.GREEN}✓ Live diarization enabled")
        print(f"{Fore.WHITE}{'='*60}")
        print(f"{Fore.WHITE}Start speaking... (Ctrl+C to stop)")
        print(f"{Fore.WHITE}{'='*60}")
    
    def on_transcript(self, *args, **kwargs):
        """Handle transcript events with diarization."""
        try:
            result = kwargs.get('result')
            if result:
                transcript = result.get('transcript', '')
                if transcript:
                    is_final = result.get('is_final', False)
                    
                    # Get speaker information
                    channel = result.get('channel', {})
                    alternatives = channel.get('alternatives', [])
                    
                    if alternatives:
                        words = alternatives[0].get('words', [])
                        
                        # Group words by speaker
                        speaker_segments = {}
                        for word in words:
                            speaker = word.get('speaker', 0)
                            if speaker not in speaker_segments:
                                speaker_segments[speaker] = []
                            speaker_segments[speaker].append(word.get('word', ''))
                        
                        # Display transcript with speaker information
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        status = "FINAL" if is_final else "INTERIM"
                        
                        for speaker, words in speaker_segments.items():
                            speaker_color = self.get_speaker_color(speaker)
                            text = ' '.join(words)
                            
                            print(f"{Fore.WHITE}[{timestamp}] {speaker_color}Speaker {speaker}: {Style.BRIGHT}{text}{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}({status})")
                    else:
                        # Fallback if no speaker information available
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        status = "FINAL" if is_final else "INTERIM"
                        print(f"{Fore.WHITE}[{timestamp}] {Fore.WHITE}Unknown Speaker: {Style.BRIGHT}{transcript}{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}({status})")
                        
        except Exception as e:
            print(f"{Fore.RED}Error processing transcript: {str(e)}")
    
    def on_metadata(self, *args, **kwargs):
        """Handle metadata events."""
        try:
            metadata = kwargs.get('metadata')
            if metadata:
                # Handle metadata properly - it might be an object, not a dict
                request_id = getattr(metadata, 'request_id', None)
                if request_id:
                    print(f"{Fore.LIGHTBLACK_EX}[DEBUG] Request ID: {request_id}")
        except Exception as e:
            print(f"{Fore.RED}Error processing metadata: {str(e)}")
    
    def on_error(self, *args, **kwargs):
        """Handle error events."""
        error = kwargs.get('error')
        print(f"{Fore.RED}❌ Deepgram Error: {error}")
    
    def on_close(self, *args, **kwargs):
        """Handle connection close event."""
        print(f"{Fore.YELLOW}Connection closed")
    
    async def start_streaming(self):
        """Start the real-time streaming process."""
        try:
            print(f"{Fore.CYAN}🎤 Initializing Real-time STT with Live Diarization...")
            
            # Setup connection
            if not self.setup_connection():
                return
            
            # Setup microphone
            if not self.setup_microphone():
                return
            
            # Start microphone
            self.microphone.start()
            self.is_running = True
            
            # Keep the connection alive
            while self.is_running:
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Stopping...")
            self.stop_streaming()
        except Exception as e:
            print(f"{Fore.RED}Error during streaming: {str(e)}")
            self.stop_streaming()
    
    def stop_streaming(self):
        """Stop the streaming process."""
        self.is_running = False
        
        if self.microphone:
            self.microphone.finish()
            
        if self.connection:
            self.connection.finish()
        
        print(f"{Fore.GREEN}✓ Streaming stopped")


def main():
    """Main function to run the real-time STT with diarization."""
    # Get API key from environment or prompt user
    api_key = os.getenv('DEEPGRAM_API_KEY')
    
    if not api_key:
        print(f"{Fore.RED}Please set your DEEPGRAM_API_KEY environment variable")
        print(f"{Fore.WHITE}You can get an API key from: https://console.deepgram.com/")
        print(f"{Fore.WHITE}Then run: export DEEPGRAM_API_KEY='your_api_key_here'")
        return
    
    # Create and run the STT system
    stt_system = RealTimeSTTDiarization(api_key)
    
    try:
        asyncio.run(stt_system.start_streaming())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Exiting...")
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {str(e)}")


if __name__ == "__main__":
    main() 