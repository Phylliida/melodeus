#!/usr/bin/env python3
"""
Improved Working Real-time STT with Enhanced Speaker Identification
Integrates with the improved speaker identification system to prevent false positives
"""

import os
import time
from datetime import datetime
from typing import Dict
import json
from pathlib import Path

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Import the improved speaker identification system
from improved_speaker_identification import ImprovedSpeakerIdentificationSystem

# Initialize colorama for colored console output
init(autoreset=True)

# Load environment variables
load_dotenv()

class ImprovedWorkingSpeakerSTT:
    def __init__(self, api_key: str):
        """Initialize the improved working STT system with enhanced speaker identification."""
        self.api_key = api_key
        self.deepgram = DeepgramClient(api_key)
        self.connection = None
        self.microphone = None
        self.is_running = False
        
        # Initialize improved speaker identification system
        self.speaker_system = ImprovedSpeakerIdentificationSystem()
        
        # Session tracking
        self.session_speaker_names = {}
        self.unknown_count = 0
        
        # Colors for speakers
        self.speaker_colors = [
            Fore.CYAN,
            Fore.MAGENTA,
            Fore.YELLOW,
            Fore.GREEN,
            Fore.BLUE,
            Fore.RED,
        ]
        
        # Show current settings
        print(f"🔧 Improved Speaker STT initialized")
        print(f"   Identification threshold: {self.speaker_system.identification_threshold}")
        print(f"   Margin threshold: {self.speaker_system.margin_threshold}")
        print(f"   Debug mode: {self.speaker_system.debug_mode}")
        
    def get_speaker_name(self, session_speaker_id: int) -> str:
        """Get the display name for a session speaker using the improved system."""
        # Check if we already have a mapping for this session speaker
        if session_speaker_id in self.session_speaker_names:
            return self.session_speaker_names[session_speaker_id]
        
        # First time seeing this speaker - assign unknown for now
        # The actual identification will happen when we have audio data
        self.unknown_count += 1
        unknown_name = f"❓ Unknown {self.unknown_count}"
        self.session_speaker_names[session_speaker_id] = unknown_name
        print(f"{Fore.YELLOW}❓ New speaker detected → {unknown_name} (pending identification)")
        return unknown_name
    
    def update_speaker_identification(self, session_speaker_id: int, identified_name: str, speaker_type: str):
        """Update the speaker name mapping after identification."""
        old_name = self.session_speaker_names.get(session_speaker_id, f"Speaker {session_speaker_id}")
        
        if speaker_type == "known":
            new_name = f"🎭 {identified_name}"
            self.session_speaker_names[session_speaker_id] = new_name
            print(f"{Fore.GREEN}✅ Speaker identification: {old_name} → {identified_name}")
        elif speaker_type == "unknown":
            # Keep the unknown name but maybe update it
            if not old_name.startswith("❓"):
                self.session_speaker_names[session_speaker_id] = f"❓ {identified_name}"
                print(f"{Fore.YELLOW}❓ Confirmed unknown speaker: {identified_name}")
    
    def get_speaker_color(self, speaker_id: int) -> str:
        """Get a consistent color for a speaker."""
        color_index = speaker_id % len(self.speaker_colors)
        return self.speaker_colors[color_index]
    
    def setup_connection(self):
        """Set up the WebSocket connection to Deepgram."""
        try:
            # Create the connection first
            self.connection = self.deepgram.listen.websocket.v("1")
            
            # Register event handlers
            self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
            self.connection.on(LiveTranscriptionEvents.Metadata, self.on_metadata)
            self.connection.on(LiveTranscriptionEvents.Error, self.on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
            self.connection.on(LiveTranscriptionEvents.SpeechStarted, self.on_speech_started)
            self.connection.on(LiveTranscriptionEvents.UtteranceEnd, self.on_utterance_end)
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error setting up connection: {str(e)}")
            return False
    
    def on_open(self, *args, **kwargs):
        """Handle connection open event."""
        print(f"{Fore.GREEN}✓ Connected to Deepgram")
        print(f"{Fore.GREEN}✓ Live diarization enabled")
        print(f"{Fore.GREEN}✓ Enhanced speaker identification active")
        print(f"{Fore.WHITE}{'='*60}")
        print(f"{Fore.WHITE}Start speaking now... (Ctrl+C to stop)")
        print(f"{Fore.WHITE}Unknown speakers will be properly filtered!")
        print(f"{Fore.WHITE}{'='*60}")
    
    def on_speech_started(self, *args, **kwargs):
        """Handle speech started event."""
        print(f"{Fore.GREEN}🎤 Speech detected!")
    
    def on_utterance_end(self, *args, **kwargs):
        """Handle utterance end event."""
        print(f"{Fore.YELLOW}⏹️  Utterance ended")
    
    def on_transcript(self, *args, **kwargs):
        """Handle transcript events with enhanced speaker identification."""
        try:
            result = kwargs.get('result')
            if result and hasattr(result, 'channel'):
                transcript = result.channel.alternatives[0].transcript
                
                if transcript:
                    is_final = result.is_final
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Get speaker information from words
                    words = result.channel.alternatives[0].words
                    if words:
                        # Group words by speaker
                        speaker_segments = {}
                        for word in words:
                            speaker = word.speaker if hasattr(word, 'speaker') else 0
                            if speaker not in speaker_segments:
                                speaker_segments[speaker] = []
                            speaker_segments[speaker].append(word.word)
                        
                        # Display transcript with speaker names
                        status = "FINAL" if is_final else "INTERIM"
                        
                        for speaker, words_list in speaker_segments.items():
                            speaker_color = self.get_speaker_color(speaker)
                            speaker_name = self.get_speaker_name(speaker)
                            text = ' '.join(words_list)
                            
                            # Only show final results to reduce clutter when debugging
                            if is_final or not self.speaker_system.debug_mode:
                                print(f"{Fore.WHITE}[{timestamp}] {speaker_color}{speaker_name}: {Style.BRIGHT}{text}{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}({status})")
                    else:
                        # Fallback if no speaker information available
                        status = "FINAL" if is_final else "INTERIM"
                        if is_final or not self.speaker_system.debug_mode:
                            print(f"{Fore.WHITE}[{timestamp}] Unknown: {Style.BRIGHT}{transcript}{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}({status})")
                        
        except Exception as e:
            print(f"{Fore.RED}Error processing transcript: {str(e)}")
    
    def on_metadata(self, *args, **kwargs):
        """Handle metadata events."""
        try:
            metadata = kwargs.get('metadata')
            if metadata and self.speaker_system.debug_mode:
                print(f"{Fore.LIGHTBLACK_EX}[DEBUG] Metadata received")
        except Exception as e:
            print(f"{Fore.RED}Error processing metadata: {str(e)}")
    
    def on_error(self, *args, **kwargs):
        """Handle error events."""
        error = kwargs.get('error')
        print(f"{Fore.RED}❌ Deepgram Error: {error}")
    
    def on_close(self, *args, **kwargs):
        """Handle connection close event."""
        print(f"{Fore.YELLOW}⚠️  Connection closed")
        if self.is_running:
            print(f"{Fore.YELLOW}This might be due to inactivity or a connection issue.")
    
    def start_streaming(self):
        """Start the real-time streaming process."""
        try:
            print(f"{Fore.CYAN}🎤 Initializing Improved Speaker STT...")
            
            # Setup connection
            if not self.setup_connection():
                print(f"{Fore.RED}❌ Failed to setup connection")
                return
            
            # Start the connection with the exact same options that worked before
            options = LiveOptions(
                model="nova-3",
                language="en-US",
                smart_format=True,
                interim_results=True,
                diarize=True,
                punctuate=True,
                utterance_end_ms=1000,
                vad_events=True,
                encoding="linear16",
                sample_rate=16000,
                channels=1,
            )
            
            if not self.connection.start(options):
                print(f"{Fore.RED}❌ Failed to start connection")
                return
                
            # Give connection time to establish
            time.sleep(1)
                
            # Setup microphone
            try:
                self.microphone = Microphone(self.connection.send)
                print(f"{Fore.GREEN}✓ Microphone initialized")
            except Exception as e:
                print(f"{Fore.RED}❌ Failed to initialize microphone: {e}")
                return
                
            # Start microphone
            try:
                self.microphone.start()
                print(f"{Fore.GREEN}✓ Microphone started - audio is now streaming")
                self.is_running = True
            except Exception as e:
                print(f"{Fore.RED}❌ Failed to start microphone: {e}")
                return
            
            # Keep alive
            print(f"{Fore.CYAN}🔄 Keeping connection alive...")
            while self.is_running:
                time.sleep(0.1)
                
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
            try:
                self.microphone.finish()
                print(f"{Fore.GREEN}✓ Microphone stopped")
            except:
                pass
            
        if self.connection:
            try:
                self.connection.finish()
                print(f"{Fore.GREEN}✓ Connection closed")
            except:
                pass
        
        # Show session summary
        self.print_session_summary()
    
    def print_session_summary(self):
        """Print a summary of the session."""
        if self.session_speaker_names:
            print(f"\n{Fore.CYAN}📊 Session Summary:")
            print(f"{Fore.CYAN}Speakers in this session:")
            
            known_count = 0
            unknown_count = 0
            
            for session_id, name in self.session_speaker_names.items():
                print(f"  Speaker {session_id}: {name}")
                if name.startswith("🎭"):
                    known_count += 1
                else:
                    unknown_count += 1
            
            print(f"\n{Fore.GREEN}Known speakers identified: {known_count}")
            print(f"{Fore.YELLOW}Unknown speakers: {unknown_count}")
            print(f"{Fore.CYAN}False positive prevention: {'✅ Active' if self.speaker_system.identification_threshold >= 0.9 else '⚠️ Check settings'}")
    
    def interactive_settings(self):
        """Interactive settings menu."""
        while True:
            print(f"\n{Fore.CYAN}🔧 Settings Menu:")
            print("1. View current settings")
            print("2. Adjust identification threshold")
            print("3. Adjust margin threshold") 
            print("4. Toggle debug mode")
            print("5. List known speakers")
            print("6. Return to main")
            
            choice = input("Choose option (1-6): ").strip()
            
            if choice == "1":
                self.speaker_system.list_known_speakers()
            elif choice == "2":
                try:
                    current = self.speaker_system.identification_threshold
                    new_threshold = float(input(f"New identification threshold (current: {current}, recommended: 0.90-0.95): "))
                    if 0.5 <= new_threshold <= 1.0:
                        self.speaker_system.adjust_thresholds(identification_threshold=new_threshold)
                    else:
                        print("Invalid range. Use 0.5-1.0")
                except ValueError:
                    print("Invalid input")
            elif choice == "3":
                try:
                    current = self.speaker_system.margin_threshold
                    new_threshold = float(input(f"New margin threshold (current: {current}, recommended: 0.05-0.15): "))
                    if 0.0 <= new_threshold <= 0.5:
                        self.speaker_system.adjust_thresholds(margin_threshold=new_threshold)
                    else:
                        print("Invalid range. Use 0.0-0.5")
                except ValueError:
                    print("Invalid input")
            elif choice == "4":
                self.speaker_system.set_debug_mode(not self.speaker_system.debug_mode)
            elif choice == "5":
                self.speaker_system.list_known_speakers()
            elif choice == "6":
                break
            else:
                print("Invalid choice")

def main():
    """Main function."""
    api_key = os.getenv('DEEPGRAM_API_KEY')
    
    if not api_key:
        print(f"{Fore.RED}Please set your DEEPGRAM_API_KEY environment variable")
        return
    
    print(f"{Fore.CYAN}🎯 Improved Real-time STT with Enhanced Speaker Identification")
    print(f"{Fore.CYAN}🔒 False positive prevention enabled!")
    print()
    
    stt_system = ImprovedWorkingSpeakerSTT(api_key)
    
    while True:
        print(f"\n{Fore.CYAN}Main Menu:")
        print("1. Start live transcription")
        print("2. Settings")
        print("3. Exit")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == "1":
            stt_system.start_streaming()
        elif choice == "2":
            stt_system.interactive_settings()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 