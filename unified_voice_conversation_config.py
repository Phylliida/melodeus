#!/usr/bin/env python3
"""
Unified Voice Conversation System with YAML Configuration
Integrates modular STT and TTS systems with YAML-based configuration management.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from openai import OpenAI

# Import our modular systems and config loader
from async_stt_module import AsyncSTTStreamer, STTEventType, STTResult
from async_tts_module import AsyncTTSStreamer
from config_loader import load_config, VoiceAIConfig

@dataclass 
class ConversationState:
    """Tracks the current state of the conversation."""
    is_active: bool = False
    is_processing_llm: bool = False
    is_speaking: bool = False
    last_utterance_time: Optional[datetime] = None
    utterance_buffer: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_speaker: Optional[int] = None

class UnifiedVoiceConversation:
    """Unified voice conversation system with YAML configuration."""
    
    def __init__(self, config: VoiceAIConfig):
        self.config = config
        self.state = ConversationState()
        
        # Initialize LLM client
        self.openai_client = OpenAI(api_key=config.conversation.openai_api_key)
        
        # Initialize STT system
        self.stt = AsyncSTTStreamer(config.stt)
        
        # Initialize TTS system
        self.tts = AsyncTTSStreamer(config.tts)
        
        # Conversation management task
        self.conversation_task = None
        
        # Set up STT callbacks
        self._setup_stt_callbacks()
        
        # Apply logging configuration
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging based on configuration."""
        import logging
        
        # Set logging level
        level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.logger = logging.getLogger(__name__)
        
        # Store logging preferences
        self.show_interim = self.config.logging.show_interim_results
        self.show_tts_chunks = self.config.logging.show_tts_chunks
        self.show_audio_debug = self.config.logging.show_audio_debug
    
    def _setup_stt_callbacks(self):
        """Set up callbacks for STT events."""
        
        # Handle completed utterances
        self.stt.on(STTEventType.UTTERANCE_COMPLETE, self._on_utterance_complete)
        
        # Handle interim results (for interruption detection)
        self.stt.on(STTEventType.INTERIM_RESULT, self._on_interim_result)
        
        # Handle speech events
        self.stt.on(STTEventType.SPEECH_STARTED, self._on_speech_started)
        self.stt.on(STTEventType.SPEECH_ENDED, self._on_speech_ended)
        
        # Handle speaker changes
        self.stt.on(STTEventType.SPEAKER_CHANGE, self._on_speaker_change)
        
        # Handle errors
        self.stt.on(STTEventType.ERROR, self._on_error)
    
    async def start_conversation(self):
        """Start the voice conversation system."""
        print("🎙️ Starting Unified Voice Conversation System (YAML Config)")
        print("=" * 60)
        
        # Show configuration summary
        self._show_config_summary()
        
        try:
            # Start STT
            print("🎤 Starting speech recognition...")
            if not await self.stt.start_listening():
                print("❌ Failed to start STT")
                return False
            
            self.state.is_active = True
            
            # Start conversation management
            self.conversation_task = asyncio.create_task(self._conversation_manager())
            
            print("✅ Conversation system active!")
            print("💡 Tips:")
            print("   - Speak naturally and pause when done")
            print("   - You can interrupt the AI by speaking while it talks")
            print("   - Press Ctrl+C to exit")
            print()
            
            # Keep running until stopped
            while self.state.is_active:
                await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting conversation: {e}")
            return False
    
    def _show_config_summary(self):
        """Show a summary of the loaded configuration."""
        print(f"📋 Configuration Summary:")
        print(f"   🎯 Voice ID: {self.config.conversation.voice_id}")
        print(f"   🎤 STT Model: {self.config.stt.model} ({self.config.stt.language})")
        print(f"   🔊 TTS Model: {self.config.tts.model_id}")
        print(f"   🤖 LLM Model: {self.config.conversation.llm_model}")
        print(f"   ⏱️  Pause Threshold: {self.config.conversation.pause_threshold}s")
        print(f"   📝 Min Words: {self.config.conversation.min_words_for_submission}")
        print(f"   🔇 Interruption Confidence: {self.config.conversation.interruption_confidence}")
        
        if self.config.development.enable_debug_mode:
            print(f"   🐛 Debug Mode: Enabled")
        
        print()
    
    async def stop_conversation(self):
        """Stop the conversation system."""
        print("🛑 Stopping conversation...")
        self.state.is_active = False
        
        # Cancel conversation management
        if self.conversation_task and not self.conversation_task.done():
            self.conversation_task.cancel()
            try:
                await self.conversation_task
            except asyncio.CancelledError:
                pass
        
        # Stop TTS if speaking
        if self.state.is_speaking:
            await self.tts.stop()
        
        # Stop STT
        await self.stt.stop_listening()
        
        print("✅ Conversation stopped")
    
    async def _on_utterance_complete(self, result: STTResult):
        """Handle completed utterances from STT."""
        speaker_info = f" (Speaker {result.speaker_id})" if result.speaker_id is not None else ""
        print(f"🎯 Final{speaker_info}: {result.text}")
        
        # Check for interruption at ANY stage of AI response
        interrupted = False
        
        if self.tts.is_currently_playing():
            print(f"🛑 Interrupting TTS playback with: {result.text}")
            await self.tts.stop()
            interrupted = True
            
        elif self.state.is_processing_llm:
            print(f"🛑 Interrupting LLM generation with: {result.text}")
            # Cancel any ongoing LLM tasks (will be handled in stream_llm_to_tts)
            interrupted = True
            
        elif self.state.is_speaking:
            print(f"🛑 Interrupting TTS setup with: {result.text}")
            # Stop any TTS preparation
            await self.tts.stop()
            interrupted = True
        
        if interrupted:
            # Clear ALL state flags to ensure next utterance processes
            self.state.is_speaking = False
            self.state.is_processing_llm = False
            print("🔄 All AI processing interrupted and state cleared")
            print(f"🔄 Utterance buffer now has: {len(self.state.utterance_buffer)} items")
            
            # Force immediate processing of the interrupting utterance
            if len(self.state.utterance_buffer) > 0:
                print("🚀 Force processing interrupting utterance immediately")
                asyncio.create_task(self._force_process_buffer())
        
        # Add to utterance buffer
        self.state.utterance_buffer.append(result.text)
        self.state.last_utterance_time = result.timestamp
        self.state.current_speaker = result.speaker_id
        
        print(f"📝 Added to buffer: '{result.text}' at {result.timestamp}")
        print(f"📊 Buffer size: {len(self.state.utterance_buffer)}, Last time: {self.state.last_utterance_time}")
        
        if self.config.development.enable_debug_mode:
            self.logger.debug(f"Utterance added to buffer. Buffer size: {len(self.state.utterance_buffer)}")
    
    async def _on_interim_result(self, result: STTResult):
        """Handle interim results for showing progress."""
        # Interruption is now handled only in _on_utterance_complete for final utterances
        # This is just for showing interim results
        
        # Show interim results based on configuration
        if self.show_interim:
            print(f"💭 Interim: {result.text}")
    
    async def _on_speech_started(self, data):
        """Handle speech start events."""
        # Just show detection, don't interrupt yet - wait for complete utterance
        if self.tts.is_currently_playing():
            print("🚫 Speech detected during TTS - waiting for complete utterance")
        elif self.state.is_speaking:
            print("🚫 Speech detected during conversation processing")
        
        if self.show_audio_debug:
            self.logger.debug("Speech started event received")
    
    async def _on_speech_ended(self, data):
        """Handle speech end events."""
        print("🔇 Speech ended")
        self.state.last_utterance_time = data['timestamp']
        
        if self.show_audio_debug:
            self.logger.debug("Speech ended event received")
    
    async def _on_speaker_change(self, data):
        """Handle speaker change events."""
        speaker_id = data['speaker_id']
        speaker_name = data.get('speaker_name', f"Speaker {speaker_id}")
        print(f"👤 Speaker changed to: {speaker_name}")
        
        if self.config.development.enable_debug_mode:
            self.logger.debug(f"Speaker change: {speaker_id} -> {speaker_name}")
    
    async def _on_error(self, data):
        """Handle STT errors."""
        error_msg = data['error']
        print(f"❌ STT Error: {error_msg}")
        self.logger.error(f"STT Error: {error_msg}")
        
        # Check if this is a connection error that might require restart
        if "Connection send failed" in error_msg or "ConnectionClosed" in error_msg:
            print("🔄 Detected connection failure, attempting to restart STT...")
            try:
                # Try to restart STT connection
                await self.stt.stop_listening()
                await asyncio.sleep(0.01)  # Minimal pause
                success = await self.stt.start_listening()
                if success:
                    print("✅ STT connection restarted successfully")
                else:
                    print("❌ Failed to restart STT connection")
                    await self._speak_text("I'm having trouble with speech recognition. Please try again.")
            except Exception as e:
                print(f"❌ Error restarting STT: {e}")
                self.logger.error(f"STT restart error: {e}")
    
    async def _conversation_manager(self):
        """Manages conversation flow and LLM submission decisions."""
        print("🔄 Conversation manager started")
        last_debug_time = datetime.now()
        
        while self.state.is_active:
            try:
                await asyncio.sleep(0.01)  # Check every 10ms for maximum responsiveness
                
                # Show debug info every 5 seconds if we have data but no submission
                now = datetime.now()
                if (now - last_debug_time).total_seconds() >= 5.0:
                    if self.state.utterance_buffer and self.state.last_utterance_time:
                        combined_transcript = " ".join(self.state.utterance_buffer).strip()
                        word_count = len(combined_transcript.split()) if combined_transcript else 0
                        time_since_speech = now - self.state.last_utterance_time
                        
                        print(f"🐛 Debug: Buffer='{combined_transcript}' ({word_count} words), "
                              f"Time since speech: {time_since_speech.total_seconds():.1f}s, "
                              f"Processing: {self.state.is_processing_llm}, "
                              f"Speaking: {self.state.is_speaking}")
                    last_debug_time = now
                
                # Skip if currently processing or speaking
                if (self.state.is_processing_llm or 
                    self.state.is_speaking or 
                    not self.state.last_utterance_time):
                    # Debug: Show why we're skipping (but less frequently)
                    if self.state.utterance_buffer and (now - last_debug_time).total_seconds() >= 2.0:
                        skip_reason = []
                        if self.state.is_processing_llm:
                            skip_reason.append("processing_llm")
                        if self.state.is_speaking:
                            skip_reason.append("speaking")
                        if not self.state.last_utterance_time:
                            skip_reason.append("no_utterance_time")
                        
                        if skip_reason:
                            print(f"⏸️ Skipping utterance buffer due to: {', '.join(skip_reason)}")
                            print(f"   Buffer: '{' '.join(self.state.utterance_buffer)}'")
                    continue
                
                # Calculate time since last speech
                time_since_speech = datetime.now() - self.state.last_utterance_time
                
                # Get combined transcript
                combined_transcript = " ".join(self.state.utterance_buffer).strip()
                word_count = len(combined_transcript.split()) if combined_transcript else 0
                
                # Decision logic for LLM submission
                should_submit = False
                reason = ""
                
                # IMMEDIATE submission - no waiting for pauses!
                if word_count >= 1:  # Process any speech immediately
                    should_submit = True
                    if self._seems_complete(combined_transcript):
                        reason = "Statement appears complete"
                    else:
                        reason = f"Immediate processing of {word_count} words"
                
                # Submit to LLM if criteria met
                if should_submit:
                    print(f"🧠 Submitting to LLM: {reason}")
                    print(f"📝 Input: {combined_transcript}")
                    
                    if self.config.development.enable_debug_mode:
                        self.logger.debug(f"LLM submission triggered: {reason}")
                    
                    await self._process_with_llm(combined_transcript)
                    
                    # Clear buffer and reset timing
                    self.state.utterance_buffer.clear()
                    self.state.last_utterance_time = None
                    print(f"🧹 Buffer cleared after successful LLM submission")
                else:
                    # Debug: show why we're not submitting
                    if self.state.utterance_buffer and (now - last_debug_time).total_seconds() >= 2.0:
                        print(f"🤔 Not submitting: '{combined_transcript}' ({word_count} words, {time_since_speech.total_seconds():.1f}s ago)")
                        print(f"   Reason: word_count={word_count}, time_since={time_since_speech.total_seconds():.1f}s")
                
            except Exception as e:
                print(f"Conversation manager error: {e}")
                self.logger.error(f"Conversation manager error: {e}")
                import traceback
                traceback.print_exc()
    
    def _seems_complete(self, text: str) -> bool:
        """Heuristic to determine if a statement seems complete."""
        if not text:
            return False
        
        # Check for ending punctuation
        if text.rstrip().endswith(('.', '!', '?')):
            return True
        
        # Check for common complete phrases
        complete_patterns = [
            r'\bthat(?:[\'s]|s) (it|all|everything)\b',
            r'\b(okay|alright|thanks?|thank you)\b\s*$',
            r'\b(done|finished|complete)\b\s*$'
        ]
        
        for pattern in complete_patterns:
            if re.search(pattern, text.lower()):
                return True
        
        return False
    
    async def _process_with_llm(self, user_input: str):
        """Process user input with LLM and speak response."""
        if self.state.is_processing_llm:
            print(f"⚠️ Already processing LLM, skipping: {user_input}")
            return
        
        try:
            self.state.is_processing_llm = True
            print(f"🔄 Starting LLM processing for: '{user_input}'")
            
            # Add to conversation history
            self.state.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            print("🤖 Getting LLM response...")
            
            # Prepare messages
            messages = [
                {"role": "system", "content": self.config.conversation.system_prompt}
            ] + self.state.conversation_history[-10:]  # Keep last 10 exchanges
            
            # Stream LLM response to TTS
            await self._stream_llm_to_tts(messages)
            
        except Exception as e:
            print(f"LLM processing error: {e}")
            self.logger.error(f"LLM processing error: {e}")
            await self._speak_text("Sorry, I encountered an error processing your request.")
        finally:
            self.state.is_processing_llm = False
            print(f"🔄 LLM processing finished, state cleared")
    
    async def _stream_llm_to_tts(self, messages: List[Dict[str, str]]):
        """Stream LLM completion to TTS."""
        assistant_response = ""  # Initialize at method level to fix scoping
        
        try:
            self.state.is_speaking = True
            
            # Create async generator for LLM response
            async def llm_generator():
                nonlocal assistant_response  # Access the method-level variable
                # Use the configured model from YAML
                models_to_try = [self.config.conversation.llm_model]
                response = None
                
                for model in models_to_try:
                    try:
                        response = self.openai_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            stream=True,
                            max_tokens=self.config.conversation.max_tokens
                        )
                        break
                    except Exception as e:
                        print(f"Failed with model {model}: {e}")
                        if model == models_to_try[-1]:
                            yield "Sorry, I couldn't connect to any available models."
                            return
                        continue
                
                if not response:
                    yield "Sorry, no models are available right now."
                    return
                
                for chunk in response:
                    # Check for interruption during LLM generation
                    if not self.state.is_speaking or not self.state.is_processing_llm:
                        print("🛑 LLM streaming interrupted by user")
                        break
                        
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        assistant_response += content
                        
                        if self.show_tts_chunks:
                            print(f"🔊 TTS chunk: {content}")
                        
                        yield content
                

            
            # Use TTS to speak the streaming response
            result = await self.tts.speak_stream(llm_generator())
            
            if result and self.state.is_speaking:
                # Only add to history if response completed successfully AND wasn't interrupted
                if assistant_response.strip():
                    self.state.conversation_history.append({
                        "role": "assistant", 
                        "content": assistant_response.strip()
                    })
                    
                    if self.config.development.enable_debug_mode:
                        self.logger.debug(f"Complete assistant response added to history: {len(assistant_response)} chars")
                
                print("✅ Response completed successfully")
            else:
                print("🛑 Response was interrupted - no history added")
                
        except Exception as e:
            print(f"TTS streaming error: {e}")
            self.logger.error(f"TTS streaming error: {e}")
        finally:
            self.state.is_speaking = False
    
    async def _speak_text(self, text: str):
        """Speak a simple text message."""
        try:
            self.state.is_speaking = True
            result = await self.tts.speak_text(text)
            if not result:
                print("🛑 Speech was interrupted")
        finally:
            self.state.is_speaking = False
    
    async def _force_process_buffer(self):
        """Force immediate processing of buffer after interruption."""
        # Wait a tiny bit for state to settle
        await asyncio.sleep(0.01)
        
        if not self.state.utterance_buffer or not self.state.last_utterance_time:
            return
            
        # Get combined transcript
        combined_transcript = " ".join(self.state.utterance_buffer).strip()
        
        if combined_transcript:
            print(f"🚀 Force processing buffer: '{combined_transcript}'")
            await self._process_with_llm(combined_transcript)
            
            # Clear buffer and reset timing
            self.state.utterance_buffer.clear()
            self.state.last_utterance_time = None
            print(f"🧹 Buffer cleared after forced processing")
    
    async def cleanup(self):
        """Clean up all resources."""
        try:
            await self.stop_conversation()
        except Exception as e:
            print(f"Error stopping conversation: {e}")
        
        try:
            await self.stt.cleanup()
        except Exception as e:
            print(f"Error cleaning up STT: {e}")
        
        try:
            await self.tts.cleanup()
        except Exception as e:
            print(f"Error cleaning up TTS: {e}")

async def main():
    """Main function for the unified voice conversation system."""
    print("🎙️ Unified Voice Conversation System (YAML Config)")
    print("==================================================")
    
    conversation = None
    
    try:
        # Load configuration from YAML
        print("📁 Loading configuration...")
        config = load_config()
        print("✅ Configuration loaded successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 To create a config file:")
        print("   1. Copy config.yaml.example to config.yaml")
        print("   2. Fill in your API keys")
        print("   3. Adjust settings as needed")
        print("\n   OR run: python config_loader.py create-example")
        return
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return
    except Exception as e:
        print(f"❌ Unexpected error loading config: {e}")
        return
    
    try:
        # Create conversation system
        conversation = UnifiedVoiceConversation(config)
        
        # Setup signal handling for graceful shutdown
        import signal
        
        def signal_handler():
            print("\n🛑 Received shutdown signal")
            if conversation:
                conversation.state.is_active = False
        
        # Register signal handlers
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                signal.signal(sig, lambda s, f: signal_handler())
            except ValueError:
                # Signal may not be available on all platforms
                pass
        
        success = await conversation.start_conversation()
        if not success:
            print("❌ Failed to start conversation")
            return
            
        # Keep running until interrupted
        print("🎯 Conversation is active. Press Ctrl+C to exit.")
        try:
            while conversation.state.is_active:
                await asyncio.sleep(0.5)  # Shorter sleep for more responsive shutdown
        except asyncio.CancelledError:
            print("\n⏹️ Task cancelled")
            
    except KeyboardInterrupt:
        print("\n👋 Conversation ended by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conversation:
            print("🧹 Cleaning up...")
            try:
                await conversation.cleanup()
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")
            print("✅ Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main()) 
