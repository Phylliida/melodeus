#!/usr/bin/env python3
"""
Async TTS Module for ElevenLabs WebSocket Streaming
Provides interruptible text-to-speech with real-time audio playback.
Now includes Whisper-based tracking of actual spoken content.
"""

import asyncio
import websockets
import json
import base64
import time
import numpy as np
import re
import difflib
from typing import Optional, AsyncGenerator, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from scipy import signal

try:
    from mel_aec_audio import (
        ensure_stream_started,
        shared_sample_rate,
        write_playback_pcm,
        interrupt_playback,
    )
except ImportError:  # pragma: no cover - fallback when running from repo root
    import os
    import sys

    CURRENT_DIR = os.path.dirname(__file__)
    if CURRENT_DIR and CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from mel_aec_audio import (
        ensure_stream_started,
        shared_sample_rate,
        write_playback_pcm,
        interrupt_playback,
    )

# Import Whisper TTS Tracker
try:
    from whisper_tts_tracker import WhisperTTSTracker, SpokenContent
    WHISPER_AVAILABLE = True
except ImportError:
    print("⚠️ Whisper TTS Tracker not available - spoken content tracking disabled")
    WHISPER_AVAILABLE = False
    SpokenContent = None

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
    # Multi-voice support
    emotive_voice_id: Optional[str] = None  # Voice for text in asterisks (*emotive text*)
    emotive_speed: float = 1.0
    emotive_stability: float = 0.5
    emotive_similarity_boost: float = 0.8
    # Audio output device
    output_device_name: Optional[str] = None  # None = default device, or specify device name

@dataclass
class ToolCall:
    """Represents a tool call embedded in the text."""
    tag_name: str  # e.g., "function", "search", etc.
    content: str  # The full XML content including tags
    start_position: int  # Character position in generated_text where tool should execute
    end_position: int  # Character position where tool content ends
    executed: bool = False

@dataclass
class ToolResult:
    """Result from tool execution."""
    should_interrupt: bool  # Whether to interrupt current speech
    content: Optional[str] = None  # Optional content to insert into conversation/speak
    
@dataclass
class TTSSession:
    """Represents a single TTS speaking session with its own isolated state."""
    session_id: str
    generated_text: str = ""  # Text that was sent to TTS for this session
    current_spoken_content: List[SpokenContent] = field(default_factory=list)  # What Whisper captured for this session
    was_interrupted: bool = False  # Whether this session was interrupted
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    whisper_tracker: Optional['WhisperTTSTracker'] = None  # Session-specific tracker
    tool_calls: List[ToolCall] = field(default_factory=list)  # Tools to execute during speech
    spoken_text_for_tts: str = ""  # Text actually sent to TTS (excludes tool content)

class AsyncTTSStreamer:
    """Async TTS Streamer with interruption capabilities and spoken content tracking."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.is_streaming = False
        self.websocket = None
        self.audio_task = None
        self.is_playing = False

        # Audio setup (shared mel-aec duplex stream)
        self.stream_sample_rate = shared_sample_rate()
        
        # Control flags
        self._stop_requested = False
        self._interrupted = False
        
        # Callback for first audio chunk
        self.first_audio_callback = None
        self._first_audio_received = False
        
        # Multi-voice completion tracking
        self._websockets_completed = 0
        self._total_websockets = 0
        self._all_text_sent = False  # Track when all text has been sent to websockets
        
        # Whisper tracking configuration
        self.track_spoken_content = WHISPER_AVAILABLE
        
        # Current session tracking
        self.current_session: Optional[TTSSession] = None
        self._session_counter = 0  # For generating unique session IDs
        self._chunks_played = 0  # Track how many audio chunks have been played
        
        # Store last session's generated text for recovery
        self.last_session_generated_text = ""
        
        # Tool execution callback
        self.on_tool_execution = None  # Callback: async def(tool_call: ToolCall) -> ToolResult
        
        # Voice connection management for prosodic continuity
        self._current_voice_connection = None  # Current voice connection
        self._current_voice_key = None  # Key of current voice
        self._current_voice_task = None  # Task handling current connection
        
        
        # Whisper will be initialized per session
    
    def _fuzzy_find_position(self, whisper_text: str, tts_text: str) -> int:
        """
        Find the approximate position in TTS text that corresponds to the end of Whisper text.
        Uses very fuzzy matching to handle Whisper transcription errors.
        
        Returns:
            Character position in tts_text that best matches the end of whisper_text
        """
        if not whisper_text or not tts_text:
            return 0
            
        # Normalize texts for comparison
        def normalize(text):
            # Convert to lowercase
            text = text.lower()
            # Keep ONLY letters and spaces - remove ALL punctuation, numbers, asterisks, etc.
            text = re.sub(r'[^a-z\s]', ' ', text)
            # Collapse multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Normalize and split into words for more robust matching
        norm_whisper = normalize(whisper_text)
        norm_tts = normalize(tts_text)
        
        whisper_words = norm_whisper.split()
        tts_words = norm_tts.split()
        
        if not whisper_words:
            return 0
            
        # Try to find the best match using a sliding window approach
        best_match_score = 0
        best_match_end = 0
        
        # Look for the last few words of Whisper in TTS (more reliable than full text)
        window_size = min(5, len(whisper_words))  # Use last 5 words or less
        search_words = whisper_words[-window_size:]
        
        # Slide through TTS text looking for best match
        for i in range(len(tts_words) - window_size + 1):
            window = tts_words[i:i + window_size]
            
            # Calculate similarity score
            matcher = difflib.SequenceMatcher(None, window, search_words)
            score = matcher.ratio()
            
            # If this is a better match, update
            if score > best_match_score:
                best_match_score = score
                best_match_end = i + window_size
        
        # If we found a good match (>60% similar), use it
        if best_match_score > 0.6:
            # Convert word position to character position
            word_count = 0
            for i, char in enumerate(tts_text):
                if char.isspace() and i > 0 and not tts_text[i-1].isspace():
                    word_count += 1
                    if word_count >= best_match_end:
                        return i
            
            # If we counted all words, return end of text
            return len(tts_text)
        
        # Fallback: Try character-level fuzzy matching on normalized text
        if norm_whisper in norm_tts:
            # Find the position in normalized text
            norm_pos = norm_tts.find(norm_whisper) + len(norm_whisper)
            # Estimate position in original text
            ratio = norm_pos / len(norm_tts)
            result = int(len(tts_text) * ratio)
            return result
        
        # Final fallback: percentage-based estimation with safety margin
        # Whisper often captures less than TTS sends, so add 10% margin
        whisper_ratio = len(whisper_text) / max(len(tts_text), 1)
        estimated_pos = int(len(tts_text) * min(whisper_ratio * 1.1, 1.0))
        
        return estimated_pos
    
    def _create_session(self) -> TTSSession:
        """Create a new TTS session with unique ID and optional Whisper tracker."""
        self._session_counter += 1
        session_id = f"tts_session_{int(time.time())}_{self._session_counter}"
        session = TTSSession(session_id=session_id)
        
        # Initialize Whisper tracker for this session if available
        if WHISPER_AVAILABLE and self.track_spoken_content:
            try:
                session.whisper_tracker = WhisperTTSTracker(sample_rate=16000)
                print(f"✅ Whisper TTS tracking enabled for session {session_id}")
            except Exception as e:
                print(f"⚠️ Failed to initialize Whisper tracker for session: {e}")
                session.whisper_tracker = None
        
        return session
        
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
            self._first_audio_received = False
            self._chunks_played = 0
            self._chunks_played = 0  # Reset chunks played counter
            
            # Create a new session for this speaking operation
            self.current_session = self._create_session()
            self.current_session.generated_text = text
            # Store for recovery in case of error
            self.last_session_generated_text = text
            
            if self.current_session.whisper_tracker and self.track_spoken_content:
                self.current_session.whisper_tracker.start_tracking()
            
            await self._start_streaming(text)
            
            # If we reached here without stop being requested, it completed successfully
            completed_successfully = not self._stop_requested
            self._interrupted = self._stop_requested  # Set interrupted flag based on actual completion
            
            return completed_successfully
            
        except Exception as e:
            print(f"TTS error: {e}")
            return False
        finally:
            await self._cleanup_stream()
    
    def _parse_emotive_text(self, text: str) -> List[Tuple[str, bool]]:
        """
        Parse text to separate regular text from emotive text (in asterisks).
        
        Args:
            text: Input text that may contain *emotive* parts
            
        Returns:
            List of (text_chunk, is_emotive) tuples
        """
        if not self.config.emotive_voice_id:
            # No emotive voice configured, return all as regular text
            return [(text, False)]
        
        parts = []
        current_pos = 0
        
        # Find all *text* patterns
        for match in re.finditer(r'\*([^*]+)\*', text):
            # Add regular text before the emotive part
            if match.start() > current_pos:
                regular_text = text[current_pos:match.start()]
                if regular_text.strip():
                    parts.append((regular_text, False))
            
            # Add emotive text (content inside asterisks)
            emotive_text = match.group(1)
            if emotive_text.strip():
                parts.append((emotive_text, True))
            
            current_pos = match.end()
        
        # Add remaining regular text
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            if remaining_text.strip():
                parts.append((remaining_text, False))
        
        return parts if parts else [(text, False)]


    async def speak_stream_multi_voice(self, text_generator: AsyncGenerator[str, None]) -> bool:
        """
        Speak streaming text with multi-voice support for emotive expressions.
        Detects *emotive text* and uses different voice for those parts.
        If no emotive voice is configured, uses main voice for everything.
        
        Args:
            text_generator: Async generator yielding text chunks
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
            
        if self.is_streaming:
            await self.stop()
            
        # Initialize progress_monitor_task early to avoid UnboundLocalError
        progress_monitor_task = None
            
        try:
            self._stop_requested = False
            self._interrupted = False
            self._first_audio_received = False
            
            # Cancel any existing session and create a new one
            if self.current_session and not self.current_session.was_interrupted:
                print(f"⚠️ Cancelling previous session: {self.current_session.session_id}")
                self.current_session.was_interrupted = True
                
            # Close any existing voice connection from previous session
            if self._current_voice_connection:
                await self._close_current_voice_connection()
            
            # Create a new session for this speaking operation
            self.current_session = self._create_session()
            print(f"🆕 Created TTS session: {self.current_session.session_id}")
            self._websockets_completed = 0
            self._total_websockets = 0
            self._all_text_sent = False
            self._speech_monitor_task = None
            
            if self.track_spoken_content and self.current_session.whisper_tracker:
                self.current_session.whisper_tracker.start_tracking()
            
            # Clear any lingering audio in the shared stream before we start
            interrupt_playback()
            self.is_playing = False
            self.is_streaming = True
            
            text_buffer = ""
            xml_buffer = ""  # Buffer for incomplete XML tags
            in_xml_tag = False
            current_xml_start = -1
            
            async for chunk in text_generator:
                if self._stop_requested:
                    self._interrupted = True
                    break
                    
                # Add to generated text regardless
                if self.current_session:
                    self.current_session.generated_text += chunk
                    # Also update recovery text
                    self.last_session_generated_text = self.current_session.generated_text
                
                # Process each character to detect XML tags
                for i, char in enumerate(chunk):
                    if char == '<' and not in_xml_tag:
                        # Potential start of XML tag
                        in_xml_tag = True
                        current_xml_start = len(self.current_session.generated_text) - len(chunk) + i
                        xml_buffer = char
                        # Debug: Log when we start detecting XML
                        print(f"🔍 DEBUG: Started XML tag detection at position {current_xml_start}, text_buffer so far: {repr(text_buffer[-20:])}")
                    elif in_xml_tag:
                        xml_buffer += char
                        if char == '>':
                            # Check if this is a closing tag or complete tag
                            if xml_buffer.endswith('/>') or self._is_closing_tag(xml_buffer):
                                # Complete XML tag found
                                tag_end = len(self.current_session.generated_text) - len(chunk) + i + 1
                                # The position where the tool should execute is where the tag started
                                self._process_xml_tag(xml_buffer, len(self.current_session.spoken_text_for_tts), tag_end)
                                print(f"🔍 DEBUG: Complete XML tag: {repr(xml_buffer)}, text_buffer after: {repr(text_buffer[-20:])}")
                                in_xml_tag = False
                                xml_buffer = ""
                                current_xml_start = -1
                            else:
                                # Found '>' but tag not complete - might be inside tag content
                                # For example: <tag attr="value>something">
                                print(f"🔍 DEBUG: Found '>' but tag not complete: {repr(xml_buffer)}")
                    else:
                        # Regular text - add to buffer only if not in XML
                        text_buffer += char
                        if self.current_session:
                            self.current_session.spoken_text_for_tts += char
                
                # Process complete sentences/phrases for voice switching
                if not in_xml_tag:  # Only process when not inside XML
                    sentences = self._split_into_sentences(text_buffer)
                    
                    # Keep the last incomplete sentence in buffer
                    if sentences and not text_buffer.rstrip().endswith(('.', '!', '?')):
                        complete_sentences = sentences[:-1]
                        text_buffer = sentences[-1]
                    else:
                        complete_sentences = sentences
                        text_buffer = ""
                    
                    # Process each complete sentence with appropriate voice
                    for sentence in complete_sentences:
                        if sentence.strip():
                            # Debug: Check if sentence has both asterisks and angle brackets
                            if '*' in sentence and ('<' in sentence or '>' in sentence):
                                print(f"⚠️ DEBUG: Sentence with asterisks and XML: {repr(sentence)}")
                            
                            await self._speak_sentence_with_voice_switching(sentence)
                            
                            if self._stop_requested:
                                self._interrupted = True
                                break
                else:
                    # We're still in an XML tag at the end of this chunk
                    print(f"🔍 DEBUG: Chunk ended while in XML tag. xml_buffer: {repr(xml_buffer)}, text_buffer: {repr(text_buffer[-30:])}")
            
            # Process any remaining text in buffer
            if text_buffer.strip() and not self._stop_requested:
                await self._speak_sentence_with_voice_switching(text_buffer)
            
            # Check for incomplete XML tag at end of message
            if xml_buffer and in_xml_tag and not self._stop_requested:
                print(f"⚠️ Incomplete tool call at end of message: {xml_buffer}")
                print(f"⚠️ DEBUG: Final text_buffer: {repr(text_buffer)}")
                print(f"⚠️ DEBUG: Was in_xml_tag: {in_xml_tag}")
                # Check if it's a complete self-closing tag or has matching closing tag
                if self._is_closing_tag(xml_buffer):
                    tag_end = len(self.current_session.generated_text)
                    self._process_xml_tag(xml_buffer, len(self.current_session.spoken_text_for_tts), tag_end)
                    print(f"✅ Processed tool call at end of message")
            
            # Mark that all text has been sent
            self._all_text_sent = True
            print(f"📝 All text sent to TTS ({self._total_websockets} websockets created)")
            
            # Close current voice connection and send completion signal
            if self._current_voice_connection:
                await self._close_current_voice_connection()
            
            # Start monitoring speech progress for tool execution if we have tool calls
            if self.current_session.tool_calls and self.track_spoken_content:
                print(f"🔍 Starting speech progress monitoring for {len(self.current_session.tool_calls)} tool calls")
                self._speech_monitor_task = asyncio.create_task(self._monitor_speech_progress())
            
            # Wait for all audio to finish playing (simple polling approach)
            if not self._stop_requested:
                await self._wait_for_audio_completion()
            
            # Stop Whisper tracking
            if self.track_spoken_content and self.current_session and self.current_session.whisper_tracker:
                self.current_session.current_spoken_content = self.current_session.whisper_tracker.stop_tracking()
            
            # Check for any unexecuted tools after audio completes
            if self.current_session.tool_calls and not self._stop_requested:
                await self._execute_remaining_tools()
            
            return not self._interrupted
            
        except Exception as e:
            print(f"Multi-voice TTS error: {e}")
            self._interrupted = True
            # Stop Whisper tracking on error
            if self.track_spoken_content and self.current_session and self.current_session.whisper_tracker:
                try:
                    self.current_session.whisper_tracker.stop_tracking()
                except Exception:
                    pass
            return False
        finally:
            # Cancel progress monitor if running
            if hasattr(self, '_speech_monitor_task') and self._speech_monitor_task and not self._speech_monitor_task.done():
                self._speech_monitor_task.cancel()
                try:
                    await self._speech_monitor_task
                except asyncio.CancelledError:
                    pass
            if progress_monitor_task and not progress_monitor_task.done():
                progress_monitor_task.cancel()
                try:
                    await progress_monitor_task
                except asyncio.CancelledError:
                    pass
            
            self.is_streaming = False
            await self._cleanup_multi_voice()
            
        # Return success/failure based on interruption
        return not self._interrupted

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for voice processing."""
        # Simple sentence splitting - can be enhanced
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_closing_tag(self, xml_buffer: str) -> bool:
        """Check if the buffer contains a complete tag (either self-closing or with closing tag)."""
        # Check for self-closing tag
        if xml_buffer.endswith('/>'):
            return True
        # Check if we have a closing tag pattern
        if re.match(r'</\w+>', xml_buffer):
            return True
        # Check if we have opening and closing tags
        tag_match = re.match(r'<(\w+)[^>]*>', xml_buffer)
        if tag_match:
            tag_name = tag_match.group(1)
            return f'</{tag_name}>' in xml_buffer
        return False
    
    def _process_xml_tag(self, xml_content: str, start_pos: int, end_pos: int):
        """Process a complete XML tag and add it to tool calls."""
        if not self.current_session:
            return
            
        # Extract tag name
        tag_match = re.match(r'<(\w+)[^>]*>', xml_content)
        if tag_match:
            tag_name = tag_match.group(1)
            
            # Create tool call
            tool_call = ToolCall(
                tag_name=tag_name,
                content=xml_content,
                start_position=start_pos,
                end_position=end_pos
            )
            
            self.current_session.tool_calls.append(tool_call)
            print(f"🔧 Found tool call: {tag_name} at TTS position {start_pos}")
            print(f"   Current spoken_text_for_tts length: {len(self.current_session.spoken_text_for_tts)}")
            print(f"   Current generated_text length: {len(self.current_session.generated_text)}")
    
    async def _monitor_speech_progress(self):
        """Monitor speech progress using Whisper and execute tools when appropriate."""
        if not self.current_session or not self.current_session.whisper_tracker:
            return
            
        while self.is_streaming and not self._stop_requested:
            await asyncio.sleep(0.5)  # Check every 500ms
            
            # Get current spoken text and TTS text
            spoken_text = self.current_session.whisper_tracker.get_spoken_text()
            tts_text = self.current_session.spoken_text_for_tts
            
            # Use fuzzy matching to find actual position
            fuzzy_position = self._fuzzy_find_position(spoken_text, tts_text)
            
            # Check which tools should be executed based on fuzzy position
            for tool_call in self.current_session.tool_calls:
                if not tool_call.executed:
                    if fuzzy_position >= tool_call.start_position:
                        # Execute the tool
                        if tool_call.start_position <= 0:
                            progress = 100.0
                        else:
                            progress = (fuzzy_position / tool_call.start_position) * 100
                            if progress > 100.0:
                                progress = 100.0
                        print(f"📊 Tool '{tool_call.tag_name}' reached! Progress: {progress:.1f}%")
                        await self._execute_tool(tool_call)
                        tool_call.executed = True
    
    async def _execute_tool(self, tool_call: ToolCall):
        """Execute a tool call and handle the result."""
        print(f"🚀 Executing tool: {tool_call.tag_name} at spoken position {tool_call.start_position}")
        
        # Call the registered callback if available
        if self.on_tool_execution:
            try:
                result = await self.on_tool_execution(tool_call)
                
                if result.should_interrupt:
                    print(f"🛑 Tool requested interruption")
                    # Stop current TTS playback
                    self._stop_requested = True
                    self.current_session.was_interrupted = True
                    
                    # If there's content to speak, queue it up
                    if result.content:
                        print(f"📢 Tool returned content to speak: {result.content[:50]}...")
                        # Store the content for the conversation handler to process
                        # This will be handled by the conversation manager
                        
            except Exception as e:
                print(f"❌ Tool execution error: {e}")
    
    async def _execute_remaining_tools(self):
        """Execute any remaining unexecuted tools after audio completes."""
        unexecuted_tools = [tc for tc in self.current_session.tool_calls if not tc.executed]
        
        if unexecuted_tools:
            print(f"🔧 Checking {len(unexecuted_tools)} remaining tool(s) after audio completion")
            
            # Check if the message was interrupted
            was_interrupted = self._interrupted or self.current_session.was_interrupted
            
            if not was_interrupted:
                # Message completed naturally - execute ALL remaining tools
                print(f"✅ Message completed naturally - executing all remaining tools")
                for tool_call in unexecuted_tools:
                    print(f"🚀 Executing tool '{tool_call.tag_name}' at position {tool_call.start_position}")
                    await self._execute_tool(tool_call)
                    tool_call.executed = True
            else:
                # Message was interrupted - use fuzzy matching to determine which tools to execute
                print(f"⚠️ Message was interrupted - checking tool progress")
                
                # Get final spoken text
                final_spoken_text = ""
                if self.current_session.current_spoken_content:
                    final_spoken_text = " ".join(content.text for content in self.current_session.current_spoken_content)
                elif self.current_session.whisper_tracker:
                    final_spoken_text = self.current_session.whisper_tracker.get_spoken_text()
                
                tts_text = self.current_session.spoken_text_for_tts
                final_fuzzy_position = self._fuzzy_find_position(final_spoken_text, tts_text)
                
                # Execute remaining tools based on progress
                for tool_call in unexecuted_tools:
                    if not tool_call.executed:
                        if tool_call.start_position <= 0:
                            progress = 100.0
                        else:
                            progress = (final_fuzzy_position / tool_call.start_position) * 100
                            if progress > 100.0:
                                progress = 100.0
                        
                        # Execute tools that were close to being reached
                        # Use a more forgiving threshold (80%) for tools near the end
                        threshold = 80 if tool_call.start_position > len(tts_text) * 0.8 else 85
                        
                        if progress >= threshold:
                            print(f"   ✅ Executing interrupted tool at {progress:.1f}% progress")
                            await self._execute_tool(tool_call)
                            tool_call.executed = True
                        else:
                            print(f"   ❌ Skipping tool at {progress:.1f}% progress (threshold: {threshold}%)")

    async def _speak_sentence_with_voice_switching(self, sentence: str):
        """Speak a sentence with appropriate voice switching for emotive parts."""
        parts = self._parse_emotive_text(sentence)
        
        # Group consecutive parts by voice to maintain prosodic continuity
        voice_groups = []
        current_group = {"voice_id": None, "voice_settings": None, "parts": []}
        
        for text_part, is_emotive in parts:
            if text_part.strip():
                voice_id = self.config.emotive_voice_id if is_emotive else self.config.voice_id
                voice_settings = self._get_voice_settings(is_emotive)
                
                # Check if this part uses the same voice as current group
                if (current_group["voice_id"] == voice_id and 
                    current_group["voice_settings"] == voice_settings):
                    # Add to current group
                    current_group["parts"].append(text_part)
                else:
                    # Start new group, but first save the current one
                    if current_group["parts"]:
                        voice_groups.append(current_group)
                    
                    current_group = {
                        "voice_id": voice_id,
                        "voice_settings": voice_settings,
                        "parts": [text_part]
                    }
        
        # Add the final group
        if current_group["parts"]:
            voice_groups.append(current_group)
        
        # Process each voice group
        for group in voice_groups:
            if self._stop_requested:
                break
                
            # Get voice key for this group
            voice_key = self._get_voice_key(group["voice_id"], group["voice_settings"])
            
            # Check if we need to switch voices (close current and create new)
            if self._current_voice_key and self._current_voice_key != voice_key:
                # Voice change detected - close current connection
                print(f"🔄 Voice change detected: {self._current_voice_key} → {voice_key}")
                await self._close_current_voice_connection()
            
            # Track websockets created
            if self._current_voice_key != voice_key:
                self._total_websockets += 1
            
            # Combine all parts for this voice into one text
            combined_text = "".join(group["parts"])
            
            # Speak using current or new connection
            await self._speak_text_part(combined_text, group["voice_id"], group["voice_settings"])

    def _get_voice_settings(self, is_emotive: bool) -> Dict[str, float]:
        """Get voice settings for regular or emotive speech."""
        if is_emotive:
            return {
                "speed": self.config.emotive_speed,
                "stability": self.config.emotive_stability,
                "similarity_boost": self.config.emotive_similarity_boost
            }
        else:
            return {
                "speed": self.config.speed,
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost
            }

    def _get_voice_key(self, voice_id: str, voice_settings: Dict[str, float]) -> str:
        """Generate a unique key for voice and settings combination."""
        settings_str = "_".join(f"{k}:{v}" for k, v in sorted(voice_settings.items()))
        return f"{voice_id}_{settings_str}"
    
    async def _get_or_create_voice_connection(self, voice_id: str, voice_settings: Dict[str, float]):
        """Get current voice connection or create a new one."""
        voice_key = self._get_voice_key(voice_id, voice_settings)
        
        # Check if this is the current voice
        if self._current_voice_key == voice_key and self._current_voice_connection:
            return self._current_voice_connection
        
        # If we have a different voice active, it should have been closed already
        # Create new connection
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        params = f"?model_id={self.config.model_id}&output_format={self.config.output_format}"
        
        websocket = await websockets.connect(uri + params)
        
        # Send initial configuration
        initial_message = {
            "text": " ",
            "voice_settings": voice_settings,
            "xi_api_key": self.config.api_key
        }
        await websocket.send(json.dumps(initial_message))
        
        # Store as current connection
        self._current_voice_connection = websocket
        self._current_voice_key = voice_key
        
        # Start task to handle audio responses
        self._current_voice_task = asyncio.create_task(self._handle_websocket_audio(websocket))
        
        print(f"🔗 Created voice connection: {voice_key}")
        return websocket
    
    async def _close_current_voice_connection(self):
        """Close the current voice connection after sending completion signal."""
        if not self._current_voice_connection:
            return
            
        try:
            # Send completion signal to trigger audio generation
            await self._current_voice_connection.send(json.dumps({"text": ""}))
            print(f"🔚 Sent completion signal for voice: {self._current_voice_key}")
            
            # IMPORTANT: Wait for audio task to complete BEFORE closing the connection
            # This ensures all audio is received and played
            if self._current_voice_task:
                try:
                    await self._current_voice_task
                except Exception as e:
                    print(f"Error in audio task: {e}")
            
            # Now safe to close the connection
            await self._current_voice_connection.close()
            print(f"🔗 Closed voice connection: {self._current_voice_key} ({self._websockets_completed}/{self._total_websockets})")
        except Exception as e:
            print(f"Error closing voice connection {self._current_voice_key}: {e}")
        
        # Clear current connection
        self._current_voice_connection = None
        self._current_voice_key = None
        self._current_voice_task = None

    async def _force_close_voice_connection(self):
        """Force-close the current voice connection without waiting for trailing audio."""
        if self._current_voice_task and not self._current_voice_task.done():
            self._current_voice_task.cancel()
            try:
                await self._current_voice_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Error while cancelling voice audio task: {e}")
        
        if self._current_voice_connection:
            try:
                await self._current_voice_connection.close()
            except Exception as e:
                print(f"Error forcing voice connection close: {e}")
        
        self._current_voice_connection = None
        self._current_voice_key = None
        self._current_voice_task = None

    async def _speak_text_part(self, text: str, voice_id: str, voice_settings: Dict[str, float]):
        """Speak a single text part with specified voice and settings."""
        try:
            # Check if we should stop before starting
            if self._stop_requested:
                return
                
            # Get or create persistent connection for this voice
            websocket = await self._get_or_create_voice_connection(voice_id, voice_settings)
            
            # Check interruption before sending text
            if self._stop_requested:
                return
                
            # Send the text to existing connection
            text_message = {
                "text": text,
                "try_trigger_generation": True
            }
            await websocket.send(json.dumps(text_message))
            print(f"📤 Sent to persistent TTS connection: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
        except Exception as e:
            if not self._stop_requested:  # Don't log errors if we were interrupted
                print(f"Error speaking text part '{text[:50]}...': {e}")

    async def _handle_websocket_audio(self, websocket):
        """Handle audio responses from a websocket connection."""
        try:
            async for message in websocket:
                if self._stop_requested:
                    break
                    
                data = json.loads(message)
                
                if data.get("audio") and not self._stop_requested:
                    # Decode and play audio only if not stopped
                    audio_data = base64.b64decode(data["audio"])
                    if not self._stop_requested:
                        if not self._first_audio_received and self.first_audio_callback:
                            self._first_audio_received = True
                            asyncio.create_task(self.first_audio_callback())

                        self._play_audio_chunk(audio_data)
                    
                    # Update last audio time for completion tracking
                    self._last_audio_time = time.time()
                
                elif data.get("isFinal"):
                    # This websocket connection has finished
                    self._websockets_completed += 1
                    print(f"🔊 Websocket completed ({self._websockets_completed}/{self._total_websockets})")
                    
                    # Just track completion, no monitoring task needed
                    
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"WebSocket audio handling error: {e}")


    async def _wait_for_audio_completion(self):
        """Wait for audio playback to complete in multi-voice mode."""
        try:
            print(f"🔊 Waiting for audio completion...")
            
            # First, wait for all websockets to finish generating chunks OR stop request
            while (self._websockets_completed < self._total_websockets and 
                   not self._stop_requested):
                await asyncio.sleep(0.1)
            
            if self._stop_requested:
                print("🛑 Stopped during chunk generation")
                return
            
            print(f"✅ All audio chunks generated ({self._websockets_completed} websockets)")
            
            # Then wait for buffered playback to finish
            await self._wait_for_playback_drain()
            
            if self._stop_requested:
                print("🛑 Stopped during playback")
                return
                
            print("✅ Audio playback complete")
                
        except Exception as e:
            print(f"Audio completion wait error: {e}")

    async def _cleanup_multi_voice(self):
        """Clean up multi-voice streaming resources."""
        try:
            # Use the existing cleanup method for consistency
            await self._cleanup_stream()
                
        except Exception as e:
            print(f"Multi-voice cleanup error: {e}")

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
            self._first_audio_received = False
            self._chunks_played = 0  # Reset chunks played counter
            
            # Create a new session for this speaking operation
            self.current_session = self._create_session()
            
            if self.current_session.whisper_tracker and self.track_spoken_content:
                self.current_session.whisper_tracker.start_tracking()
            
            await self._start_streaming_generator(text_generator)
            
            # If we reached here without stop being requested, it completed successfully
            completed_successfully = not self._stop_requested
            self._interrupted = self._stop_requested  # Set interrupted flag based on actual completion
            
            return completed_successfully
            
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

        # Interrupt hardware playback right away
        interrupt_playback()
        
        # Mark current session as interrupted
        if self.current_session:
            self.current_session.was_interrupted = True
            self.current_session.end_time = time.time()

        # Ensure we force-close any multi-voice websocket connection
        await self._force_close_voice_connection()

        # Close primary websocket promptly to stop further streaming
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                print(f"Error closing TTS WebSocket: {e}")
            self.websocket = None

        # Cancel audio response task if still running
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
        self.audio_task = None

        self.is_playing = False

        self.is_streaming = False
        print("✅ TTS stopped")
    
    def is_currently_playing(self) -> bool:
        """Check if TTS is currently playing (both streaming and audio output)."""
        if self._stop_requested or not self.is_streaming or not self.is_playing:
            return False

        try:
            stream = ensure_stream_started()
            return stream.get_buffered_duration() > 0.0
        except Exception:
            return False
    
    def has_played_audio(self) -> bool:
        """Check if any audio chunks have been played in the current session."""
        return self._chunks_played > 0
    
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
            print(f"📤 Sent to TTS: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
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
            # Accumulate for Whisper tracking
            if self.current_session:
                self.current_session.generated_text += text_chunk
            
            # Send chunks at natural breaks
            if any(punct in text_buffer for punct in ['.', '!', '?', ',', ';']) or len(text_buffer) > 40:
                if not self._stop_requested:
                    message = {
                        "text": text_buffer,
                        "try_trigger_generation": True
                    }
                    await self.websocket.send(json.dumps(message))
                    print(f"📤 Sent to TTS stream: '{text_buffer.strip()[:50]}{'...' if len(text_buffer.strip()) > 50 else ''}'")
                    text_buffer = ""
        
        # Send remaining text
        if text_buffer.strip() and not self._stop_requested:
            message = {
                "text": text_buffer,
                "try_trigger_generation": True
            }
            await self.websocket.send(json.dumps(message))
            print(f"📤 Sent final TTS chunk: '{text_buffer.strip()[:50]}{'...' if len(text_buffer.strip()) > 50 else ''}'")
        
        # Signal completion
        if not self._stop_requested:
            await self.websocket.send(json.dumps({"text": ""}))
        
        # Wait for completion
        await self._wait_for_completion()
    
    async def _setup_websocket(self):
        """Setup WebSocket connection and audio playback."""
        # Clear any leftover audio
        interrupt_playback()
        self.is_playing = False
        
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
                        if len(audio_data) > 0 and not self._stop_requested:
                            # Call first audio callback if not already called
                            if not self._first_audio_received and self.first_audio_callback:
                                self._first_audio_received = True
                                asyncio.create_task(self.first_audio_callback())

                            self._play_audio_chunk(audio_data)
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

    def _play_audio_chunk(self, audio_chunk: bytes):
        """Write an audio chunk to the shared output and feed Whisper tracking."""
        if not audio_chunk or self._stop_requested:
            return

        session = self.current_session

        try:
            write_playback_pcm(audio_chunk, self.config.sample_rate)
            self._chunks_played += 1
            self.is_playing = True
        except Exception as e:
            if not self._stop_requested:
                print(f"Audio playback error: {e}")
            return

        if (
            self.track_spoken_content
            and session
            and session.whisper_tracker
            and session == self.current_session
        ):
            try:
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                if audio_np.size:
                    target_rate = 16000
                    num_samples = int(len(audio_np) * target_rate / self.config.sample_rate)
                    if num_samples > 0:
                        audio_16k = signal.resample(audio_np, num_samples)
                        session.whisper_tracker.add_audio_chunk(audio_16k.astype(np.float32))
            except Exception as e:
                print(f"Whisper audio capture error: {e}")

    async def _wait_for_playback_drain(self):
        """Wait until the shared stream output buffer is effectively empty."""
        if self._stop_requested or not self.is_playing:
            self.is_playing = False
            return

        try:
            stream = ensure_stream_started()
        except Exception:
            self.is_playing = False
            return

        idle_checks = 0
        start_time = time.time()
        max_wait = 3.0  # seconds

        while not self._stop_requested:
            try:
                buffered = stream.get_buffered_duration()
            except Exception:
                break

            if buffered <= 0.02:  # ~20ms remaining
                idle_checks += 1
                if idle_checks >= 3:
                    break
            else:
                idle_checks = 0

            if (time.time() - start_time) > max_wait:
                break

            await asyncio.sleep(0.05)

        self.is_playing = False
    
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
        
        # Wait for remaining buffered audio to play out
        if not self._stop_requested:
            await self._wait_for_playback_drain()
    
    
    async def _cleanup_stream(self):
        """Clean up streaming resources."""
        self.is_streaming = False

        await self._force_close_voice_connection()
        
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
        
        self.is_playing = False
        
        # Stop Whisper tracking and collect spoken content
        if self.current_session and self.current_session.whisper_tracker and self.track_spoken_content:
            try:
                self.current_session.current_spoken_content = self.current_session.whisper_tracker.stop_tracking()
                print(f"🎙️ Captured {len(self.current_session.current_spoken_content)} spoken segments")
            except Exception as e:
                print(f"Error stopping Whisper tracking: {e}")
    
    def get_spoken_content(self) -> List[SpokenContent]:
        """
        Get the actual spoken content from the last TTS session.
        
        Returns:
            List of SpokenContent objects representing what was actually spoken
        """
        if not self.current_session:
            return []
        return self.current_session.current_spoken_content.copy()
    
    def get_spoken_text(self) -> str:
        """
        Get the actual spoken text as a single string.
        
        Returns:
            Combined text of what was actually spoken
        """
        if not self.current_session:
            return ""
        return " ".join(content.text for content in self.current_session.current_spoken_content)
    
    def get_spoken_text_heuristic(self) -> str:
        """
        Get spoken text using character-count heuristic from original generated text.
        Uses Whisper's character count but preserves exact LLM vocabulary/formatting.
        Only applies heuristic when there was an interruption - otherwise returns full text.
        
        Returns:
            Portion of original generated text that was likely spoken
        """
        if not self.current_session or not self.current_session.generated_text:
            return ""
        
        # If TTS was interrupted, use Whisper character count heuristic
        if self._interrupted and self.current_session.current_spoken_content:
            # Get total character count from Whisper
            whisper_char_count = sum(len(content.text) for content in self.current_session.current_spoken_content)
            
            # Debug logging
            whisper_texts = [content.text for content in self.current_session.current_spoken_content]
            print(f"🔍 [HEURISTIC DEBUG] Whisper segments: {len(whisper_texts)}")
            print(f"🔍 [HEURISTIC DEBUG] Whisper texts: {whisper_texts}")
            print(f"🔍 [HEURISTIC DEBUG] Whisper char count: {whisper_char_count}")
            print(f"🔍 [HEURISTIC DEBUG] Generated text length: {len(self.current_session.generated_text)}")
            print(f"🔍 [HEURISTIC DEBUG] Generated text: '{self.current_session.generated_text[:100]}...'")
            
            if whisper_char_count == 0:
                print("🔍 [HEURISTIC DEBUG] Returning empty - no Whisper characters")
                return ""
            
            # Use spoken_text_for_tts as the reference since that's what was actually sent to TTS
            reference_text = self.current_session.spoken_text_for_tts or self.current_session.generated_text
            
            # Find this position in the reference text
            target_position = min(whisper_char_count, len(reference_text))
            
            # Round up to nearest complete word boundary
            if target_position >= len(self.current_session.generated_text):
                # If Whisper captured more than generated (shouldn't happen), return full text
                return self.current_session.generated_text
            
            # Find the end of the word at target position
            word_end_position = target_position
            
            # If we're in the middle of a word, find the end
            while (word_end_position < len(self.current_session.generated_text) and 
                   self.current_session.generated_text[word_end_position] not in [' ', '.', ',', '!', '?', ';', ':', '\n']):
                word_end_position += 1
            
            # Check if we're in the middle of an emotive marker (between asterisks)
            result_text = self.current_session.generated_text[:word_end_position]
            asterisk_count = result_text.count('*')
            
            # If odd number of asterisks, we're in the middle of an emotive marker
            if asterisk_count % 2 == 1:
                # Find the closing asterisk
                close_pos = self.current_session.generated_text.find('*', word_end_position)
                if close_pos != -1:
                    word_end_position = close_pos + 1
            
            result = self.current_session.generated_text[:word_end_position].strip()
            print(f"🔍 [HEURISTIC DEBUG] Returning heuristic result: '{result}' ({len(result)} chars)")
            return result
        
        else:
            # TTS completed successfully - return full generated text
            return self.current_session.generated_text
    
    def get_generated_vs_spoken(self) -> Dict[str, str]:
        """
        Compare generated text vs actually spoken text.
        
        Returns:
            Dictionary with 'generated', 'spoken_whisper', and 'spoken_heuristic' keys
        """
        if not self.current_session:
            return {"generated": "", "spoken_whisper": "", "spoken_heuristic": ""}
            
        return {
            "generated": self.current_session.generated_text,
            "spoken_whisper": self.get_spoken_text(),
            "spoken_heuristic": self.get_spoken_text_heuristic()
        }
    
    def was_fully_spoken(self) -> bool:
        """
        Check if the generated text was fully spoken (not interrupted).
        Uses heuristic approach based on character count.
        
        Returns:
            True if likely fully spoken, False if interrupted
        """
        if not self.current_session:
            return False
            
        spoken_heuristic = self.get_spoken_text_heuristic()
        if not self.current_session.generated_text or not spoken_heuristic:
            return False
        
        # If heuristic captured at least 90% of generated text, consider it fully spoken
        return len(spoken_heuristic) >= (len(self.current_session.generated_text) * 0.9)

    async def cleanup(self):
        """Clean up all resources."""
        await self.stop()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
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
