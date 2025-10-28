#!/usr/bin/env python3
"""
Async TTS Module for ElevenLabs WebSocket Streaming
Provides interruptible text-to-speech with real-time audio playback.
Now uses ElevenLabs alignment data to track spoken content.
"""

import asyncio
import websockets
import json
import base64
import time
import re
import difflib
import bisect
import threading
import queue
from typing import Optional, AsyncGenerator, Dict, Any, List, Tuple
from dataclasses import dataclass, field

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

@dataclass
class SpokenContent:
    """Represents content that was actually spoken by TTS."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class AlignmentChar:
    """Represents a single character alignment with its timing."""
    char: str
    start_time: float  # Seconds since audio playback start
    end_time: float    # Seconds since audio playback start
    global_index: int

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
    current_spoken_content: List[SpokenContent] = field(default_factory=list)  # Alignment-derived spoken segments
    was_interrupted: bool = False  # Whether this session was interrupted
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tool_calls: List[ToolCall] = field(default_factory=list)  # Tools to execute during speech
    spoken_text_for_tts: str = ""  # Text actually sent to TTS (excludes tool content)
    audio_start_time: Optional[float] = None  # When audio playback began
    total_audio_duration: float = 0.0  # Total seconds of audio that have been written to output
    alignment_chars: List[AlignmentChar] = field(default_factory=list)  # Flattened alignment characters with timing
    raw_alignment_chunks: List[Dict[str, Any]] = field(default_factory=list)  # Raw alignment payloads for debugging
    alignment_char_count: int = 0  # Number of alignment characters processed
    alignment_text: str = ""  # Concatenated alignment characters
    alignment_start_times: List[float] = field(default_factory=list)  # Start times for binary search

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

        # Background playback worker
        self._playback_queue = None  # type: Optional["queue.Queue[Tuple[bytes, Optional[Dict[str, Any]]]]"]
        self._playback_thread: Optional[threading.Thread] = None
        self._playback_stop_event: Optional[threading.Event] = None
        self._last_playback_queue_log = 0.0
        self._session_lock = threading.RLock()
        
    def _fuzzy_find_position(self, source_text: str, target_text: str) -> int:
        """
        Find the approximate position in target_text that corresponds to the end of source_text.
        Uses fuzzy matching to tolerate differences in normalization and punctuation.
        
        Returns:
            Character position in target_text that best matches the end of source_text
        """
        if not source_text or not target_text:
            return 0
            
        # Normalize texts for comparison
        def normalize(text):
            # Convert to lowercase
            text = text.lower()
            # Keep common punctuation and numbers for better alignment fidelity
            text = re.sub(r'[^a-z0-9\s.,;:!?-]', ' ', text)
            # Collapse multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Normalize and split into words for more robust matching
        norm_source = normalize(source_text)
        norm_target = normalize(target_text)
        
        source_words = norm_source.split()
        target_words = norm_target.split()
        
        if not source_words:
            return 0
            
        # Try to find the best match using a sliding window approach
        best_match_score = 0
        best_match_end = 0
        
        # Look for the last few words of the source in the target (more reliable than full text)
        window_size = min(5, len(source_words))  # Use last 5 words or less
        search_words = source_words[-window_size:]
        
        # Slide through TTS text looking for best match
        for i in range(len(target_words) - window_size + 1):
            window = target_words[i:i + window_size]
            
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
            for i, char in enumerate(target_text):
                if char.isspace() and i > 0 and not target_text[i-1].isspace():
                    word_count += 1
                    if word_count >= best_match_end:
                        return i
            
            # If we counted all words, return end of text
            return len(target_text)
        
        # Fallback: Try character-level fuzzy matching on normalized text
        if norm_source in norm_target and norm_target:
            # Find the position in normalized text
            norm_pos = norm_target.find(norm_source) + len(norm_source)
            # Estimate position in original text
            ratio = norm_pos / len(norm_target)
            result = int(len(target_text) * ratio)
            return result
        
        # Final fallback: percentage-based estimation with safety margin
        # Provide slight buffer to account for normalization differences
        source_ratio = len(source_text) / max(len(target_text), 1)
        estimated_pos = int(len(target_text) * min(source_ratio * 1.1, 1.0))
        
        return estimated_pos
    
    def _create_session(self) -> TTSSession:
        """Create a new TTS session with unique ID and fresh alignment tracking."""
        self._session_counter += 1
        session_id = f"tts_session_{int(time.time())}_{self._session_counter}"
        session = TTSSession(session_id=session_id)
        
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
            session = self._create_session()
            with self._session_lock:
                self.current_session = session
                session.generated_text = text
                session.spoken_text_for_tts = text
            # Store for recovery in case of error
            self.last_session_generated_text = text
            
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
            print("here here")
            # Cancel any existing session and create a new one
            previous_session = None
            cancelled_previous = False
            with self._session_lock:
                previous_session = self.current_session
                if previous_session and not previous_session.was_interrupted:
                    previous_session.was_interrupted = True
                    cancelled_previous = True
                session = self._create_session()
                self.current_session = session
            if cancelled_previous and previous_session:
                print(f"⚠️ Cancelling previous session: {previous_session.session_id}")
                
            print("here here f")
            # Close any existing voice connection from previous session
            if self._current_voice_connection:
                await self._close_current_voice_connection()
            
            print("here here f ffff")
            # Create a new session for this speaking operation
            print(f"🆕 Created TTS session: {session.session_id}")
            self._websockets_completed = 0
            self._total_websockets = 0
            self._all_text_sent = False
            self._speech_monitor_task = None
            
            # Clear any lingering audio in the shared stream before we start
            interrupt_playback()
            self.is_playing = False
            self.is_streaming = True
            print("here here f ffff aaaaaa")
            
            text_buffer = ""
            xml_buffer = ""  # Buffer for incomplete XML tags
            in_xml_tag = False
            current_xml_start = -1
            
            print("here here f ffff aaaffffffffaaa")
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
            
            print("here here bbbbbf ffff aaaffffffffaaa")
            # Close current voice connection and send completion signal
            #if self._current_voice_connection:
            #    await self._close_current_voice_connection()
            
            print("4here here bbbbbf ffff aaaffffffffaaa")
            # Start monitoring speech progress for tool execution if we have tool calls
            if self.current_session.tool_calls:
                print(f"🔍 Starting speech progress monitoring for {len(self.current_session.tool_calls)} tool calls")
                self._speech_monitor_task = asyncio.create_task(self._monitor_speech_progress())
            
            # Wait for all audio to finish playing (simple polling approach)
            if not self._stop_requested:
                await self._wait_for_audio_completion()
            
            # Check for any unexecuted tools after audio completes
            if self.current_session.tool_calls and not self._stop_requested:
                await self._execute_remaining_tools()
            
            return not self._interrupted
            
        except Exception as e:
            print(f"Multi-voice TTS error: {e}")
            self._interrupted = True
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
        with self._session_lock:
            session = self.current_session
            if not session:
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
                
                session.tool_calls.append(tool_call)
                spoken_len = len(session.spoken_text_for_tts)
                generated_len = len(session.generated_text)
        
        if 'tool_call' in locals():
            print(f"🔧 Found tool call: {tool_call.tag_name} at TTS position {start_pos}")
            print(f"   Current spoken_text_for_tts length: {spoken_len}")
            print(f"   Current generated_text length: {generated_len}")
    
    async def _monitor_speech_progress(self):
        """Monitor speech progress using alignment timings and execute tools when appropriate."""
        if not self.current_session:
            return
            
        while self.is_streaming and not self._stop_requested:
            await asyncio.sleep(0.5)  # Check every 500ms

            with self._session_lock:
                session = self.current_session
                if not session or session.audio_start_time is None or not session.alignment_chars:
                    continue
                audio_start_time = session.audio_start_time
                tts_text = session.spoken_text_for_tts

            elapsed = time.time() - audio_start_time
            spoken_alignment_chars = self._alignment_chars_spoken(elapsed, session)
            fuzzy_position = self._alignment_chars_to_tts_index(spoken_alignment_chars, session)

            # Map spoken alignment progress onto the TTS text
            if not tts_text:
                continue

            pending_tools = []
            with self._session_lock:
                current_session = self.current_session
                if session is not current_session:
                    continue
                for tool_call in current_session.tool_calls:
                    if tool_call.executed:
                        continue
                    if fuzzy_position >= tool_call.start_position:
                        if tool_call.start_position <= 0:
                            progress = 100.0
                        else:
                            progress = (fuzzy_position / tool_call.start_position) * 100
                            if progress > 100.0:
                                progress = 100.0
                        tool_call.executed = True
                        pending_tools.append((tool_call, progress))

            for tool_call, progress in pending_tools:
                print(f"📊 Tool '{tool_call.tag_name}' reached! Progress: {progress:.1f}%")
                await self._execute_tool(tool_call)
    
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
        with self._session_lock:
            session = self.current_session
            if not session:
                return
            unexecuted_tools = [tc for tc in session.tool_calls if not tc.executed]
            was_interrupted = self._interrupted or session.was_interrupted
        
        if not unexecuted_tools:
            return
        
        print(f"🔧 Checking {len(unexecuted_tools)} remaining tool(s) after audio completion")
        
        if not was_interrupted:
            # Message completed naturally - execute ALL remaining tools
            print(f"✅ Message completed naturally - executing all remaining tools")
            for tool_call in unexecuted_tools:
                print(f"🚀 Executing tool '{tool_call.tag_name}' at position {tool_call.start_position}")
                await self._execute_tool(tool_call)
                with self._session_lock:
                    tool_call.executed = True
            return
        
        # Message was interrupted - use fuzzy matching to determine which tools to execute
        print(f"⚠️ Message was interrupted - checking tool progress")
        with self._session_lock:
            session = self.current_session
            if not session:
                return
            tts_text = session.spoken_text_for_tts
            audio_start_time = session.audio_start_time
            session_end_time = session.end_time
            total_audio_duration = session.total_audio_duration
        
        if not tts_text:
            final_fuzzy_position = 0
        else:
            if audio_start_time is not None:
                if session_end_time is not None:
                    elapsed = session_end_time - audio_start_time
                else:
                    elapsed = time.time() - audio_start_time
            else:
                elapsed = 0.0
            elapsed = max(0.0, min(elapsed, total_audio_duration))
            spoken_alignment_chars = self._alignment_chars_spoken(elapsed)
            final_fuzzy_position = self._alignment_chars_to_tts_index(spoken_alignment_chars)
        
        # Execute remaining tools based on progress
        for tool_call in unexecuted_tools:
            with self._session_lock:
                if tool_call.executed:
                    continue
            if tool_call.start_position <= 0:
                progress = 100.0
            else:
                progress = (final_fuzzy_position / tool_call.start_position) * 100 if tool_call.start_position else 100.0
                if progress > 100.0:
                    progress = 100.0
            
            # Execute tools that were close to being reached
            # Use a more forgiving threshold (80%) for tools near the end
            threshold = 80 if tool_call.start_position > len(tts_text) * 0.8 else 85
            
            if progress >= threshold:
                print(f"   ✅ Executing interrupted tool at {progress:.1f}% progress")
                await self._execute_tool(tool_call)
                with self._session_lock:
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

                        alignment_payload = self._extract_alignment_payload(data)
                        self._play_audio_chunk(audio_data, alignment_payload)
                    
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
            session = self._create_session()
            with self._session_lock:
                self.current_session = session
            
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
        self._stop_playback_worker()
        
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
            # Accumulate for alignment progress tracking
            if self.current_session:
                self.current_session.generated_text += text_chunk
            
            # Send chunks at natural breaks
            if any(punct in text_buffer for punct in ['.', '!', '?', ',', ';']) or len(text_buffer) > 40:
                if not self._stop_requested:
                    chunk_to_send = text_buffer
                    message = {
                        "text": chunk_to_send,
                        "try_trigger_generation": True
                    }
                    await self.websocket.send(json.dumps(message))
                    if self.current_session:
                        self.current_session.spoken_text_for_tts += chunk_to_send
                    print(f"📤 Sent to TTS stream: '{chunk_to_send.strip()[:50]}{'...' if len(chunk_to_send.strip()) > 50 else ''}'")
                    text_buffer = ""
        
        # Send remaining text
        if text_buffer.strip() and not self._stop_requested:
            chunk_to_send = text_buffer
            message = {
                "text": chunk_to_send,
                "try_trigger_generation": True
            }
            await self.websocket.send(json.dumps(message))
            if self.current_session:
                self.current_session.spoken_text_for_tts += chunk_to_send
            print(f"📤 Sent final TTS chunk: '{chunk_to_send.strip()[:50]}{'...' if len(chunk_to_send.strip()) > 50 else ''}'")
        
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
        print("SETUP WEBSOCKET HHH")
        # Start audio response handler
        self.audio_task = asyncio.create_task(self._handle_audio_responses())
    
    async def _handle_audio_responses(self):
        """Handle incoming audio responses from ElevenLabs."""
        try:
            async for message in self.websocket:
                print("got message")
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

                            alignment_payload = self._extract_alignment_payload(data)
                            self._play_audio_chunk(audio_data, alignment_payload)
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

    def _ensure_playback_worker(self) -> bool:
        """Ensure a background thread exists to process playback chunks."""
        if self._playback_thread and self._playback_thread.is_alive():
            return True

        self._playback_queue = queue.Queue()
        self._playback_stop_event = threading.Event()

        def _worker():
            while True:
                if self._playback_stop_event and self._playback_stop_event.is_set():
                    break

                if self._stop_requested and self._playback_queue.empty():
                    break

                try:
                    item = self._playback_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:
                    break

                audio_chunk, alignment = item

                if self._playback_stop_event and self._playback_stop_event.is_set():
                    break

                try:
                    self._process_audio_chunk(audio_chunk, alignment)
                except Exception as playback_error:
                    print(f"Audio playback worker error: {playback_error}")
                    if self._stop_requested:
                        break

        self._playback_thread = threading.Thread(
            target=_worker,
            name="TTSPlaybackWorker",
            daemon=True,
        )
        self._playback_thread.start()
        return True

    def _stop_playback_worker(self):
        """Stop the background playback worker thread."""
        thread = self._playback_thread
        if not thread:
            return

        if self._playback_stop_event and not self._playback_stop_event.is_set():
            self._playback_stop_event.set()

        if self._playback_queue:
            try:
                self._playback_queue.put_nowait(None)
            except Exception:
                pass

        if thread.is_alive():
            thread.join(timeout=1.0)

        self._playback_thread = None
        self._playback_stop_event = None
        self._playback_queue = None

    def _play_audio_chunk(self, audio_chunk: bytes, alignment: Optional[Dict[str, Any]] = None):
        """Queue an audio chunk for background processing."""
        if not audio_chunk or self._stop_requested:
            return

        if not self._ensure_playback_worker():
            return

        if not self._playback_queue:
            return

        try:
            self._playback_queue.put_nowait((audio_chunk, alignment))
        except Exception:
            now = time.time()
            if now - self._last_playback_queue_log > 2.0:
                print("⚠️ TTS playback queue rejected chunk; dropping audio")
                self._last_playback_queue_log = now

    def _process_audio_chunk(self, audio_chunk: bytes, alignment: Optional[Dict[str, Any]]):
        """Write an audio chunk to the shared output and capture metadata."""
        if not audio_chunk or self._stop_requested:
            return

        with self._session_lock:
            session = self.current_session
            if not session:
                return
            bytes_per_sample = 2  # 16-bit PCM
            chunk_duration = len(audio_chunk) / (bytes_per_sample * self.config.sample_rate)
            chunk_start_offset = session.total_audio_duration
            chunk_end_offset = chunk_start_offset + chunk_duration

        try:
            print("playback pcm " + len(audio_chunk))
            write_playback_pcm(audio_chunk, self.config.sample_rate)
            self._chunks_played += 1
            self.is_playing = True
        except Exception as playback_error:
            if not self._stop_requested:
                print(f"Audio playback error: {playback_error}")
            return

        chunk_playback_time = time.time()

        with self._session_lock:
            session = self.current_session
            if not session:
                return
            if session.audio_start_time is None:
                session.audio_start_time = chunk_playback_time
            session.total_audio_duration = chunk_end_offset

        if alignment:
            self._record_alignment(alignment, chunk_start_offset)

    def _record_alignment(self, alignment: Dict[str, Any], chunk_start_offset: float):
        """Record alignment data for a chunk to support interruption recovery."""
        with self._session_lock:
            session = self.current_session
            if not session:
                return

            # Normalize keys: some payloads use camelCase, others snake_case
            chars = alignment.get("chars") or alignment.get("characters") or []
            start_times = alignment.get("charStartTimesMs") or alignment.get("char_start_times_ms") or []
            durations = alignment.get("charDurationsMs") or alignment.get("char_durations_ms") or []

            if not chars or not isinstance(chars, list):
                return

            session.raw_alignment_chunks.append(alignment)

            for idx, char in enumerate(chars):
                start_ms = start_times[idx] if idx < len(start_times) else 0
                duration_ms = durations[idx] if idx < len(durations) else 0

                # Convert to seconds relative to audio start
                start_time = chunk_start_offset + (start_ms / 1000.0)
                end_time = start_time + max(duration_ms, 0) / 1000.0

                alignment_char = AlignmentChar(
                    char=char,
                    start_time=start_time,
                    end_time=end_time,
                    global_index=session.alignment_char_count
                )

                session.alignment_chars.append(alignment_char)
                session.alignment_char_count += 1
                session.alignment_text += char
                session.alignment_start_times.append(start_time)

    @staticmethod
    def _extract_alignment_payload(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract alignment payload from ElevenLabs message, preferring non-normalized data."""
        alignment = data.get("alignment")
        if alignment and isinstance(alignment, dict) and alignment.get("chars"):
            return alignment

        normalized_alignment = data.get("normalizedAlignment")
        if normalized_alignment and isinstance(normalized_alignment, dict) and normalized_alignment.get("chars"):
            return normalized_alignment

        return None

    def _alignment_chars_spoken(self, elapsed_time: float, session: Optional[TTSSession] = None) -> int:
        """Return how many alignment characters have started playback within elapsed_time."""
        with self._session_lock:
            session = session or self.current_session
            if not session or not session.alignment_start_times:
                return 0

            clamped_time = max(0.0, elapsed_time)
            index = bisect.bisect_right(session.alignment_start_times, clamped_time)
            return min(index, session.alignment_char_count)

    def _alignment_chars_to_tts_index(self, alignment_char_count: int, session: Optional[TTSSession] = None) -> int:
        """Map a count of alignment characters to an index in the TTS text."""
        with self._session_lock:
            session = session or self.current_session
            if not session or not session.spoken_text_for_tts:
                return 0

            alignment_char_count = max(0, min(alignment_char_count, session.alignment_char_count))
            if alignment_char_count == 0:
                return 0

            tts_text = session.spoken_text_for_tts
            total_alignment = max(session.alignment_char_count, 1)
            ratio_index = int(len(tts_text) * min(alignment_char_count / total_alignment, 1.0))

            if session.alignment_text:
                alignment_prefix = session.alignment_text[:alignment_char_count]
                fuzzy_index = self._fuzzy_find_position(alignment_prefix, tts_text)
                if fuzzy_index:
                    return min(fuzzy_index, len(tts_text))

            return min(ratio_index, len(tts_text))

    def _resolve_session_elapsed_time(self, session: TTSSession) -> float:
        """Determine how much audio time elapsed for a session."""
        with self._session_lock:
            if not session.audio_start_time:
                return 0.0

            if session.end_time is not None:
                elapsed = session.end_time - session.audio_start_time
            else:
                elapsed = time.time() - session.audio_start_time

            max_duration = session.total_audio_duration if session.total_audio_duration > 0 else elapsed
            return max(0.0, min(elapsed, max_duration))

    def _finalize_spoken_segments(self, session: TTSSession):
        """Populate session.current_spoken_content using alignment data."""
        with self._session_lock:
            if not session:
                return

            elapsed = self._resolve_session_elapsed_time(session)
            spoken_alignment_chars = self._alignment_chars_spoken(elapsed, session)
            tts_index = self._alignment_chars_to_tts_index(spoken_alignment_chars, session)

            if tts_index <= 0:
                session.current_spoken_content = []
                return

            spoken_text = session.spoken_text_for_tts[:tts_index].strip()
            if not spoken_text:
                session.current_spoken_content = []
                return

            segment_end_time = elapsed
            segment = SpokenContent(
                text=spoken_text,
                start_time=0.0,
                end_time=segment_end_time,
                confidence=1.0
            )
            session.current_spoken_content = [segment]

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
        max_wait = 300.0  # seconds

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
                await asyncio.wait_for(self.audio_task, timeout=1500.0)
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
        self._stop_playback_worker()

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
        
        with self._session_lock:
            session = self.current_session
            if session and session.audio_start_time is not None and session.end_time is None:
                session.end_time = session.audio_start_time + session.total_audio_duration
        if session:
            self._finalize_spoken_segments(session)
    
    def get_spoken_content(self) -> List[SpokenContent]:
        """
        Get the actual spoken content from the last TTS session.
        
        Returns:
            List of SpokenContent objects representing what was actually spoken
        """
        with self._session_lock:
            session = self.current_session
        if not session:
            return []
        self._finalize_spoken_segments(session)
        with self._session_lock:
            return session.current_spoken_content.copy()
    
    def get_spoken_text(self) -> str:
        """
        Get the actual spoken text as a single string.
        
        Returns:
            Combined text of what was actually spoken
        """
        with self._session_lock:
            session = self.current_session
        if not session:
            return ""
        self._finalize_spoken_segments(session)
        with self._session_lock:
            if not session.current_spoken_content:
                return ""
            return " ".join(content.text for content in session.current_spoken_content)
    
    def get_spoken_text_heuristic(self) -> str:
        """
        Get spoken text using ElevenLabs alignment timings mapped back to the generated text.
        Only applies heuristic when there was an interruption - otherwise returns full text.
        
        Returns:
            Portion of original generated text that was likely spoken
        """
        with self._session_lock:
            session = self.current_session
            interrupted = self._interrupted
            if not session or not session.generated_text:
                return ""
            audio_start_time = session.audio_start_time
            alignment_char_count = session.alignment_char_count
            generated_text = session.generated_text
            spoken_text_for_tts = session.spoken_text_for_tts

        # If playback completed without interruption, return full text
        if not interrupted:
            return generated_text

        if audio_start_time is None:
            return ""

        elapsed = self._resolve_session_elapsed_time(session)
        spoken_alignment_chars = self._alignment_chars_spoken(elapsed, session)

        if spoken_alignment_chars == 0:
            print("🔍 [HEURISTIC DEBUG] No alignment characters recorded — returning empty string.")
            return ""

        tts_char_index = self._alignment_chars_to_tts_index(spoken_alignment_chars, session)
        reference_text = spoken_text_for_tts[:tts_char_index]

        print(f"🔍 [HEURISTIC DEBUG] Alignment chars spoken: {spoken_alignment_chars}/{alignment_char_count}")
        print(f"🔍 [HEURISTIC DEBUG] Reference text length: {len(reference_text)}")
        print(f"🔍 [HEURISTIC DEBUG] Generated text length: {len(generated_text)}")

        if not reference_text.strip():
            print("🔍 [HEURISTIC DEBUG] Reference text empty after trimming — returning empty string.")
            return ""

        # Map reference text back to generated text using fuzzy matching
        fuzzy_position = self._fuzzy_find_position(reference_text, generated_text)
        if fuzzy_position <= 0 and len(spoken_text_for_tts) > 0:
            ratio = len(reference_text) / len(spoken_text_for_tts)
            fuzzy_position = int(len(generated_text) * min(ratio * 1.1, 1.0))

        fuzzy_position = min(fuzzy_position, len(generated_text))

        # Extend to nearest word boundary for cleaner truncation
        word_end_position = fuzzy_position
        while (
            word_end_position < len(generated_text)
            and generated_text[word_end_position] not in [' ', '.', ',', '!', '?', ';', ':', '\n']
        ):
            word_end_position += 1

        result_text = generated_text[:word_end_position]

        # Ensure we don't leave emotive markers unbalanced
        asterisk_count = result_text.count('*')
        if asterisk_count % 2 == 1:
            close_pos = generated_text.find('*', word_end_position)
            if close_pos != -1:
                word_end_position = close_pos + 1
                result_text = generated_text[:word_end_position]

        result = result_text.strip()
        print(f"🔍 [HEURISTIC DEBUG] Returning heuristic result: '{result}' ({len(result)} chars)")
        return result
    
    def get_generated_vs_spoken(self) -> Dict[str, str]:
        """
        Compare generated text vs actually spoken text.
        
        Returns:
            Dictionary with 'generated', 'spoken_alignment', 'spoken_whisper', and 'spoken_heuristic' keys.
            'spoken_whisper' is kept for backward compatibility and mirrors 'spoken_alignment'.
        """
        with self._session_lock:
            session = self.current_session
        if not session:
            return {"generated": "", "spoken_alignment": "", "spoken_whisper": "", "spoken_heuristic": ""}
            
        return {
            "generated": session.generated_text,
            "spoken_alignment": self.get_spoken_text(),
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
        with self._session_lock:
            session = self.current_session
        if not session:
            return False
            
        spoken_heuristic = self.get_spoken_text_heuristic()
        with self._session_lock:
            generated_length = len(session.generated_text) if session else 0
        if generated_length == 0 or not spoken_heuristic:
            return False
        
        # If heuristic captured at least 90% of generated text, consider it fully spoken
        return len(spoken_heuristic) >= (generated_length * 0.9)

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
