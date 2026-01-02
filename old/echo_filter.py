"""
Echo filter for preventing TTS output from triggering interruptions.
Uses fuzzy matching to detect when Deepgram transcribes the AI's own speech.
"""
import time
import difflib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class TTSSession:
    """Track a TTS session with its text and timing."""
    session_id: str
    text: str
    start_time: float
    character: Optional[str] = None
    chunks: List[str] = field(default_factory=list)  # Track chunks as they're sent
    completion_time: Optional[float] = None  # When the session completed
    
    def get_full_text(self) -> str:
        """Get the accumulated text from all chunks."""
        return "".join(self.chunks) if self.chunks else self.text


class EchoFilter:
    """
    Filter out echo/feedback from TTS output being picked up by STT.
    Handles multiple parallel TTS sessions from different characters.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.75,
                 time_window: float = 15.0,
                 min_length: int = 3):
        """
        Initialize echo filter.
        
        Args:
            similarity_threshold: How similar text must be to be considered echo (0-1)
            time_window: How long to keep TTS sessions for comparison (seconds)
            min_length: Minimum text length to check (ignore very short utterances)
        """
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window
        self.min_length = min_length
        
        # Track active TTS sessions by session ID
        self.active_sessions: Dict[str, TTSSession] = {}
        
        # Track recent TTS sessions for echo detection
        self.recent_sessions: deque[TTSSession] = deque()
        
    def on_tts_start(self, session_id: str, character: Optional[str] = None) -> None:
        """Called when a new TTS session starts."""
        self.active_sessions[session_id] = TTSSession(
            session_id=session_id,
            text="",
            start_time=time.time(),
            character=character
        )
        print(f"🔊 Echo filter: Started tracking TTS session {session_id} for {character or 'unknown'}")
        
    def on_tts_chunk(self, session_id: str, text_chunk: str) -> None:
        """Called when a chunk of text is sent to TTS."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.chunks.append(text_chunk)
            session.text = session.get_full_text()
            
    def on_tts_complete(self, session_id: str) -> None:
        """Called when a TTS session completes."""
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            session.completion_time = time.time()  # Mark when it completed
            self.recent_sessions.append(session)
            print(f"🔊 Echo filter: Completed TTS session {session_id}, tracking for echo detection")
            
            # Clean up old sessions
            self._cleanup_old_sessions()
            
    def on_tts_interrupted(self, session_id: str, spoken_text: Optional[str] = None) -> None:
        """Called when a TTS session is interrupted."""
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            session.completion_time = time.time()  # Mark when it was interrupted
            
            # If we have the actual spoken text from Whisper, use that
            if spoken_text:
                session.text = spoken_text
                
            self.recent_sessions.append(session)
            print(f"🔊 Echo filter: TTS session {session_id} interrupted, partial text tracked")
            
            # Clean up old sessions
            self._cleanup_old_sessions()
            
    def is_echo(self, stt_text: str) -> Tuple[bool, Optional[str], float]:
        """
        Check if STT text is likely an echo of recent TTS output.
        
        Args:
            stt_text: Text from STT (Deepgram)
            
        Returns:
            Tuple of (is_echo, matched_tts_text, similarity_score)
        """
        # Skip very short utterances
        if len(stt_text.strip()) < self.min_length:
            return False, None, 0.0
            
        stt_lower = stt_text.lower().strip()
        
        # Check active sessions first (currently speaking)
        for session in self.active_sessions.values():
            is_match, similarity = self._check_similarity(stt_lower, session.text.lower().strip())
            if is_match:
                print(f"🔇 Echo detected (active): '{stt_text}' ≈ TTS '{session.text[:50]}...' ({similarity:.0%})")
                return True, session.text, similarity
                
        # Check recent completed sessions
        for session in self.recent_sessions:
            is_match, similarity = self._check_similarity(stt_lower, session.text.lower().strip())
            if is_match:
                elapsed = time.time() - session.start_time
                print(f"🔇 Echo detected (recent, {elapsed:.1f}s ago): '{stt_text}' ≈ TTS '{session.text[:50]}...' ({similarity:.0%})")
                return True, session.text, similarity
                
        return False, None, 0.0
        
    def _check_similarity(self, stt_text: str, tts_text: str) -> Tuple[bool, float]:
        """
        Check similarity between STT and TTS text.
        
        Returns:
            Tuple of (is_similar, similarity_score)
        """
        if not stt_text or not tts_text:
            return False, 0.0
            
        # Check if STT is a substring (partial echo)
        if len(stt_text) > 3 and stt_text in tts_text:
            return True, 1.0
            
        # Use sequence matcher for fuzzy comparison
        matcher = difflib.SequenceMatcher(None, stt_text, tts_text)
        similarity = matcher.ratio()
        
        # Also check if STT might be a partial match of TTS
        # (Deepgram might only catch part of what was said)
        if len(stt_text) < len(tts_text):
            # Find best matching substring in TTS
            best_partial_similarity = 0.0
            words_stt = stt_text.split()
            words_tts = tts_text.split()
            
            # Sliding window to find best match
            for i in range(len(words_tts) - len(words_stt) + 1):
                window = " ".join(words_tts[i:i + len(words_stt)])
                matcher = difflib.SequenceMatcher(None, stt_text, window.lower())
                partial_sim = matcher.ratio()
                best_partial_similarity = max(best_partial_similarity, partial_sim)
                
            similarity = max(similarity, best_partial_similarity)
            
        return similarity >= self.similarity_threshold, similarity
        
    def _cleanup_old_sessions(self) -> None:
        """Remove sessions older than the time window."""
        current_time = time.time()
        
        # Remove sessions based on completion time, not start time
        while self.recent_sessions:
            session = self.recent_sessions[0]
            # Use completion_time if available, otherwise use start_time + a buffer
            session_age_time = session.completion_time if session.completion_time else session.start_time
            
            # Only remove if the session has been completed/interrupted for longer than time_window
            if current_time - session_age_time > self.time_window:
                old_session = self.recent_sessions.popleft()
                print(f"🗑️ Echo filter: Removed old TTS session {old_session.session_id} (aged out after {self.time_window}s)")
            else:
                # Sessions are ordered, so if this one isn't old enough, neither are the rest
                break
            
    def clear(self) -> None:
        """Clear all tracked sessions."""
        self.active_sessions.clear()
        self.recent_sessions.clear()
        
    def get_active_sessions_info(self) -> List[Dict]:
        """Get information about current active sessions for debugging."""
        return [
            {
                "session_id": session.session_id,
                "character": session.character,
                "text_length": len(session.text),
                "duration": time.time() - session.start_time
            }
            for session in self.active_sessions.values()
        ]