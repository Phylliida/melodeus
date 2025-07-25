#!/usr/bin/env python3
"""
Speaker Identification System for Persistent Speaker Recognition
Matches speaker voices across sessions using voice embeddings
"""

import json
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class SpeakerProfile:
    """Speaker profile containing voice characteristics and metadata."""
    name: str
    speaker_id: str
    embeddings: List[List[float]]  # Multiple voice embeddings
    created_at: str
    last_seen: str
    session_count: int
    confidence_scores: List[float]
    
    def add_embedding(self, embedding: List[float], confidence: float = 1.0):
        """Add a new voice embedding to this speaker's profile."""
        self.embeddings.append(embedding)
        self.confidence_scores.append(confidence)
        self.last_seen = datetime.now().isoformat()
        
        # Keep only the best embeddings (limit to 10)
        if len(self.embeddings) > 10:
            # Remove the embedding with lowest confidence
            min_idx = self.confidence_scores.index(min(self.confidence_scores))
            self.embeddings.pop(min_idx)
            self.confidence_scores.pop(min_idx)
    
    def get_average_embedding(self) -> np.ndarray:
        """Get the average embedding for this speaker."""
        if not self.embeddings:
            return np.array([])
        return np.mean(self.embeddings, axis=0)

class SpeakerIdentificationSystem:
    """System for identifying speakers across sessions."""
    
    def __init__(self, profiles_dir: str = "speaker_profiles"):
        """Initialize the speaker identification system."""
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        
        self.known_speakers: Dict[str, SpeakerProfile] = {}
        self.session_speakers: Dict[int, str] = {}  # Maps session speaker_id to known speaker_id
        self.unknown_count = 0
        
        # Similarity thresholds
        self.identification_threshold = 0.85  # Confidence needed to identify
        self.registration_threshold = 0.75    # Confidence needed to add to profile
        
        self.load_speaker_profiles()
    
    def load_speaker_profiles(self):
        """Load existing speaker profiles from disk."""
        try:
            profiles_file = self.profiles_dir / "speaker_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                
                for speaker_id, profile_data in profiles_data.items():
                    self.known_speakers[speaker_id] = SpeakerProfile(**profile_data)
                
                print(f"📚 Loaded {len(self.known_speakers)} known speaker profiles")
            else:
                print("📝 No existing speaker profiles found - starting fresh")
        except Exception as e:
            print(f"⚠️  Error loading speaker profiles: {e}")
    
    def save_speaker_profiles(self):
        """Save speaker profiles to disk."""
        try:
            profiles_file = self.profiles_dir / "speaker_profiles.json"
            profiles_data = {}
            
            for speaker_id, profile in self.known_speakers.items():
                profiles_data[speaker_id] = asdict(profile)
            
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
                
            print(f"💾 Saved {len(self.known_speakers)} speaker profiles")
        except Exception as e:
            print(f"⚠️  Error saving speaker profiles: {e}")
    
    def extract_speaker_embedding(self, audio_segment: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract speaker embedding from audio segment.
        This is a simplified version - in practice, you'd use a pre-trained model
        like SpeakerNet, X-Vector, or resemblyzer.
        """
        # Simplified feature extraction (normally you'd use a neural network)
        # This extracts basic acoustic features as a placeholder
        
        # Compute basic spectral features
        fft = np.fft.fft(audio_segment)
        magnitude = np.abs(fft)
        
        # Extract features (this is very basic - real systems use much more sophisticated methods)
        features = []
        
        # Spectral centroid
        freqs = np.fft.fftfreq(len(magnitude), 1/sample_rate)
        spectral_centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
        features.append(spectral_centroid)
        
        # Spectral rolloff
        cumsum = np.cumsum(magnitude[:len(magnitude)//2])
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0][0]
        spectral_rolloff = freqs[rolloff_idx]
        features.append(spectral_rolloff)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio_segment)) != 0)
        zcr = zero_crossings / len(audio_segment)
        features.append(zcr)
        
        # MFCC-like features (simplified)
        # In practice, you'd compute proper MFCCs
        mel_filters = 13
        for i in range(mel_filters):
            start_idx = i * len(magnitude) // (2 * mel_filters)
            end_idx = (i + 1) * len(magnitude) // (2 * mel_filters)
            mel_energy = np.sum(magnitude[start_idx:end_idx])
            features.append(mel_energy)
        
        return np.array(features)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two speaker embeddings."""
        if len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0
        
        # Normalize embeddings
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        # Convert to 0-1 range
        return (similarity + 1) / 2
    
    def identify_speaker(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Identify a speaker based on their voice embedding.
        Returns (speaker_name, confidence) or (None, 0) if unknown.
        """
        if len(self.known_speakers) == 0:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for speaker_id, profile in self.known_speakers.items():
            avg_embedding = profile.get_average_embedding()
            if len(avg_embedding) == 0:
                continue
                
            similarity = self.calculate_similarity(embedding, avg_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id
        
        # Only return identification if confidence is high enough
        if best_similarity >= self.identification_threshold:
            return self.known_speakers[best_match].name, best_similarity
        
        return None, best_similarity
    
    def register_speaker(self, name: str, embedding: np.ndarray, force: bool = False) -> str:
        """
        Register a new speaker with their voice embedding.
        Returns the speaker_id.
        """
        # Generate unique speaker ID
        speaker_id = f"speaker_{len(self.known_speakers):03d}_{int(time.time())}"
        
        # Create new profile
        profile = SpeakerProfile(
            name=name,
            speaker_id=speaker_id,
            embeddings=[embedding.tolist()],
            created_at=datetime.now().isoformat(),
            last_seen=datetime.now().isoformat(),
            session_count=1,
            confidence_scores=[1.0]
        )
        
        self.known_speakers[speaker_id] = profile
        self.save_speaker_profiles()
        
        print(f"✅ Registered new speaker: {name} (ID: {speaker_id})")
        return speaker_id
    
    def update_speaker_profile(self, speaker_id: str, embedding: np.ndarray, confidence: float):
        """Update an existing speaker's profile with new embedding."""
        if speaker_id in self.known_speakers:
            self.known_speakers[speaker_id].add_embedding(embedding.tolist(), confidence)
            self.known_speakers[speaker_id].session_count += 1
            self.save_speaker_profiles()
    
    def process_session_speaker(self, session_speaker_id: int, audio_segment: np.ndarray, 
                              sample_rate: int = 16000) -> Tuple[str, str]:
        """
        Process a speaker from the current session and return (display_name, speaker_type).
        speaker_type can be 'known', 'unknown', or 'new'
        """
        # Extract embedding from audio
        embedding = self.extract_speaker_embedding(audio_segment, sample_rate)
        
        # Check if we already processed this session speaker
        if session_speaker_id in self.session_speakers:
            known_id = self.session_speakers[session_speaker_id]
            if known_id in self.known_speakers:
                # Update profile with new embedding
                self.update_speaker_profile(known_id, embedding, 0.8)
                return self.known_speakers[known_id].name, "known"
        
        # Try to identify the speaker
        identified_name, confidence = self.identify_speaker(embedding)
        
        if identified_name:
            # Found a match - map this session speaker to known speaker
            for speaker_id, profile in self.known_speakers.items():
                if profile.name == identified_name:
                    self.session_speakers[session_speaker_id] = speaker_id
                    self.update_speaker_profile(speaker_id, embedding, confidence)
                    return identified_name, "known"
        
        # Unknown speaker - assign temporary name
        self.unknown_count += 1
        unknown_name = f"Unknown Speaker {self.unknown_count}"
        return unknown_name, "unknown"
    
    def get_speaker_name(self, session_speaker_id: int) -> str:
        """Get the display name for a session speaker."""
        if session_speaker_id in self.session_speakers:
            known_id = self.session_speakers[session_speaker_id]
            if known_id in self.known_speakers:
                return self.known_speakers[known_id].name
        
        return f"Speaker {session_speaker_id}"
    
    def interactive_speaker_registration(self):
        """Interactive mode for registering speakers."""
        print("\n🎤 Speaker Registration Mode")
        print("Say your name and a few sentences so I can learn your voice...")
        
        # In a real implementation, you'd record audio here
        # For now, we'll simulate this
        
        name = input("Enter speaker name: ").strip()
        if name:
            # Simulate recording and embedding extraction
            dummy_embedding = np.random.rand(16)  # In practice, extract from actual audio
            self.register_speaker(name, dummy_embedding)
    
    def list_known_speakers(self):
        """List all known speakers."""
        if not self.known_speakers:
            print("📝 No speakers registered yet")
            return
        
        print(f"\n👥 Known Speakers ({len(self.known_speakers)}):")
        for profile in self.known_speakers.values():
            print(f"  • {profile.name}")
            print(f"    - ID: {profile.speaker_id}")
            print(f"    - Registered: {profile.created_at[:10]}")
            print(f"    - Sessions: {profile.session_count}")
            print(f"    - Voice samples: {len(profile.embeddings)}")

# Integration with the main STT system
class EnhancedSpeakerSTT:
    """Enhanced STT system with speaker identification."""
    
    def __init__(self, api_key: str):
        """Initialize with speaker identification."""
        # Initialize the base STT system (your existing code)
        self.speaker_system = SpeakerIdentificationSystem()
        
        # Audio buffer for speaker identification
        self.speaker_audio_buffers: Dict[int, List[np.ndarray]] = {}
        self.min_audio_length = 16000 * 2  # 2 seconds minimum for identification
    
    def process_speaker_audio(self, speaker_id: int, audio_data: np.ndarray):
        """Process audio data for speaker identification."""
        if speaker_id not in self.speaker_audio_buffers:
            self.speaker_audio_buffers[speaker_id] = []
        
        self.speaker_audio_buffers[speaker_id].append(audio_data)
        
        # Check if we have enough audio for identification
        total_length = sum(len(chunk) for chunk in self.speaker_audio_buffers[speaker_id])
        
        if total_length >= self.min_audio_length:
            # Concatenate audio and identify speaker
            full_audio = np.concatenate(self.speaker_audio_buffers[speaker_id])
            display_name, speaker_type = self.speaker_system.process_session_speaker(
                speaker_id, full_audio
            )
            
            # Clear buffer
            self.speaker_audio_buffers[speaker_id] = []
            
            return display_name, speaker_type
        
        return None, None

def main():
    """Test the speaker identification system."""
    speaker_system = SpeakerIdentificationSystem()
    
    print("🎯 Speaker Identification System Test")
    
    # Show existing speakers
    speaker_system.list_known_speakers()
    
    # Interactive registration
    while True:
        print("\nOptions:")
        print("1. Register new speaker")
        print("2. List known speakers")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            speaker_system.interactive_speaker_registration()
        elif choice == "2":
            speaker_system.list_known_speakers()
        elif choice == "3":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 