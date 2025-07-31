#!/usr/bin/env python3
"""
Standalone voice fingerprinting testing and calibration tool.
"""

import numpy as np
import sounddevice as sd
import wave
import time
import os
from pathlib import Path
from titanet_voice_fingerprinting import TitaNetVoiceFingerprinter, TitaNetFingerprint
from config_loader import load_config

class VoiceFingerprintTester:
    def __init__(self, config_path="config.yaml"):
        """Initialize the tester with configuration."""
        self.config = load_config(config_path)
        self.fingerprinter = TitaNetVoiceFingerprinter(self.config.speakers)
        self.test_samples_dir = Path("test_samples")
        self.test_samples_dir.mkdir(exist_ok=True)
        
        print("🔊 Voice Fingerprint Tester Initialized")
        print(f"📁 Test samples will be saved to: {self.test_samples_dir}")
        print(f"🎯 Current similarity threshold: {self.config.speakers.recognition.confidence_threshold}")
        
    def record_test_sample(self, speaker_name: str, sample_name: str, duration: float = 5.0):
        """Record a test sample for a specific speaker."""
        print(f"\n🎤 Recording {duration}s test sample: {speaker_name}/{sample_name}")
        print(f"📍 Speak clearly for {duration} seconds...")
        print("⏱️  Starting in 3...")
        time.sleep(1)
        print("⏱️  2...")
        time.sleep(1)
        print("⏱️  1...")
        time.sleep(1)
        print("🔴 RECORDING NOW!")
        
        # Record audio
        sample_rate = 16000
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()
        
        print("✅ Recording complete!")
        
        # Save to file
        speaker_dir = self.test_samples_dir / speaker_name
        speaker_dir.mkdir(exist_ok=True)
        filepath = speaker_dir / f"{sample_name}.wav"
        
        # Convert to int16 for WAV
        recording_int16 = (recording * 32767).astype(np.int16)
        
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(recording_int16.tobytes())
        
        print(f"💾 Saved to: {filepath}")
        return filepath
        
    def test_sample_against_profiles(self, audio_file: Path) -> dict:
        """Test a recorded sample against all known profiles."""
        print(f"\n🧪 Testing sample: {audio_file}")
        
        # Load audio
        import librosa
        audio_data, sr = librosa.load(str(audio_file), sr=16000)
        
        # Create fingerprint
        fingerprint = self.fingerprinter._create_fingerprint(audio_data)
        if fingerprint is None:
            print("❌ Failed to create fingerprint")
            return {}
        
        print(f"🔬 Created fingerprint with {len(fingerprint.embedding)} features")
        
        # Test against all known speakers
        results = {}
        print(f"🎯 Testing against {len(self.fingerprinter.reference_fingerprints)} speaker profiles:")
        
        for speaker_id, fingerprints in self.fingerprinter.reference_fingerprints.items():
            if not fingerprints:
                continue
                
            # Find best similarity
            similarities = []
            for ref_fp in fingerprints:
                similarity = self._cosine_similarity(fingerprint.embedding, ref_fp.embedding)
                similarities.append(similarity)
            
            best_similarity = max(similarities) if similarities else 0.0
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            results[speaker_id] = {
                'best_similarity': best_similarity,
                'avg_similarity': avg_similarity,
                'num_references': len(similarities),
                'all_similarities': similarities
            }
            
            # Get speaker name
            speaker_name = "Unknown"
            for profile_id, profile in self.config.speakers.profiles.items():
                if profile_id == speaker_id:
                    speaker_name = profile.name
                    break
            
            print(f"  📊 {speaker_name} ({speaker_id}): best={best_similarity:.3f}, avg={avg_similarity:.3f} ({len(similarities)} refs)")
        
        # Find best match
        if results:
            best_speaker = max(results.keys(), key=lambda x: results[x]['best_similarity'])
            best_score = results[best_speaker]['best_similarity']
            
            # Get speaker name
            best_name = "Unknown"
            for profile_id, profile in self.config.speakers.profiles.items():
                if profile_id == best_speaker:
                    best_name = profile.name
                    break
            
            print(f"🏆 Best match: {best_name} ({best_speaker}) with similarity {best_score:.3f}")
            
            threshold = self.config.speakers.recognition.confidence_threshold
            if best_score >= threshold:
                print(f"✅ Match accepted (above threshold {threshold:.3f})")
            else:
                print(f"❌ Match rejected (below threshold {threshold:.3f})")
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    def run_calibration_test(self, speaker_name: str, num_samples: int = 5):
        """Record multiple samples and test recognition accuracy."""
        print(f"\n🔬 CALIBRATION TEST for {speaker_name}")
        print(f"📝 Will record {num_samples} test samples")
        
        results = []
        
        for i in range(num_samples):
            print(f"\n--- Sample {i+1}/{num_samples} ---")
            
            # Record sample
            sample_file = self.record_test_sample(speaker_name, f"test_{i+1}", duration=3.0)
            
            # Test recognition
            test_results = self.test_sample_against_profiles(sample_file)
            results.append({
                'file': sample_file,
                'results': test_results
            })
            
            print("Press Enter to continue to next sample...")
            input()
        
        # Analyze results
        print(f"\n📊 CALIBRATION RESULTS for {speaker_name}")
        self._analyze_calibration_results(speaker_name, results)
        
        return results
    
    def _analyze_calibration_results(self, expected_speaker: str, results: list):
        """Analyze calibration test results."""
        correct_matches = 0
        total_tests = len(results)
        
        # Find expected speaker ID
        expected_id = None
        for profile_id, profile in self.config.speakers.profiles.items():
            if profile.name.lower() == expected_speaker.lower():
                expected_id = profile_id
                break
        
        if expected_id is None:
            print(f"❌ Could not find speaker profile for '{expected_speaker}'")
            return
        
        print(f"🎯 Expected speaker: {expected_speaker} (ID: {expected_id})")
        print(f"📋 Results summary:")
        
        all_similarities = {profile_id: [] for profile_id in self.fingerprinter.reference_fingerprints.keys()}
        
        for i, result in enumerate(results):
            test_results = result['results']
            
            if not test_results:
                print(f"  Sample {i+1}: ❌ No results")
                continue
            
            # Find best match
            best_speaker = max(test_results.keys(), key=lambda x: test_results[x]['best_similarity'])
            best_score = test_results[best_speaker]['best_similarity']
            
            # Check if correct
            is_correct = best_speaker == expected_id
            if is_correct:
                correct_matches += 1
            
            # Get speaker name
            best_name = "Unknown"
            for profile_id, profile in self.config.speakers.profiles.items():
                if profile_id == best_speaker:
                    best_name = profile.name
                    break
            
            status = "✅" if is_correct else "❌"
            print(f"  Sample {i+1}: {status} → {best_name} ({best_score:.3f})")
            
            # Collect similarities for analysis
            for speaker_id, speaker_results in test_results.items():
                all_similarities[speaker_id].append(speaker_results['best_similarity'])
        
        # Overall accuracy
        accuracy = correct_matches / total_tests * 100
        print(f"\n🎯 Overall Accuracy: {correct_matches}/{total_tests} ({accuracy:.1f}%)")
        
        # Similarity analysis
        print(f"\n📈 Similarity Analysis:")
        for speaker_id, similarities in all_similarities.items():
            if not similarities:
                continue
                
            speaker_name = "Unknown"
            for profile_id, profile in self.config.speakers.profiles.items():
                if profile_id == speaker_id:
                    speaker_name = profile.name
                    break
            
            avg_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)
            
            marker = "🎯" if speaker_id == expected_id else "📊"
            print(f"  {marker} {speaker_name}: avg={avg_sim:.3f}±{std_sim:.3f}, range=[{min_sim:.3f}, {max_sim:.3f}]")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if accuracy < 80:
            print("  - Accuracy is low. Consider:")
            print("    • Recording longer reference samples")
            print("    • Speaking more clearly/consistently")
            print("    • Adjusting similarity threshold")
            print("    • Improving audio quality (noise reduction)")
        
        # Check if threshold needs adjustment
        if expected_id in all_similarities:
            expected_sims = all_similarities[expected_id]
            other_sims = []
            for speaker_id, sims in all_similarities.items():
                if speaker_id != expected_id:
                    other_sims.extend(sims)
            
            if expected_sims and other_sims:
                expected_avg = np.mean(expected_sims)
                other_max = np.max(other_sims)
                
                print(f"  - Expected speaker avg similarity: {expected_avg:.3f}")
                print(f"  - Highest other speaker similarity: {other_max:.3f}")
                
                threshold = self.config.speakers.recognition.confidence_threshold
                if expected_avg < threshold:
                    suggested_threshold = expected_avg * 0.9
                    print(f"  - Consider lowering threshold to {suggested_threshold:.3f}")
                elif other_max > threshold:
                    suggested_threshold = (expected_avg + other_max) / 2
                    print(f"  - Consider setting threshold to {suggested_threshold:.3f}")

def main():
    """Main interactive testing interface."""
    tester = VoiceFingerprintTester()
    
    while True:
        print("\n" + "="*60)
        print("🔊 VOICE FINGERPRINT TESTING TOOL")
        print("="*60)
        print("1. Record single test sample")
        print("2. Test existing audio file")
        print("3. Run calibration test (multiple samples)")
        print("4. List available speakers")
        print("5. Show reference fingerprint info")
        print("6. Exit")
        print("-"*60)
        
        choice = input("Choose option (1-6): ").strip()
        
        if choice == "1":
            speaker_name = input("Enter speaker name: ").strip()
            sample_name = input("Enter sample name: ").strip()
            duration = float(input("Enter duration in seconds (default 5): ") or "5")
            
            sample_file = tester.record_test_sample(speaker_name, sample_name, duration)
            tester.test_sample_against_profiles(sample_file)
            
        elif choice == "2":
            file_path = input("Enter path to audio file: ").strip()
            if os.path.exists(file_path):
                tester.test_sample_against_profiles(Path(file_path))
            else:
                print("❌ File not found!")
                
        elif choice == "3":
            speaker_name = input("Enter speaker name for calibration: ").strip()
            num_samples = int(input("Enter number of test samples (default 5): ") or "5")
            tester.run_calibration_test(speaker_name, num_samples)
            
        elif choice == "4":
            print("\n📋 Available speakers:")
            for profile_id, profile in tester.config.speakers.profiles.items():
                ref_count = len(tester.fingerprinter.reference_fingerprints.get(profile_id, []))
                print(f"  - {profile.name} ({profile_id}): {ref_count} reference fingerprints")
                
        elif choice == "5":
            print("\n📊 Reference fingerprint info:")
            total_fingerprints = 0
            for speaker_id, fingerprints in tester.fingerprinter.reference_fingerprints.items():
                speaker_name = "Unknown"
                for profile_id, profile in tester.config.speakers.profiles.items():
                    if profile_id == speaker_id:
                        speaker_name = profile.name
                        break
                        
                count = len(fingerprints)
                total_fingerprints += count
                print(f"  {speaker_name} ({speaker_id}): {count} fingerprints")
                
                if fingerprints:
                    embedding_size = len(fingerprints[0].embedding)
                    print(f"    → Embedding size: {embedding_size} features")
                    
            print(f"\nTotal: {total_fingerprints} reference fingerprints")
            print(f"Confidence threshold: {tester.config.speakers.recognition.confidence_threshold}")
            
        elif choice == "6":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()