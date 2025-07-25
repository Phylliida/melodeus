# Speaker Identification Across Sessions - Complete Guide

## Overview

Speaker identification across sessions is the process of recognizing and labeling speakers consistently between different recording sessions. Instead of showing "Speaker 0, Speaker 1", the system can display actual names like "Alice", "Bob", etc.

## 🎯 Approaches to Speaker Identification

### 1. **Voice Biometrics / Speaker Embeddings**
Extract unique acoustic features (voiceprints) from each speaker's voice and match them across sessions.

**How it works:**
- Extract speaker embeddings (mathematical representations of voice characteristics)
- Store these embeddings as speaker profiles
- Compare new speakers against known profiles using similarity metrics
- Assign names when confidence exceeds threshold

**Pros:**
- Automatic recognition
- Works across different sessions
- Improves accuracy over time

**Cons:**
- Requires initial training/enrollment
- Can be affected by audio quality, emotion, health
- Computationally intensive

### 2. **Manual Assignment with Learning**
Allow users to manually assign speaker names during transcription, building profiles over time.

**How it works:**
- Start with "Speaker 0, 1, 2..." labels
- User manually assigns names during or after session
- System learns voice characteristics for future sessions
- Combines manual input with automatic learning

**Pros:**
- User has full control
- Can handle edge cases
- Builds accurate profiles over time

**Cons:**
- Requires user intervention
- Initial sessions need manual work

### 3. **Enrollment-Based System**
Speakers formally register their voices before participating in sessions.

**How it works:**
- Each speaker records voice samples during enrollment
- System creates detailed voice profiles
- Future sessions automatically identify enrolled speakers
- Unknown speakers are flagged for enrollment

**Pros:**
- High accuracy for enrolled speakers
- Clear process for adding new speakers
- Works well in controlled environments

**Cons:**
- Requires upfront setup
- Doesn't handle unexpected speakers well

## 🔧 Technical Implementation

### Core Components

#### 1. **Speaker Profile System**
```python
@dataclass
class SpeakerProfile:
    name: str
    speaker_id: str
    embeddings: List[List[float]]  # Voice characteristics
    created_at: str
    last_seen: str
    session_count: int
    confidence_scores: List[float]
```

#### 2. **Voice Embedding Extraction**
Convert audio segments into mathematical representations:
- **Basic**: Spectral features (centroid, rolloff, MFCC)
- **Advanced**: Deep learning models (X-vectors, SpeakerNet, Resemblyzer)
- **Production**: Pre-trained models from Azure Cognitive Services, AWS, etc.

#### 3. **Similarity Matching**
Compare embeddings using:
- **Cosine Similarity**: Measures angle between vectors
- **Euclidean Distance**: Measures direct distance
- **Neural Networks**: Learned similarity functions

#### 4. **Threshold Management**
- **Identification Threshold**: Confidence needed to assign name (e.g., 85%)
- **Registration Threshold**: Confidence needed to update profile (e.g., 75%)
- **Rejection Threshold**: Below this, mark as "Unknown Speaker"

## 🚀 Production-Ready Solutions

### 1. **Azure Cognitive Services Speaker Recognition**
```python
# Azure Speaker Recognition API
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer

speech_config = SpeechConfig(subscription="key", region="region")
speech_config.speaker_recognition_language = "en-US"
```

### 2. **Resemblyzer (Open Source)**
```python
# Using Resemblyzer for voice embeddings
from resemblyzer import VoiceEncoder, preprocess_wav

encoder = VoiceEncoder()
wav = preprocess_wav(audio_file_path)
embed = encoder.embed_utterance(wav)
```

### 3. **SpeechBrain (Research)**
```python
# SpeechBrain for speaker recognition
from speechbrain.pretrained import SpeakerRecognition

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)
```

## 📁 File Structure

```
voicetest/
├── speaker_identification.py      # Core speaker ID system
├── integrated_speaker_stt.py      # Full integration with STT
├── custom_dictionary_stt.py       # STT with suppressed interim results
├── speaker_profiles/              # Stored speaker profiles
│   └── speaker_profiles.json      # Profile database
├── custom_dictionary_config.py    # Dictionary configuration
└── requirements.txt               # Updated with numpy
```

## 🔄 Workflow

### Session Start
1. Load existing speaker profiles
2. Initialize session speaker mapping
3. Start real-time transcription with diarization

### During Transcription
1. Deepgram identifies speakers as "Speaker 0, 1, 2..."
2. For each new session speaker:
   - Extract voice embedding from audio
   - Compare against known speakers
   - If match found: assign known name
   - If no match: label as "Unknown Speaker X"
3. Display transcription with identified names

### Session End
1. Save updated speaker profiles
2. Show session summary with speaker statistics
3. Optionally allow manual corrections

## 🎮 Usage Examples

### Basic Usage (Current Implementation)
```bash
python custom_dictionary_stt.py
# Output: [20:46:22] Speaker 0: Hello everyone
```

### With Speaker Identification
```bash
python integrated_speaker_stt.py
# Output: [20:46:22] 🎭 Alice: Hello everyone
```

### Manual Registration
```bash
python speaker_identification.py
# Interactive mode to register new speakers
```

## ⚙️ Configuration Options

### Similarity Thresholds
```python
identification_threshold = 0.85  # 85% confidence to identify
registration_threshold = 0.75    # 75% confidence to update profile
```

### Audio Requirements
- **Minimum Duration**: 2-3 seconds for reliable identification
- **Sample Rate**: 16kHz (matches Deepgram)
- **Quality**: Clear audio without background noise

### Profile Management
- **Max Embeddings**: 10 per speaker (keeps best samples)
- **Auto-cleanup**: Remove old, low-confidence embeddings
- **Backup**: Regular profile backups

## 🔍 Troubleshooting

### Low Identification Accuracy
- **Increase audio quality**: Better microphones, noise reduction
- **Longer voice samples**: Collect more training data
- **Adjust thresholds**: Lower identification threshold
- **Re-enrollment**: Have speakers re-register their voices

### False Positives
- **Increase threshold**: Require higher confidence
- **More training data**: Collect diverse voice samples
- **Manual verification**: Allow user corrections

### Performance Issues
- **Limit embeddings**: Keep only best samples per speaker
- **Batch processing**: Process identification in background
- **Caching**: Cache computed similarities

## 🚀 Future Enhancements

### 1. **Real-time Voice Embedding**
Extract embeddings directly from Deepgram's audio stream instead of simulated data.

### 2. **Advanced ML Models**
Integrate state-of-the-art speaker recognition models like:
- **X-vectors** with PLDA backend
- **ECAPA-TDNN** models
- **Transformer-based** speaker embeddings

### 3. **Multi-modal Identification**
Combine voice with other features:
- **Speech patterns** (talking speed, pauses)
- **Language patterns** (vocabulary, phrases)
- **Contextual clues** (meeting participants, calendar)

### 4. **Cloud Integration**
Use cloud-based speaker recognition:
- **Azure Speaker Recognition**
- **AWS Amazon Transcribe Speaker Identification**
- **Google Cloud Speech-to-Text Speaker Diarization**

### 5. **Privacy and Security**
- **On-device processing** for sensitive environments
- **Encrypted profiles** for data protection
- **Consent management** for voice data collection

## 📚 References

- [Deepgram Diarization Documentation](https://developers.deepgram.com/docs/diarization)
- [Azure Speaker Recognition](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/speaker-recognition-overview)
- [Resemblyzer: Real-time Voice Cloning](https://github.com/resemble-ai/Resemblyzer)
- [SpeechBrain Speaker Recognition](https://speechbrain.github.io/tutorial_speaker_rec.html)
- [X-vector Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) 