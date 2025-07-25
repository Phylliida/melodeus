# Voice-to-LLM Conversation System Guide

## Overview

This system provides a complete voice conversation interface with AI that includes:

- **Real-time Speech-to-Text** (Deepgram)
- **Intelligent conversation management** with pause detection and completion analysis
- **Streaming LLM responses** (OpenAI GPT)
- **Real-time Text-to-Speech** (ElevenLabs WebSocket streaming)
- **Smart interruption handling** for natural conversation flow

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Microphone    │───▶│   STT (Deepgram) │───▶│  Conversation   │
│     Input       │    │                  │    │    Manager      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Speaker       │◀───│ TTS (ElevenLabs) │◀───│ LLM (OpenAI)    │
│    Output       │    │   WebSocket      │    │   Streaming     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components

### 1. **Simple ElevenLabs Streaming Test** (`test_elevenlabs_streaming.py`)
Tests basic TTS streaming functionality:
- Connects to ElevenLabs WebSocket API
- Streams LLM completions to TTS
- Plays audio in real-time
- Shows text-to-audio alignment

### 2. **Full Voice Conversation System** (`voice_llm_conversation.py`)
Complete integrated system with:
- Real-time STT with speaker detection
- Intelligent LLM submission logic
- Streaming TTS with interruption support
- Conversation state management

## Key Features

### Intelligent LLM Submission
The system decides when to submit text to the LLM based on:

- **Pause Detection**: 2-second silence after speech
- **Minimum Words**: At least 3 words before submission
- **Maximum Wait**: Force submission after 10 seconds
- **Completion Analysis**: Detects complete statements (punctuation, phrases)

### Smart Interruption Handling
- **Detection**: Monitors speech during TTS playback
- **Threshold**: Requires confident speech (>0.8 confidence)
- **Action**: Immediately stops TTS and clears audio queue
- **Recovery**: Allows immediate resumption of conversation

### Conversation State Management
- **History Tracking**: Maintains conversation context
- **State Flags**: Tracks listening, TTS playback, pending submissions
- **Buffer Management**: Accumulates transcripts before submission

## Configuration

### Timing Settings
```python
pause_threshold = 2.0          # Seconds of silence for submission
min_words_for_submission = 3   # Minimum words required
max_wait_time = 10.0          # Maximum wait before force submission
interruption_threshold = 1.0   # Speech duration to interrupt TTS
interruption_confidence = 0.8  # Minimum confidence for interruption
```

### Audio Settings
```python
sample_rate = 16000    # STT sample rate
chunk_size = 8000      # Audio chunk size
tts_rate = 44100       # TTS playback rate
```

## Setup Requirements

### API Keys Needed
1. **Deepgram API Key** - For speech-to-text
2. **ElevenLabs API Key** - For text-to-speech
3. **OpenAI API Key** - For LLM responses

### Voice ID Configuration
- Default: `pNInz6obpgDQGcFmaJgB` (Rachel voice)
- You can find other voices in your ElevenLabs dashboard
- Test different voices for optimal experience

## Usage Examples

### Testing ElevenLabs Streaming
```bash
python test_elevenlabs_streaming.py
```
Enter your API keys and test prompts to verify TTS streaming works.

### Full Voice Conversation
```bash
python voice_llm_conversation.py
```
Start a complete voice conversation with the AI.

## Conversation Flow

1. **🎤 Listening Phase**
   - System captures audio from microphone
   - Deepgram provides real-time transcription
   - Shows interim and final results

2. **🧠 Decision Phase**
   - Analyzes pause duration and word count
   - Checks for completion indicators
   - Decides when to submit to LLM

3. **🤖 LLM Phase**
   - Sends accumulated transcript to OpenAI
   - Streams response in chunks
   - Maintains conversation history

4. **🔊 TTS Phase**
   - Streams LLM response to ElevenLabs
   - Plays audio in real-time
   - Monitors for interruptions

5. **🔄 Loop**
   - Returns to listening phase
   - Handles interruptions gracefully

## Advanced Features

### Completion Detection
The system recognizes complete statements through:
- **Punctuation**: Sentences ending with `.`, `!`, `?`
- **Phrases**: Common completion phrases like "that's it", "thank you"
- **Context**: Natural conversation flow analysis

### Interruption Management
- **Immediate Response**: Sub-second interruption detection
- **Audio Cleanup**: Clears pending audio buffers
- **State Recovery**: Seamlessly resumes conversation
- **False Positive Prevention**: Requires confident speech detection

### Error Handling
- **Connection Recovery**: Handles WebSocket disconnections
- **Audio Fallbacks**: Manages microphone/speaker issues
- **API Errors**: Graceful degradation for API failures
- **Resource Cleanup**: Proper cleanup on exit

## Troubleshooting

### Common Issues

1. **No Audio Input**
   - Check microphone permissions
   - Verify PyAudio installation
   - Test with `test_microphone.py`

2. **TTS Not Playing**
   - Verify ElevenLabs API key
   - Check voice ID validity
   - Test with simple streaming script

3. **LLM Not Responding**
   - Confirm OpenAI API key
   - Check rate limits
   - Verify model availability

4. **Interruption Not Working**
   - Adjust confidence threshold
   - Check microphone sensitivity
   - Verify VAD settings

### Performance Optimization

- **Buffer Sizes**: Adjust chunk sizes for your hardware
- **API Limits**: Monitor rate limits for all services
- **Memory Usage**: Clear conversation history periodically
- **Latency**: Optimize network connection quality

## Future Enhancements

- **Multi-language Support**: Extend to other languages
- **Voice Cloning**: Custom voice creation
- **Emotion Detection**: Analyze speech sentiment
- **Context Awareness**: Better conversation understanding
- **Visual Interface**: Add GUI for easier control

## Integration with Existing System

This system builds on the existing speaker identification framework:
- Can be combined with speaker profiles
- Maintains conversation history per speaker
- Supports multi-speaker scenarios
- Compatible with existing custom dictionaries

## Best Practices

1. **Speak Naturally**: The system is designed for conversational speech
2. **Clear Pauses**: Brief pauses help with submission timing
3. **Interruption**: Feel free to interrupt - it's designed for it
4. **Context**: Reference previous conversation for better responses
5. **Testing**: Start with simple tests before complex conversations 