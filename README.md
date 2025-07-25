# Voice AI System with Real-time STT, LLM Integration & Speaker Identification

A comprehensive voice AI system that combines real-time speech-to-text, intelligent conversation management, LLM integration, streaming text-to-speech, and advanced speaker identification.

## 🚀 New: Voice-to-LLM Conversation System

The latest addition enables natural voice conversations with AI:
- **Intelligent LLM Submission**: Smart pause detection and completion analysis
- **Streaming TTS**: Real-time text-to-speech with ElevenLabs WebSocket API
- **Interruption Handling**: Natural conversation flow with voice interruption support
- **Conversation Management**: Context-aware dialogue with history tracking

## Core Features

### Real-time Speech-to-Text
- **Live STT**: Real-time transcription using Deepgram's nova-3 model
- **Live Diarization**: Identifies and separates different speakers in real-time
- **Color-coded Output**: Each speaker gets a unique color for easy identification
- **Interim Results**: Shows both interim and final transcription results
- **Voice Activity Detection**: Automatically detects speech events
- **Smart Formatting**: Includes punctuation and formatting

### Speaker Identification System
- **Persistent Recognition**: Remembers speakers across sessions
- **Voice Profiles**: Creates and stores speaker voice signatures
- **False Positive Prevention**: Advanced similarity analysis with margin requirements
- **Session Tracking**: Maps session speakers to known identities
- **Audio Feature Analysis**: Multi-dimensional voice characteristic extraction

### Custom Dictionary & Configuration
- **Custom Vocabulary**: Boost specific terms and industry keywords
- **Word Replacements**: Automatic text substitution and correction
- **Configurable Settings**: Adjustable transcription parameters
- **Preset Libraries**: Industry-specific vocabulary sets

### Voice-to-LLM Conversation
- **Natural Conversation Flow**: Intelligent submission timing based on pauses and completion
- **Real-time TTS**: Streaming audio response with ElevenLabs
- **Smart Interruption**: Ability to interrupt AI speech naturally
- **Context Awareness**: Maintains conversation history and context

## Prerequisites

### For Basic STT and Speaker Identification:
1. **Deepgram API Key**: Sign up at [https://console.deepgram.com/](https://console.deepgram.com/) to get your API key
2. **Python 3.7+**: Make sure you have Python 3.7 or higher installed
3. **Microphone**: A working microphone connected to your system

### Additional for Voice-to-LLM Conversation:
4. **ElevenLabs API Key**: Sign up at [https://elevenlabs.io/](https://elevenlabs.io/) for text-to-speech
5. **OpenAI API Key**: Get your API key from [https://platform.openai.com/](https://platform.openai.com/)
6. **Speakers/Headphones**: Audio output device for AI responses

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Deepgram API key as an environment variable:
```bash
export DEEPGRAM_API_KEY='your_api_key_here'
```

Or create a `.env` file in the project root:
```
DEEPGRAM_API_KEY=your_api_key_here
```

## Usage

### 🎙️ Voice-to-LLM Conversation (New!)

For natural AI conversations with voice:

1. **Test API Integration**:
```bash
python test_voice_llm_integration.py
```

2. **Test TTS Streaming**:
```bash
python test_elevenlabs_streaming.py
```

3. **Start Voice Conversation**:
```bash
python voice_llm_conversation.py
```

Features:
- Speak naturally and the system will intelligently decide when to respond
- Interrupt the AI by speaking while it's talking
- Maintains conversation context and history
- Real-time audio streaming for immediate responses

See `VOICE_LLM_GUIDE.md` for detailed usage instructions.

### 📝 Basic Real-time STT

Run the basic STT application:
```bash
python realtime_stt_diarization.py
```

The application will:
1. Connect to Deepgram's real-time API
2. Start capturing audio from your microphone
3. Display live transcription with speaker identification
4. Show both interim (temporary) and final (confirmed) results

### 👥 Speaker Identification

For persistent speaker recognition:
```bash
python improved_working_speaker_stt.py
```

This system:
- Remembers speakers across sessions
- Provides high accuracy with false positive prevention
- Allows manual speaker registration and identification
- Shows real-time similarity analysis

### Output Format

The output shows:
- **Timestamp**: When each utterance was spoken
- **Speaker ID**: Numbered speakers (0, 1, 2, etc.)
- **Color Coding**: Each speaker gets a unique color
- **Status**: Shows whether the result is "INTERIM" or "FINAL"

Example output:
```
[14:30:15] Speaker 0: Hello everyone, welcome to the meeting (FINAL)
[14:30:18] Speaker 1: Thanks for having me, excited to be here (INTERIM)
[14:30:20] Speaker 1: Thanks for having me, excited to be here today (FINAL)
```

## Configuration

The application uses the following Deepgram settings:
- **Model**: nova-3 (latest and most accurate)
- **Language**: English (US)
- **Smart Format**: Enabled for better punctuation
- **Interim Results**: Enabled for real-time feedback
- **Diarization**: Enabled for speaker identification
- **Voice Activity Detection**: Enabled for better speech detection

## Troubleshooting

### Common Issues

1. **"No module named 'pyaudio'"**: 
   - On macOS: `brew install portaudio && pip install pyaudio`
   - On Ubuntu/Debian: `sudo apt-get install libasound-dev portaudio19-dev && pip install pyaudio`
   - On Windows: `pip install pyaudio`

2. **"Permission denied" for microphone**:
   - Make sure your terminal/IDE has microphone permissions
   - On macOS: Go to System Preferences > Security & Privacy > Privacy > Microphone

3. **"API key not found"**:
   - Ensure your `DEEPGRAM_API_KEY` environment variable is set
   - Check that your API key is valid and active

4. **Connection issues**:
   - Check your internet connection
   - Verify your API key has sufficient credits
   - Ensure you're not hitting rate limits

## Advanced Usage

### Customizing the Model

You can modify the `LiveOptions` in the code to use different models or settings:

```python
options = LiveOptions(
    model="nova-3",        # or "nova", "enhanced", "base"
    language="en-US",      # or other supported languages
    smart_format=True,
    interim_results=True,
    diarize=True,          # Enable/disable diarization
    punctuate=True,        # Enable/disable punctuation
    # ... other options
)
```

### Supported Languages

Deepgram supports many languages. Some examples:
- `en-US` (English - US)
- `en-GB` (English - UK)
- `es` (Spanish)
- `fr` (French)
- `de` (German)
- `pt` (Portuguese)
- `ja` (Japanese)
- `ko` (Korean)
- `zh` (Chinese)

## API Credits

This application uses Deepgram's real-time API, which consumes credits based on usage. Monitor your usage at [https://console.deepgram.com/](https://console.deepgram.com/).

## License

This project is provided as-is for educational and development purposes.

## 📁 Available Scripts & Documentation

### 🆕 Modular Voice AI System (YAML Config)
- `setup_config.py` - **NEW!** Interactive configuration setup
- `unified_voice_conversation_config.py` - **LATEST!** YAML configuration-based system
- `config_loader.py` - YAML configuration loader
- `config.yaml.example` - Example configuration file
- `unified_voice_conversation.py` - Complete modular voice conversation system  
- `async_stt_module.py` - Modular STT with callback support
- `async_tts_module.py` - Modular TTS with interruption capabilities

### Legacy/Testing Scripts  
- `voice_llm_conversation.py` - Original integrated voice-to-LLM system
- `integrated_voice_llm_conversation.py` - Previous modular integration attempt
- `test_elevenlabs_streaming.py` - Test TTS streaming functionality
- `test_voice_llm_integration.py` - Validate all APIs work together
- `improved_working_speaker_stt.py` - Advanced speaker identification
- `realtime_stt_diarization.py` - Basic real-time STT with diarization
- `configurable_stt.py` - Customizable STT with dictionary support

### Documentation
- `VOICE_LLM_GUIDE.md` - Comprehensive voice-to-LLM system guide
- `FALSE_POSITIVE_FIX_GUIDE.md` - Speaker identification accuracy guide
- `SPEAKER_IDENTIFICATION_GUIDE.md` - Speaker system usage guide

### Configuration & Testing
- `custom_dictionary_config.py` - Custom vocabulary configuration
- `test_microphone.py` - Audio device testing
- `test_api_key.py` - API connectivity testing

## 🎯 Quick Start Recommendations

### 🆕 **Recommended: YAML Configuration System**
1. **Easy Setup**: Run `python setup_config.py` for interactive configuration
2. **Voice Chat**: Run `python unified_voice_conversation_config.py` 
3. **Test Modules**: Use `python test_modular_components.py` to test individual components
4. **API Testing**: Run `python test_voice_llm_integration.py` to validate all APIs

**Alternative Setup:**
- Manual: Copy `config.yaml.example` to `config.yaml` and edit with your API keys

### Legacy Options  
4. **Original System**: Try `voice_llm_conversation.py` for the original implementation
5. **Speaker Recognition**: Use `improved_working_speaker_stt.py` for persistent identification
6. **Basic STT**: Start with `realtime_stt_diarization.py` for simple transcription

## 🏗️ Modular Architecture

The new modular system separates concerns into reusable components:

### 1. **STT Module** (`async_stt_module.py`)
```python
from async_stt_module import AsyncSTTStreamer, STTConfig, STTEventType

# Setup
config = STTConfig(api_key="your_deepgram_key")
stt = AsyncSTTStreamer(config)

# Register callbacks
stt.on(STTEventType.UTTERANCE_COMPLETE, handle_utterance)
stt.on(STTEventType.SPEECH_STARTED, handle_speech_start)

# Start listening
await stt.start_listening()
```

### 2. **TTS Module** (`async_tts_module.py`)
```python
from async_tts_module import AsyncTTSStreamer, TTSConfig

# Setup
config = TTSConfig(api_key="your_elevenlabs_key", voice_id="voice_id")
tts = AsyncTTSStreamer(config)

# Speak text
result = await tts.speak_text("Hello world!")

# Speak streaming text (like from LLM)
result = await tts.speak_stream(your_async_generator)

# Interrupt anytime
await tts.stop()
```

### 3. **Unified System** (`unified_voice_conversation.py`)
Complete voice conversation system that integrates both modules with intelligent conversation management.

```python
from unified_voice_conversation import UnifiedVoiceConversation, ConversationConfig

# Setup
config = ConversationConfig(
    deepgram_api_key="your_key",
    elevenlabs_api_key="your_key", 
    openai_api_key="your_key"
)

conversation = UnifiedVoiceConversation(config)
await conversation.start_conversation()
```

### 🔧 **Key Benefits of Modular Design:**
- **Reusable Components**: Use STT and TTS modules in other projects
- **Easy Configuration**: Clean configuration classes for all settings
- **Event-Driven**: Callback system for flexible interaction handling
- **Interruption Support**: Full control over when to stop TTS
- **Resource Management**: Proper async cleanup and error handling

## 📋 YAML Configuration System

The new YAML configuration system makes it easy to manage API keys and settings without modifying code:

### **Setup:**
```bash
# 1. Copy the example config
cp config.yaml.example config.yaml

# 2. Edit with your API keys
# (config.yaml is in .gitignore so it won't be committed)
```

### **Example config.yaml:**
```yaml
# API Keys
api_keys:
  deepgram: "your_deepgram_api_key_here"
  elevenlabs: "your_elevenlabs_api_key_here"
  openai: "your_openai_api_key_here"

# Voice Settings
voice:
  id: "T2KZm9rWPG5TgXTyjt7E"  # Catalyst voice

# Conversation timing
conversation:
  pause_threshold: 2.0        # seconds before LLM submission
  min_words_for_submission: 3 # minimum words required
  interruption_confidence: 0.8 # confidence for interruption

# STT/TTS models and settings
stt:
  model: "nova-3"
  language: "en-US"
  
tts:
  model_id: "eleven_multilingual_v2"
  speed: 1.0
  stability: 0.5
```

### **Usage:**
```bash
# Run the YAML-configured system
python unified_voice_conversation_config.py

# Test configuration loading
python config_loader.py

# Create a new example config
python config_loader.py create-example
```

### **Configuration Benefits:**
- 🔐 **Secure**: API keys separate from code
- 🎛️ **Customizable**: All settings in one place
- 🔄 **Flexible**: Easy to test different configurations
- 📁 **Organized**: Clean separation of concerns
- 🚫 **Safe**: config.yaml automatically ignored by git

## 🔧 Troubleshooting

### **Event Loop Warnings**
If you see warnings like `RuntimeWarning: coroutine was never awaited` or `no running event loop`, these have been fixed in the latest version:

**Fixed Issues:**
- ✅ Async event handling in STT callbacks
- ✅ Graceful shutdown and cleanup  
- ✅ Signal handling for Ctrl+C interruption
- ✅ Thread-safe event scheduling

**Test the fixes:**
```bash
python test_async_fixes.py
```

### **Common Issues**

**1. API Key Errors:**
- Check that your `config.yaml` contains valid API keys
- Run `python config_loader.py` to validate configuration
- Use `python setup_config.py` for interactive setup

**2. Audio Device Issues:**
- Ensure microphone permissions are granted
- Check `audio.input_device_index` in config for specific devices
- Run `python test_voice_llm_integration.py` to test audio

**3. Connection Timeouts:**
- Check internet connectivity
- Verify API key permissions (Deepgram Nova tier, ElevenLabs access)
- Try different models in configuration

**4. Import Errors:**
- Run `pip install -r requirements.txt` to install dependencies
- Activate virtual environment: `source venv/bin/activate`

## Support

For issues with:
- **Voice-to-LLM System**: Check `VOICE_LLM_GUIDE.md` for detailed troubleshooting
- **Speaker Identification**: See `FALSE_POSITIVE_FIX_GUIDE.md` for accuracy issues
- **Deepgram API**: Check [Deepgram's documentation](https://developers.deepgram.com/)
- **ElevenLabs API**: Visit [ElevenLabs documentation](https://docs.elevenlabs.io/)
- **OpenAI API**: Check [OpenAI documentation](https://platform.openai.com/docs/)
- **General Issues**: Review error messages and troubleshooting sections in the guides 