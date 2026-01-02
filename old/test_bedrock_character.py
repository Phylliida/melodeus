#!/usr/bin/env python3
"""
Test Bedrock integration with character-specific provider
"""

import asyncio
import sys
import os

async def test_bedrock_character():
    """Test loading a character with Bedrock provider"""
    
    # Check if AWS credentials are available
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not aws_key or not aws_secret:
        print("❌ AWS credentials not found in environment")
        print("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        return False
    
    try:
        # Import the voice conversation module
        from config_loader import ConfigLoader
        from unified_voice_conversation_config import UnifiedVoiceConversation
        
        # Create a test config with a Bedrock character
        test_config_data = {
            'api_keys': {
                'deepgram': 'test_key',
                'elevenlabs': 'test_key', 
                'openai': 'test_key',
                'aws_access_key_id': aws_key,
                'aws_secret_access_key': aws_secret,
                'aws_region': 'us-west-2'
            },
            'conversation': {
                'llm_provider': 'anthropic',  # Default is anthropic
                'characters_config': {
                    'TestBedrock': {
                        'llm_provider': 'bedrock',  # But this character uses Bedrock
                        'llm_model': 'claude-3-5-sonnet-20241022',
                        'voice_id': 'test_voice',
                        'max_tokens': 100,
                        'temperature': 0.7
                    }
                }
            },
            'voice': {'id': 'test_voice'},
            'stt': {'language': 'en-US'},
            'tts': {},
            'audio': {}
        }
        
        print("🔧 Testing Bedrock character initialization...")
        
        # Load config 
        config = ConfigLoader._create_config(test_config_data)
        print("✅ Config loaded successfully")
        
        # Initialize conversation system - this should initialize Bedrock client
        # even though the default provider is anthropic
        conversation = UnifiedVoiceConversation(config)
        print("✅ UnifiedVoiceConversation initialized")
        
        # Check if Bedrock client was initialized
        if conversation.async_bedrock_client is not None:
            print("✅ Bedrock async client initialized successfully!")
            print(f"   Region: {config.conversation.aws_region}")
            return True
        else:
            print("❌ Bedrock async client is None")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Bedrock Character Integration")
    print("=" * 50)
    
    success = asyncio.run(test_bedrock_character())
    
    if success:
        print("\n✅ Bedrock character test passed!")
    else:
        print("\n❌ Bedrock character test failed!")
    
    sys.exit(0 if success else 1)