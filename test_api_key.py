#!/usr/bin/env python3
"""
Simple script to test Deepgram API key validity
"""

import asyncio
from deepgram import DeepgramClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

async def test_api_key():
    """Test if the API key is valid with a simple request."""
    
    api_key = os.getenv('DEEPGRAM_API_KEY')
    if not api_key:
        print("❌ No API key found in environment variables")
        return False
    
    print(f"🔑 Testing API key: {api_key[:8]}...")
    
    try:
        # Create client
        deepgram = DeepgramClient(api_key)
        
        # Test with a simple pre-recorded audio API call
        # This is less likely to have permission issues than live streaming
        response = await deepgram.listen.asyncrest.v("1").transcribe_url(
            {
                "url": "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"
            },
            {
                "model": "nova-3",
                "smart_format": True,
            }
        )
        
        print("✅ API key is valid!")
        print(f"📝 Test transcription: {response.results.channels[0].alternatives[0].transcript}")
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_api_key()) 