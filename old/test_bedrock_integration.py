#!/usr/bin/env python3
"""
Test script to verify Bedrock integration
"""

import asyncio
import os
from anthropic import AnthropicBedrock, AsyncAnthropicBedrock

async def test_bedrock_connection():
    # Check if AWS credentials are set
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-west-2")
    
    if not aws_access_key or not aws_secret_key:
        print("❌ AWS credentials not found in environment variables")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
    
    print(f"✅ AWS credentials found, region: {aws_region}")
    
    try:
        # Test sync client
        print("\nTesting sync AnthropicBedrock client...")
        sync_client = AnthropicBedrock(
            aws_region=aws_region,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key
        )
        print("✅ Sync client created successfully")
        
        # Test async client
        print("\nTesting async AnthropicBedrock client...")
        async_client = AsyncAnthropicBedrock(
            aws_region=aws_region,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key
        )
        print("✅ Async client created successfully")
        
        # Test a simple API call
        print("\nTesting API call...")
        response = await async_client.messages.create(
            model="anthropic.claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Say 'Hello Bedrock!' and nothing else."}],
            max_tokens=20,
            temperature=0
        )
        
        print(f"✅ API call successful!")
        print(f"Response: {response.content[0].text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_bedrock_connection())