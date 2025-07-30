#!/usr/bin/env python3
"""Test if images are being properly included in conversation."""

import json
from pathlib import Path
import sys

def check_for_images_in_logs():
    """Check recent log files for image content."""
    log_dir = Path("llm_logs")
    
    # Get most recent request files
    request_files = sorted(
        [f for f in log_dir.glob("*_request.json") if not "director" in f.name],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:5]  # Last 5 non-director requests
    
    print("Checking recent request logs for images...")
    print("=" * 50)
    
    for request_file in request_files:
        print(f"\n📄 {request_file.name}")
        
        with open(request_file, 'r') as f:
            data = json.load(f)
            
        # Check messages for image content
        has_images = False
        image_count = 0
        
        messages = data.get("messages", [])
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            
            if isinstance(content, list):
                # This is structured content - check for images
                for item in content:
                    if item.get("type") == "image":
                        has_images = True
                        image_count += 1
                        print(f"  ✅ Found image in message {i}")
                        print(f"     Media type: {item.get('source', {}).get('media_type', 'unknown')}")
                        
                        # Check if base64 data exists
                        if item.get('source', {}).get('data'):
                            data_len = len(item['source']['data'])
                            print(f"     Base64 data length: {data_len} chars")
                        else:
                            print(f"     ❌ No base64 data found!")
                    elif item.get("type") == "text":
                        text_preview = item.get("text", "")[:50]
                        print(f"  📝 Text: {text_preview}...")
        
        if not has_images:
            print(f"  ❌ No images found in {len(messages)} messages")
        else:
            print(f"  📸 Total images: {image_count}")
            
        # Also check if prefill format is being used
        if len(messages) > 0 and messages[-1].get("role") == "assistant":
            print(f"  📋 Format: Prefill (assistant message at end)")
        else:
            print(f"  📋 Format: Regular chat")

if __name__ == "__main__":
    check_for_images_in_logs()