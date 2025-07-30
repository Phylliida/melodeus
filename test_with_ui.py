#!/usr/bin/env python3
"""Test script to run voice conversation with UI."""
import asyncio
import webbrowser
import sys
import os

async def main():
    """Run voice conversation with UI."""
    print("🚀 Starting Voice Conversation with UI")
    print("=" * 50)
    
    # Import after print to show startup
    from unified_voice_conversation_config import main as voice_main
    
    # Open UI in browser after a short delay
    async def open_ui():
        await asyncio.sleep(2)  # Wait for WebSocket server to start
        ui_path = os.path.abspath("ui_client.html")
        print(f"\n🌐 Opening UI at: file://{ui_path}")
        webbrowser.open(f"file://{ui_path}")
    
    # Start UI opener task
    asyncio.create_task(open_ui())
    
    # Run voice conversation
    await voice_main()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)