#!/usr/bin/env python3
"""Test script to run voice conversation with UI."""
import asyncio
import webbrowser
import sys
import os
import subprocess

def kill_existing_processes():
    """Kill any existing voice system or WebSocket server processes."""
    print("🔍 Checking for existing processes...")
    
    # Check for processes using port 8765 (WebSocket server)
    try:
        result = subprocess.run(['lsof', '-ti', ':8765'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                print(f"   🛑 Killing process {pid} using port 8765")
                subprocess.run(['kill', '-9', pid])
            print("   ✅ Cleared port 8765")
    except Exception as e:
        print(f"   ⚠️  Could not check port 8765: {e}")
    
    # Check for running Python voice processes
    try:
        result = subprocess.run(['pgrep', '-f', 'python.*unified_voice'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                # Don't kill ourselves
                if pid != str(os.getpid()):
                    print(f"   🛑 Killing voice system process {pid}")
                    subprocess.run(['kill', '-9', pid])
            print("   ✅ Cleared existing voice processes")
    except Exception as e:
        print(f"   ⚠️  Could not check voice processes: {e}")
    
    print("✅ Ready to start fresh")
    print()

async def main():
    """Run voice conversation with UI."""
    # Kill any existing processes first
    kill_existing_processes()
    
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