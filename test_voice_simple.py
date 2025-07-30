#!/usr/bin/env python3
"""Simple test of voice system."""
import asyncio
import sys
import os
import subprocess

# Set preset
os.environ['VOICE_AI_PRESET'] = 'opus_and_36'

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
    """Run voice conversation."""
    # Kill any existing processes first
    kill_existing_processes()
    
    print("🚀 Starting Voice Conversation")
    print("=" * 50)
    
    from config_loader import load_config
    from unified_voice_conversation_config import UnifiedVoiceConversation
    
    try:
        # Load configuration
        print("📁 Loading configuration...")
        config = load_config()
        print("✅ Configuration loaded successfully!")
        
        # Create conversation system
        print("🔧 Creating conversation system...")
        conversation = UnifiedVoiceConversation(config)
        print("✅ Conversation system created!")
        
        # Start conversation
        print("🎙️ Starting conversation...")
        success = await conversation.start_conversation()
        if not success:
            print("❌ Failed to start conversation")
            return
            
        # Keep running until interrupted
        print("🎯 Conversation is active. Press Ctrl+C to exit.")
        while conversation.state.is_active:
            await asyncio.sleep(0.5)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)