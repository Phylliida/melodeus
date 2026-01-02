#!/usr/bin/env python3
"""
Test script to verify OSC functionality
"""

import time
import sys

def test_osc_client():
    """Test OSC client connection and message sending."""
    try:
        from pythonosc import udp_client
        print("✅ python-osc library is installed")
    except ImportError:
        print("❌ python-osc library not found!")
        print("   Install with: pip install python-osc")
        return False
    
    # Test configuration
    host = "127.0.0.1"
    port = 7000
    
    print(f"\n📡 Testing OSC client to {host}:{port}")
    
    try:
        # Create client
        client = udp_client.SimpleUDPClient(host, port)
        print("✅ OSC client created successfully")
        
        # Send test messages
        test_messages = [
            ("/test", "Hello OSC"),
            ("/character/speaking/start", "TestCharacter"),
            ("/character/speaking/stop", "TestCharacter"),
        ]
        
        for address, data in test_messages:
            print(f"\n📤 Sending: {address} '{data}'")
            client.send_message(address, data)
            print("   ✅ Sent successfully")
            time.sleep(0.5)
        
        print("\n✅ All test messages sent successfully!")
        print("\n💡 If you didn't see these messages on your OSC receiver:")
        print("   1. Make sure an OSC receiver is listening on port 7000")
        print("   2. Check firewall settings")
        print("   3. Try the test receiver script from OSC_SETUP_GUIDE.md")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during OSC test: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test loading OSC configuration from config.yaml."""
    print("\n📋 Testing config loading...")
    
    try:
        from config_loader import VoiceAIConfig
        config = VoiceAIConfig.load()
        
        if config.osc:
            print("✅ OSC config loaded:")
            print(f"   Enabled: {config.osc.enabled}")
            print(f"   Host: {config.osc.host}")
            print(f"   Port: {config.osc.port}")
            print(f"   Start address: {config.osc.speaking_start_address}")
            print(f"   Stop address: {config.osc.speaking_stop_address}")
        else:
            print("❌ No OSC configuration found in config.yaml")
            
    except Exception as e:
        print(f"❌ Error loading config: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 OSC Test Tool")
    print("=" * 50)
    
    # Test OSC functionality
    osc_ok = test_osc_client()
    
    # Test config loading
    test_config_loading()
    
    if osc_ok:
        print("\n✅ OSC test completed successfully!")
    else:
        print("\n❌ OSC test failed - check the errors above")
        sys.exit(1)