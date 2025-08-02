#!/usr/bin/env python3
"""
Test script to demonstrate OSC blank output message for TouchDesigner laser rendering
"""

from pythonosc import udp_client
import time

def test_osc_blank_message():
    """Test sending blank output OSC message to TouchDesigner"""
    
    # TouchDesigner OSC settings
    host = "127.0.0.1"  # Change to your TouchDesigner machine IP
    port = 7000         # Default OSC port
    
    # Create OSC client
    client = udp_client.SimpleUDPClient(host, port)
    
    print(f"📡 Testing OSC blank output to {host}:{port}")
    
    # Send blank output message (empty string for laser)
    address = "/character/blank"
    data = ""
    
    print(f"🚀 Sending: {address} with data: '{data}'")
    client.send_message(address, data)
    
    print("✅ Blank output OSC message sent!")
    print("\nIn TouchDesigner, you can receive this with:")
    print(f"   OSC In CHOP -> Address: {address}")
    print(f"   Data will be empty string: '{data}'")

if __name__ == "__main__":
    try:
        test_osc_blank_message()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure TouchDesigner is running and listening on OSC port 7000")