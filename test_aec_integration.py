#!/usr/bin/env python3
"""
Test echo cancellation integration with the voice system components
"""

import sys
import os

print("🔍 Testing Echo Cancellation Integration")
print("=" * 50)

# Test 1: Import speexdsp
try:
    import speexdsp
    print("✅ speexdsp module imported")
except ImportError as e:
    print(f"❌ Failed to import speexdsp: {e}")
    sys.exit(1)

# Test 2: Import echo cancellation module
try:
    from echo_cancellation import EchoCancellationProcessor
    print("✅ EchoCancellationProcessor imported")
except ImportError as e:
    print(f"❌ Failed to import EchoCancellationProcessor: {e}")
    sys.exit(1)

# Test 3: Load config
try:
    from config_loader import load_config
    config = load_config(preset="opus_and_36")
    print("✅ Configuration loaded")
    print(f"   Echo cancellation enabled: {config.conversation.enable_echo_cancellation}")
    print(f"   Frame size: {config.conversation.aec_frame_size}")
    print(f"   Filter length: {config.conversation.aec_filter_length}")
    print(f"   Delay: {config.conversation.aec_delay_ms}ms")
except Exception as e:
    print(f"❌ Failed to load config: {e}")
    sys.exit(1)

# Test 4: Check STT config
print("\n📊 Checking STT configuration:")
print(f"   STT sample rate: {config.stt.sample_rate}")
print(f"   STT has enable_echo_cancellation: {hasattr(config.stt, 'enable_echo_cancellation')}")

# Test 5: Test echo canceller initialization
try:
    echo_canceller = EchoCancellationProcessor(
        frame_size=config.conversation.aec_frame_size,
        filter_length=config.conversation.aec_filter_length,
        sample_rate=config.stt.sample_rate,
        reference_delay_ms=config.conversation.aec_delay_ms
    )
    print("\n✅ Echo canceller initialized successfully!")
except Exception as e:
    print(f"\n❌ Failed to initialize echo canceller: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check if echo cancellation is available in async_stt_module
try:
    import async_stt_module
    print(f"\n📊 async_stt_module checks:")
    print(f"   ECHO_CANCELLATION_AVAILABLE: {async_stt_module.ECHO_CANCELLATION_AVAILABLE}")
except Exception as e:
    print(f"\n❌ Error checking async_stt_module: {e}")

print("\n✅ All tests completed!")
 