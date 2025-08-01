#!/usr/bin/env python3
"""
Test the actual speexdsp API
"""

import numpy as np
from speexdsp import EchoCanceller
import inspect

# Create echo canceller
ec = EchoCanceller.create(256, 2048, 16000)

# Check the process method signature
print("🔍 Inspecting EchoCanceller.process method:")
print(f"Method: {ec.process}")

# Try to get help
try:
    help(ec.process)
except:
    pass

# Check attributes
print("\n📊 EchoCanceller attributes:")
for attr in dir(ec):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Check if there's documentation
print("\n📄 Module documentation:")
print(EchoCanceller.__doc__)

# Try different ways to call it
print("\n🧪 Testing different call patterns:")

# Create test data
near_data = np.random.randint(-1000, 1000, 256, dtype=np.int16)
far_data = np.random.randint(-1000, 1000, 256, dtype=np.int16)

# Test 1: Direct numpy arrays
try:
    result = ec.process(near_data, far_data)
    print("✅ Direct numpy arrays: Success!")
    print(f"   Result type: {type(result)}, shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
except Exception as e:
    print(f"❌ Direct numpy arrays: {e}")

# Test 2: As bytes
try:
    near_bytes = near_data.tobytes()
    far_bytes = far_data.tobytes()
    result = ec.process(near_bytes, far_bytes)
    print("✅ As bytes: Success!")
except Exception as e:
    print(f"❌ As bytes: {e}")

# Test 3: Check if it needs strings (unlikely but testing)
try:
    # The error mentioned std::string, so let's check
    result = ec.process(near_data.tobytes().decode('latin-1'), far_data.tobytes().decode('latin-1'))
    print("✅ As strings: Success!")
except Exception as e:
    print(f"❌ As strings: {e}")

# Test 4: Check the Python binding source if available
try:
    import speexdsp
    print(f"\n📁 speexdsp module location: {speexdsp.__file__}")
except:
    pass