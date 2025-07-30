#!/bin/bash

echo "🔧 macOS Camera Permission Fix Script"
echo "===================================="
echo

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

echo "📋 Current camera permissions:"
echo

# Check Terminal permission
if [[ -n "$TERM_PROGRAM" ]]; then
    echo "Running in: $TERM_PROGRAM"
fi

# Method 1: Open System Preferences directly to Camera settings
echo
echo "🔧 Method 1: Opening System Preferences..."
echo "   Please check the box next to Terminal (or your terminal app)"
echo
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"

echo "⏳ Waiting for you to grant permissions..."
echo "   Press Enter after granting permission in System Preferences"
read -r

# Method 2: Test with system Python
echo
echo "🔧 Method 2: Testing with system Python..."
echo

/usr/bin/python3 -c "
import cv2
print('Testing camera access...')
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, _ = cap.read()
    if ret:
        print('✅ Camera access working!')
    else:
        print('❌ Camera opened but cannot read frames')
    cap.release()
else:
    print('❌ Cannot open camera')
"

# Method 3: Create Info.plist for Python
echo
echo "🔧 Method 3: Creating Info.plist for better permission handling..."
echo

cat > /tmp/PythonCamera.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSCameraUsageDescription</key>
    <string>Python needs access to your camera for the voice assistant to see and respond to visual input.</string>
    <key>NSCameraUseContinuityCameraDeviceType</key>
    <true/>
</dict>
</plist>
EOF

echo "✅ Created Info.plist at /tmp/PythonCamera.plist"
echo

# Final instructions
echo "📋 Final steps to fix camera access:"
echo
echo "1. ✅ Camera permissions have been opened in System Preferences"
echo "   - Make sure Terminal (or iTerm2/VS Code) is checked"
echo
echo "2. 🔄 Restart your terminal application completely"
echo "   - Quit Terminal (Cmd+Q), then reopen"
echo
echo "3. 🧪 Test camera access:"
echo "   python3 list_cameras.py"
echo
echo "4. 🚀 If still not working, try running from VS Code:"
echo "   - VS Code usually handles permissions better"
echo "   - Install Python extension and run from there"
echo
echo "5. 🔧 Nuclear option - reset all camera permissions:"
echo "   sudo tccutil reset Camera"
echo "   (This will reset ALL app camera permissions)"
echo

# Make this script executable
chmod +x "$0"