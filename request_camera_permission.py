#!/usr/bin/env python3
"""
Request Camera Permission on macOS
This script helps trigger the camera permission dialog.
"""

import cv2
import time
import os
import sys

def request_camera_permission():
    """Request camera permission by attempting to access the camera."""
    print("📷 Requesting camera permission...")
    print("\n⚠️  IMPORTANT: Look for a permission dialog!")
    print("   If you don't see one, check System Preferences:")
    print("   Security & Privacy → Privacy → Camera")
    print("\n" + "="*50 + "\n")
    
    # Try each camera index
    for i in range(3):
        print(f"Attempting to access camera {i}...")
        
        try:
            cap = cv2.VideoCapture(i)
            
            # Try to read a frame
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Camera {i} is accessible!")
                    cap.release()
                    return True
                else:
                    print(f"❌ Camera {i} opened but couldn't read frame")
            else:
                print(f"❌ Camera {i} failed to open")
                
            cap.release()
            
        except Exception as e:
            print(f"❌ Error accessing camera {i}: {e}")
        
        time.sleep(1)
    
    return False

def check_terminal_permission():
    """Check if we're running in Terminal and provide guidance."""
    if 'TERM_PROGRAM' in os.environ:
        term_program = os.environ.get('TERM_PROGRAM', 'Unknown')
        print(f"\n📱 Running in: {term_program}")
        
        if term_program == 'Apple_Terminal':
            print("✅ Running in macOS Terminal")
        elif term_program == 'iTerm.app':
            print("✅ Running in iTerm2")
        else:
            print(f"⚠️  Running in {term_program}")
    
    print("\n📋 Next steps:")
    print("1. Open System Preferences")
    print("2. Go to Security & Privacy → Privacy → Camera")
    print("3. Look for your terminal app in the list")
    print("4. Check the box next to it to grant permission")
    print("5. You may need to restart your terminal")

def main():
    print("🔐 macOS Camera Permission Helper")
    print("=" * 50)
    
    # Check if we're on macOS
    if sys.platform != 'darwin':
        print("⚠️  This script is for macOS only")
        return
    
    # Try to access camera
    success = request_camera_permission()
    
    if not success:
        print("\n❌ Camera access failed!")
        check_terminal_permission()
        
        print("\n💡 Alternative solutions:")
        print("\n1. Run from VS Code or PyCharm (they usually handle permissions better)")
        print("\n2. Create a standalone app with py2app:")
        print("   pip install py2app")
        print("   py2applet --make-setup list_cameras.py")
        print("   python setup.py py2app")
        print("   open dist/list_cameras.app")
        
        print("\n3. Use system Python instead of venv:")
        print("   /usr/bin/python3 list_cameras.py")
        
        print("\n4. Reset camera permissions:")
        print("   tccutil reset Camera")
        print("   (This will reset ALL camera permissions)")
    else:
        print("\n✅ Camera access successful!")
        print("You can now run the camera utilities.")

if __name__ == "__main__":
    main()