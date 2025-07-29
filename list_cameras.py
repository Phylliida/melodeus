#!/usr/bin/env python3
"""
List Available Cameras
Utility to discover and test available camera devices on your system.
"""

import cv2
import platform
import time
from typing import List, Dict, Any

def get_camera_name_macos(index: int) -> str:
    """Get camera name on macOS using system_profiler."""
    try:
        import subprocess
        import json
        
        # Get camera info from system_profiler
        result = subprocess.run(
            ['system_profiler', 'SPCameraDataType', '-json'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            cameras = data.get('SPCameraDataType', [])
            
            # Try to match index to camera
            if index < len(cameras):
                return cameras[index].get('_name', f'Camera {index}')
                
    except Exception:
        pass
    
    return f"Camera {index}"

def test_camera(index: int, timeout: float = 2.0) -> Dict[str, Any]:
    """Test if a camera index is valid and get its properties."""
    info = {
        "index": index,
        "available": False,
        "name": None,
        "resolution": None,
        "fps": None,
        "backend": None
    }
    
    try:
        # Try to open the camera
        cap = cv2.VideoCapture(index)
        
        # Wait a bit for camera to initialize
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            ret, frame = cap.read()
            if ret and frame is not None:
                info["available"] = True
                info["resolution"] = {
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                }
                info["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
                info["backend"] = cap.getBackendName()
                
                # Get camera name based on platform
                if platform.system() == "Darwin":  # macOS
                    info["name"] = get_camera_name_macos(index)
                else:
                    info["name"] = f"Camera {index}"
                
                break
            time.sleep(0.1)
        
        cap.release()
        
    except Exception as e:
        info["error"] = str(e)
    
    return info

def list_available_cameras(max_index: int = 10) -> List[Dict[str, Any]]:
    """List all available cameras up to max_index."""
    cameras = []
    
    print("🔍 Searching for cameras...")
    print("This may take a moment as we test each device...\n")
    
    for i in range(max_index):
        print(f"Testing camera index {i}...", end=" ", flush=True)
        info = test_camera(i)
        
        if info["available"]:
            cameras.append(info)
            print("✅ Found!")
        else:
            print("❌ Not available")
    
    return cameras

def preview_camera(index: int, duration: int = 5):
    """Show a preview window for the specified camera."""
    print(f"\n📸 Opening preview for camera {index}...")
    print(f"Press 'q' to close the preview window or wait {duration} seconds")
    
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    
    # Set resolution for preview
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    window_name = f'Camera {index} Preview'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Add text overlay
        cv2.putText(frame, f'Camera Index: {index}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, 'Press "q" to close', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, frame)
        
        # Check for 'q' key or timeout
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if time.time() - start_time > duration:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Preview closed")

def main():
    """Main function to list cameras and optionally preview them."""
    print("📷 Camera Detection Utility")
    print("=" * 50)
    
    # List available cameras
    cameras = list_available_cameras()
    
    if not cameras:
        print("\n❌ No cameras found!")
        print("\nPossible reasons:")
        print("- No cameras connected")
        print("- Camera permissions not granted")
        print("- Camera drivers not installed")
        return
    
    # Display found cameras
    print(f"\n✅ Found {len(cameras)} camera(s):\n")
    
    for cam in cameras:
        print(f"📷 Camera Index: {cam['index']}")
        print(f"   Name: {cam['name']}")
        print(f"   Resolution: {cam['resolution']['width']}x{cam['resolution']['height']}")
        print(f"   FPS: {cam['fps']}")
        print(f"   Backend: {cam['backend']}")
        print()
    
    # Ask if user wants to preview
    if len(cameras) > 0:
        print("\nWould you like to preview a camera?")
        print("Enter the camera index to preview, or press Enter to skip: ", end="")
        
        try:
            user_input = input().strip()
            if user_input:
                index = int(user_input)
                if any(cam['index'] == index for cam in cameras):
                    preview_camera(index)
                else:
                    print(f"❌ Camera index {index} not found")
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Skipping preview")
    
    # Show configuration example
    print("\n📝 To use a camera in your configuration, add:")
    print("```yaml")
    print("camera:")
    print("  enabled: true")
    print(f"  device_id: {cameras[0]['index']}  # {cameras[0]['name']}")
    print("  resolution: [640, 480]")
    print("  capture_on_speech: true")
    print("```")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()