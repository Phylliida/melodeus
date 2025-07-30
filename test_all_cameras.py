#!/usr/bin/env python3
"""
Test All Cameras - Interactive Camera Selector
Shows live previews of all available cameras for easy identification.
"""

import cv2
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple
import platform
import subprocess
import json

class CameraManager:
    """Manages multiple camera feeds simultaneously."""
    
    def __init__(self):
        self.cameras: Dict[int, cv2.VideoCapture] = {}
        self.frames: Dict[int, Optional[np.ndarray]] = {}
        self.camera_info: Dict[int, Dict[str, any]] = {}
        self.threads: Dict[int, threading.Thread] = {}
        self.running = False
        
    def discover_cameras(self, max_cameras: int = 10) -> List[int]:
        """Discover available cameras."""
        available = []
        
        print("🔍 Discovering cameras...")
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available.append(i)
                    
                    # Get camera info
                    self.camera_info[i] = {
                        "index": i,
                        "name": self._get_camera_name(i),
                        "resolution": (
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        ),
                        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
                        "backend": cap.getBackendName()
                    }
                cap.release()
        
        return available
    
    def _get_camera_name(self, index: int) -> str:
        """Get camera name, especially on macOS."""
        if platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ['system_profiler', 'SPCameraDataType', '-json'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    cameras = data.get('SPCameraDataType', [])
                    
                    if index < len(cameras):
                        return cameras[index].get('_name', f'Camera {index}')
            except:
                pass
        
        return f"Camera {index}"
    
    def start_camera(self, index: int):
        """Start capturing from a specific camera."""
        if index not in self.cameras:
            self.cameras[index] = cv2.VideoCapture(index)
            self.frames[index] = None
            
            # Start capture thread
            thread = threading.Thread(target=self._capture_loop, args=(index,), daemon=True)
            self.threads[index] = thread
            thread.start()
    
    def _capture_loop(self, index: int):
        """Continuous capture loop for a camera."""
        cap = self.cameras[index]
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.frames[index] = frame
            else:
                time.sleep(0.1)
    
    def start_all(self, indices: List[int]):
        """Start all cameras."""
        self.running = True
        for idx in indices:
            self.start_camera(idx)
    
    def stop_all(self):
        """Stop all cameras."""
        self.running = False
        
        # Wait for threads to stop
        for thread in self.threads.values():
            thread.join(timeout=1.0)
        
        # Release cameras
        for cap in self.cameras.values():
            cap.release()
        
        self.cameras.clear()
        self.frames.clear()
        self.threads.clear()
    
    def create_grid_view(self, max_width: int = 1200) -> Optional[np.ndarray]:
        """Create a grid view of all camera feeds."""
        active_frames = [(idx, frame) for idx, frame in self.frames.items() if frame is not None]
        
        if not active_frames:
            return None
        
        # Calculate grid dimensions
        n_cameras = len(active_frames)
        cols = min(3, n_cameras)  # Max 3 columns
        rows = (n_cameras + cols - 1) // cols
        
        # Calculate individual frame size
        frame_width = max_width // cols
        frame_height = int(frame_width * 0.75)  # Assume 4:3 aspect ratio
        
        # Create grid
        grid_height = frame_height * rows
        grid_width = frame_width * cols
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, (idx, frame) in enumerate(active_frames):
            row = i // cols
            col = i % cols
            
            # Resize frame
            resized = cv2.resize(frame, (frame_width, frame_height))
            
            # Add text overlay
            info = self.camera_info.get(idx, {})
            text = f"Camera {idx}: {info.get('name', 'Unknown')}"
            cv2.putText(resized, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            res_text = f"{info.get('resolution', ['?', '?'])[0]}x{info.get('resolution', ['?', '?'])[1]} @ {info.get('fps', '?')}fps"
            cv2.putText(resized, res_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add border
            cv2.rectangle(resized, (0, 0), (frame_width-1, frame_height-1), (0, 255, 0), 2)
            
            # Place in grid
            y1 = row * frame_height
            y2 = (row + 1) * frame_height
            x1 = col * frame_width
            x2 = (col + 1) * frame_width
            grid[y1:y2, x1:x2] = resized
        
        return grid

def main():
    """Main function to test all cameras."""
    print("📷 Multi-Camera Test Utility")
    print("=" * 50)
    
    manager = CameraManager()
    
    # Discover cameras
    available = manager.discover_cameras()
    
    if not available:
        print("❌ No cameras found!")
        print("\nTroubleshooting:")
        print("1. Check camera connections")
        print("2. Grant camera permissions to Terminal/Python")
        print("3. Install camera drivers if needed")
        return
    
    print(f"\n✅ Found {len(available)} camera(s):")
    for idx in available:
        info = manager.camera_info[idx]
        print(f"\n📷 Camera {idx}: {info['name']}")
        print(f"   Resolution: {info['resolution'][0]}x{info['resolution'][1]}")
        print(f"   FPS: {info['fps']}")
        print(f"   Backend: {info['backend']}")
    
    # Ask user which cameras to test
    print("\n" + "=" * 50)
    print("Options:")
    print("1. Test all cameras simultaneously")
    print("2. Test specific camera")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Test all cameras
        print(f"\n🎬 Starting preview of all {len(available)} cameras...")
        print("Press 'q' to quit, 's' to save a screenshot")
        
        manager.start_all(available)
        
        window_name = "All Cameras"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        screenshot_count = 0
        
        while True:
            grid = manager.create_grid_view()
            
            if grid is not None:
                cv2.imshow(window_name, grid)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"camera_grid_screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(filename, grid)
                    print(f"💾 Saved screenshot: {filename}")
                    screenshot_count += 1
            else:
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        manager.stop_all()
        
    elif choice == "2":
        # Test specific camera
        try:
            idx = int(input("Enter camera index: "))
            if idx in available:
                print(f"\n🎬 Starting preview of camera {idx}...")
                print("Press 'q' to quit")
                
                cap = cv2.VideoCapture(idx)
                window_name = f"Camera {idx}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                
                while True:
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow(window_name, frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        time.sleep(0.1)
                
                cap.release()
                cv2.destroyAllWindows()
            else:
                print(f"❌ Camera index {idx} not available")
        except ValueError:
            print("❌ Invalid input")
    
    # Show configuration snippet
    if available:
        print("\n" + "=" * 50)
        print("📝 Configuration snippet for your YAML file:")
        print("\n```yaml")
        print("camera:")
        print("  enabled: true")
        print(f"  device_id: {available[0]}  # {manager.camera_info[available[0]]['name']}")
        print("  resolution: [640, 480]")
        print("  capture_on_speech: true")
        print("  save_captures: false")
        print("  capture_dir: \"camera_captures\"")
        print("  jpeg_quality: 85")
        print("```")
        
        if len(available) > 1:
            print("\n# Other available cameras:")
            for idx in available[1:]:
                print(f"# device_id: {idx}  # {manager.camera_info[idx]['name']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()