#!/usr/bin/env python3
"""
Camera Capture Module
Captures images from the camera for attaching to conversation messages.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
import queue
import time
from typing import Optional, Callable, Tuple
import base64
from dataclasses import dataclass

@dataclass
class CameraConfig:
    """Configuration for camera capture."""
    device_id: int = 0  # Camera device ID (0 for default camera)
    resolution: Tuple[int, int] = (640, 480)  # (width, height)
    capture_on_speech: bool = True  # Capture when user starts speaking
    save_captures: bool = False  # Save captured images to disk
    capture_dir: str = "camera_captures"  # Directory to save captures
    jpeg_quality: int = 85  # JPEG compression quality (0-100)

class CameraCapture:
    """Handles camera capture for conversation context."""
    
    def __init__(self, config: CameraConfig = None):
        """Initialize camera capture system."""
        self.config = config or CameraConfig()
        self.camera = None
        self.is_running = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=2)  # Keep only latest frames
        self.latest_frame = None
        self.capture_callbacks = []
        
        # Create capture directory if saving is enabled
        if self.config.save_captures:
            Path(self.config.capture_dir).mkdir(exist_ok=True)
    
    def start(self) -> bool:
        """Start the camera capture thread."""
        if self.is_running:
            print("📷 Camera already running")
            return True
            
        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(self.config.device_id)
            
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret or frame is None:
                print("❌ Failed to read from camera")
                self.camera.release()
                return False
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            print(f"📷 Camera started (device {self.config.device_id}, {self.config.resolution[0]}x{self.config.resolution[1]})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start camera: {e}")
            if self.camera:
                self.camera.release()
            return False
    
    def stop(self):
        """Stop the camera capture thread."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Wait for thread to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
        
        print("📷 Camera stopped")
    
    def _capture_loop(self):
        """Background thread that continuously captures frames."""
        while self.is_running:
            try:
                if self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        # Update latest frame
                        self.latest_frame = frame
                        
                        # Try to add to queue (non-blocking)
                        try:
                            # Remove old frame if queue is full
                            if self.frame_queue.full():
                                self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"⚠️ Camera capture error: {e}")
                time.sleep(0.1)
    
    def capture_image(self) -> Optional[Tuple[np.ndarray, str]]:
        """Capture a single image from the camera.
        
        Returns:
            Tuple of (image_array, base64_encoded_jpeg) or None if capture failed
        """
        if not self.is_running or self.latest_frame is None:
            print("⚠️ No camera frame available")
            return None
            
        try:
            # Get the latest frame
            frame = self.latest_frame.copy()
            
            # Encode as JPEG
            _, jpeg_buffer = cv2.imencode('.jpg', frame, 
                [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
            
            # Convert to base64
            jpeg_base64 = base64.b64encode(jpeg_buffer).decode('utf-8')
            
            # Save if configured
            if self.config.save_captures:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = Path(self.config.capture_dir) / f"capture_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"💾 Saved capture: {filename}")
            
            # Notify callbacks
            for callback in self.capture_callbacks:
                try:
                    callback(frame, jpeg_base64)
                except Exception as e:
                    print(f"⚠️ Capture callback error: {e}")
            
            return frame, jpeg_base64
            
        except Exception as e:
            print(f"❌ Failed to capture image: {e}")
            return None
    
    def add_capture_callback(self, callback: Callable[[np.ndarray, str], None]):
        """Add a callback to be notified when an image is captured.
        
        Args:
            callback: Function that receives (image_array, base64_jpeg)
        """
        self.capture_callbacks.append(callback)
    
    def get_camera_info(self) -> dict:
        """Get information about the current camera."""
        if not self.camera or not self.camera.isOpened():
            return {"status": "not connected"}
            
        return {
            "status": "connected",
            "device_id": self.config.device_id,
            "resolution": {
                "width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            },
            "fps": int(self.camera.get(cv2.CAP_PROP_FPS)),
            "backend": self.camera.getBackendName()
        }

# Test function
def test_camera_capture():
    """Test the camera capture functionality."""
    print("🧪 Testing camera capture...")
    
    config = CameraConfig(
        device_id=0,
        resolution=(640, 480),
        save_captures=True,
        capture_dir="test_captures"
    )
    
    camera = CameraCapture(config)
    
    if camera.start():
        print("✅ Camera started successfully")
        print(f"📊 Camera info: {camera.get_camera_info()}")
        
        # Wait a moment for camera to stabilize
        time.sleep(1)
        
        # Capture a test image
        result = camera.capture_image()
        if result:
            frame, jpeg_base64 = result
            print(f"✅ Captured image: {frame.shape}, Base64 length: {len(jpeg_base64)}")
        else:
            print("❌ Failed to capture image")
        
        camera.stop()
    else:
        print("❌ Failed to start camera")

if __name__ == "__main__":
    test_camera_capture()