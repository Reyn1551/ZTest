#!/usr/bin/env python3
"""
ATCS Jogja - Real-time Video Streaming Gradio Interface
Smooth 20-30 FPS Video Output
"""

import gradio as gr
import numpy as np
import cv2
import subprocess
import threading
import time
import logging
from datetime import datetime
from typing import Optional, Generator
import queue
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

# ============== CONFIGURATION ==============
STREAM_URL = "https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Wirosaban_View_Barat.stream/chunklist_w821725872.m3u8"
RESOLUTION = (1280, 720)
TARGET_FPS = 25
BUFFER_SIZE = 150


# ============== STREAM CAPTURE ==============
class SmoothStreamCapture:
    """High-performance stream capture with smooth frame delivery"""
    
    def __init__(self, url, resolution=RESOLUTION, target_fps=TARGET_FPS):
        self.url = url
        self.resolution = resolution
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        self.process = None
        self.frame_buffer = queue.Queue(maxsize=BUFFER_SIZE)
        self.running = False
        self.capture_thread = None
        self.stats = {'frames': 0, 'dropped': 0, 'fps': 0.0}
        
    def _build_ffmpeg_cmd(self):
        w, h = self.resolution
        return [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-fflags', 'nobuffer+fastseek+genpts',
            '-flags', 'low_delay',
            '-strict', 'experimental',
            '-i', self.url,
            '-vf', f'fps={self.target_fps},scale={w}:{h}:flags=fast_bilinear',
            '-pix_fmt', 'bgr24',
            '-f', 'rawvideo',
            '-an', 'pipe:1'
        ]
    
    def _start_ffmpeg(self):
        try:
            if self.process:
                self.process.terminate()
                time.sleep(0.5)
            
            cmd = self._build_ffmpeg_cmd()
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            logger.info("FFmpeg started")
            time.sleep(2)  # Wait for stream to stabilize
        except Exception as e:
            logger.error(f"FFmpeg error: {e}")
    
    def _capture_loop(self):
        frame_size = self.resolution[0] * self.resolution[1] * 3
        last_time = time.time()
        frame_count = 0
        
        while self.running:
            try:
                if self.process is None or self.process.poll() is not None:
                    logger.warning("FFmpeg died, restarting...")
                    self._start_ffmpeg()
                    continue
                
                raw = self.process.stdout.read(frame_size)
                
                if len(raw) != frame_size:
                    continue
                
                # Convert to frame
                frame = np.frombuffer(raw, dtype=np.uint8)
                frame = frame.reshape((self.resolution[1], self.resolution[0], 3)).copy()
                
                # Add to buffer
                try:
                    self.frame_buffer.put_nowait(frame)
                    frame_count += 1
                    self.stats['frames'] = frame_count
                    
                    # Calculate FPS
                    current = time.time()
                    if current - last_time >= 1.0:
                        self.stats['fps'] = frame_count / (current - last_time)
                        frame_count = 0
                        last_time = current
                        
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait(frame)
                        self.stats['dropped'] += 1
                    except:
                        pass
                        
            except Exception as e:
                logger.debug(f"Capture error: {e}")
                time.sleep(0.01)
    
    def start(self):
        self.running = True
        self._start_ffmpeg()
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Pre-buffer
        logger.info("Pre-buffering frames...")
        for _ in range(50):
            if self.frame_buffer.qsize() >= 30:
                break
            time.sleep(0.1)
        
        logger.info(f"Ready! Buffer: {self.frame_buffer.qsize()}")
        return self.frame_buffer.qsize() > 0
    
    def read(self):
        try:
            return True, self.frame_buffer.get(timeout=5)
        except queue.Empty:
            return False, None
    
    def get_stats(self):
        return {
            **self.stats,
            'buffer_size': self.frame_buffer.qsize()
        }
    
    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
        self.frame_buffer.queue.clear()


# ============== DETECTOR ==============
class VehicleDetector:
    def __init__(self):
        self.model = None
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.colors = {
            'car': (255, 100, 100),
            'motorcycle': (100, 255, 100),
            'bus': (100, 100, 255),
            'truck': (255, 255, 100)
        }
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO("yolov8n.pt")
                logger.info("YOLOv8 loaded")
            except Exception as e:
                logger.warning(f"YOLO load failed: {e}")
    
    def detect(self, frame):
        if self.model is None:
            return frame, 0
        
        try:
            results = self.model(frame, conf=0.5, classes=[2, 3, 5, 7], verbose=False)[0]
            
            count = 0
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    name = self.class_names.get(cls, 'unknown')
                    color = self.colors.get(name, (200, 200, 200))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name} {conf:.0%}", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    count += 1
            
            return frame, count
        except:
            return frame, 0


# ============== VIDEO PROCESSOR ==============
class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self):
        self.capture = None
        self.detector = VehicleDetector()
        self.running = False
        self.frame_count = 0
        self.vehicle_count = 0
        self.last_frame = None
        self.lock = threading.Lock()
        self.process_thread = None
        
    def start(self):
        if self.running:
            return "Already running"
        
        self.capture = SmoothStreamCapture(STREAM_URL)
        if not self.capture.start():
            return "Failed to start stream"
        
        self.running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        return "✅ Stream started"
    
    def stop(self):
        self.running = False
        if self.capture:
            self.capture.stop()
        return "⏹️ Stream stopped"
    
    def _process_loop(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue
            
            # Detect
            frame, count = self.detector.detect(frame)
            self.vehicle_count = count
            self.frame_count += 1
            
            # Draw overlay
            self._draw_overlay(frame)
            
            # Store frame
            with self.lock:
                self.last_frame = frame
    
    def _draw_overlay(self, frame):
        # Header
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(frame, "ATCS JOGJA - AI Traffic Surveillance", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Footer
        cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, f"Vehicles: {self.vehicle_count} | FPS: {self.capture.stats.get('fps', 0):.1f}",
                   (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def get_frame(self):
        with self.lock:
            if self.last_frame is not None:
                return self.last_frame.copy()
        return None
    
    def get_stats(self):
        stats = {
            'running': self.running,
            'frame_count': self.frame_count,
            'vehicle_count': self.vehicle_count
        }
        if self.capture:
            stats.update(self.capture.get_stats())
        return stats


# ============== GRADIO STREAMING ==============
processor = VideoProcessor()


def video_stream():
    """Generator for video streaming"""
    last_time = time.time()
    
    while processor.running:
        frame = processor.get_frame()
        if frame is not None:
            # Control frame rate
            elapsed = time.time() - last_time
            sleep_time = max(0, (1.0 / TARGET_FPS) - elapsed)
            time.sleep(sleep_time)
            last_time = time.time()
            
            yield frame
        else:
            time.sleep(0.01)


def start_stream():
    result = processor.start()
    return result


def stop_stream():
    result = processor.stop()
    return result


def get_current_frame():
    """Get single frame for display"""
    frame = processor.get_frame()
    stats = processor.get_stats()
    status = "🟢 Running" if stats['running'] else "🔴 Stopped"
    fps = stats.get('fps', 0)
    return frame, stats.get('frame_count', 0), stats.get('vehicle_count', 0), status, f"{fps:.1f}"


# ============== GRADIO UI ==============
def create_ui():
    with gr.Blocks(title="ATCS Jogja - Real-time Video", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚦 ATCS Jogja - Real-time Traffic Surveillance
        ### Smooth Video Streaming (25 FPS)
        """)
        
        mode = "🟢 **YOLOv8 Detection Active**" if YOLO_AVAILABLE else "🟡 **Stream Mode** (Install ultralytics for AI detection)"
        gr.Markdown(mode)
        
        with gr.Row():
            start_btn = gr.Button("▶️ Start Stream", variant="primary", size="lg")
            stop_btn = gr.Button("⏹️ Stop Stream", variant="secondary", size="lg")
        
        status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
        
        # Video display - continuously updating
        with gr.Row():
            with gr.Column(scale=4):
                video_display = gr.Image(label="Live Video Feed", streaming=True, height=500)
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Statistics")
                frame_count = gr.Number(label="Frames", value=0)
                vehicle_count = gr.Number(label="Vehicles", value=0)
                fps_display = gr.Textbox(label="FPS", value="0.0")
                stream_status = gr.Textbox(label="Status", value="Stopped")
        
        # Auto-refresh timer
        timer = gr.Timer(0.04, active=True)  # ~25 FPS refresh
        
        # Event handlers
        start_btn.click(start_stream, outputs=[status_text])
        stop_btn.click(stop_stream, outputs=[status_text])
        
        # Auto-update frame
        timer.tick(get_current_frame, outputs=[video_display, frame_count, vehicle_count, stream_status, fps_display])
        
        with gr.Accordion("ℹ️ Information", open=False):
            gr.Markdown(f"""
            **Stream URL:** `{STREAM_URL[:60]}...`
            
            **Resolution:** 1280x720
            **Target FPS:** {TARGET_FPS}
            **Buffer:** {BUFFER_SIZE} frames
            
            **Features:**
            - Real-time video streaming
            - Vehicle detection (car, motorcycle, bus, truck)
            - Smooth 25 FPS playback
            - Anti-freeze buffer system
            """)
    
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ATCS Jogja - Real-time Video Streaming")
    print("=" * 60)
    print(f"YOLO: {'Available' if YOLO_AVAILABLE else 'Not Available'}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Resolution: {RESOLUTION}")
    print("=" * 60)
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
