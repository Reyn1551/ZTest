#!/usr/bin/env python3
"""
ATCS Jogja - Gradio Web Interface with Public Share
"""

import gradio as gr
import numpy as np
import cv2
import subprocess
import threading
import time
import logging
from datetime import datetime
from typing import Optional
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed - running in stream-only mode")


class StreamCapture:
    """FFmpeg-based stream capture"""
    
    def __init__(self, url, resolution=(1280, 720), fps=15):
        self.url = url
        self.resolution = resolution
        self.fps = fps
        self.process = None
        self.buffer = queue.Queue(maxsize=100)
        self.running = False
        self.thread = None
        
    def start(self):
        self.running = True
        self._start_ffmpeg()
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()
        
        # Wait for buffer
        for _ in range(30):
            if self.buffer.qsize() > 0:
                return True
            time.sleep(0.1)
        return self.buffer.qsize() > 0
    
    def _start_ffmpeg(self):
        w, h = self.resolution
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', self.url,
            '-vf', f'fps={self.fps},scale={w}:{h}',
            '-pix_fmt', 'bgr24',
            '-f', 'rawvideo', '-an', 'pipe:1'
        ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
        
    def _capture(self):
        frame_size = self.resolution[0] * self.resolution[1] * 3
        while self.running:
            try:
                if self.process.poll() is not None:
                    self._start_ffmpeg()
                    continue
                raw = self.process.stdout.read(frame_size)
                if len(raw) == frame_size:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.resolution[1], self.resolution[0], 3)).copy()
                    try:
                        self.buffer.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.buffer.get_nowait()
                            self.buffer.put_nowait(frame)
                        except:
                            pass
            except:
                time.sleep(0.1)
                
    def read(self):
        try:
            return True, self.buffer.get(timeout=5)
        except:
            return False, None
            
    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()


# Global state
class AppState:
    def __init__(self):
        self.stream = None
        self.model = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.detections = 0
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO("yolov8n.pt")
            except:
                pass

state = AppState()
STREAM_URL = "https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Wirosaban_View_Barat.stream/chunklist_w821725872.m3u8"


def processing_loop():
    """Main processing loop"""
    class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    colors = {'car': (255,100,100), 'motorcycle': (100,255,100), 'bus': (100,100,255), 'truck': (255,255,100)}
    
    while state.running:
        try:
            ret, frame = state.stream.read()
            if not ret or frame is None:
                continue
                
            state.frame_count += 1
            det_count = 0
            
            # Detection
            if state.model:
                try:
                    results = state.model(frame, conf=0.5, classes=[2,3,5,7], verbose=False)[0]
                    if results.boxes is not None and len(results.boxes) > 0:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        classes = results.boxes.cls.cpu().numpy().astype(int)
                        confs = results.boxes.conf.cpu().numpy()
                        
                        for box, cls, conf in zip(boxes, classes, confs):
                            x1, y1, x2, y2 = map(int, box)
                            name = class_names.get(cls, 'unknown')
                            color = colors.get(name, (200,200,200))
                            
                            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                            cv2.putText(frame, f"{name} {conf:.0%}", (x1, y1-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            det_count += 1
                except:
                    pass
            
            # Draw overlay
            cv2.putText(frame, "ATCS JOGJA - AI Surveillance", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, f"Vehicles: {det_count}", (10, frame.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            state.detections = det_count
            
            with state.lock:
                state.frame = frame
                
        except Exception as e:
            time.sleep(0.1)


def start_stream():
    """Start streaming"""
    if state.running:
        return "Already running", None, 0, 0
    
    state.stream = StreamCapture(STREAM_URL)
    if not state.stream.start():
        return "Failed to start stream", None, 0, 0
    
    state.running = True
    state.frame_count = 0
    threading.Thread(target=processing_loop, daemon=True).start()
    
    return "✅ Stream started", None, 0, 0


def stop_stream():
    """Stop streaming"""
    state.running = False
    if state.stream:
        state.stream.stop()
    return "⏹️ Stream stopped", None, 0, 0


def refresh_frame():
    """Get current frame and stats"""
    with state.lock:
        frame = state.frame.copy() if state.frame is not None else None
    
    status = "🟢 Running" if state.running else "🔴 Stopped"
    return frame, state.frame_count, state.detections, status


def create_ui():
    """Create Gradio UI"""
    
    with gr.Blocks(title="ATCS Jogja") as demo:
        gr.Markdown("""
        # 🚦 ATCS Jogja - AI Traffic Surveillance
        
        **Speed + Direction + Helmet Detection System**
        """)
        
        mode = "🟢 **Full Mode** (YOLOv8 Detection)" if YOLO_AVAILABLE else "🟡 **Stream Mode** (Install ultralytics for detection)"
        gr.Markdown(mode)
        
        with gr.Row():
            start_btn = gr.Button("▶️ Start Stream", variant="primary")
            stop_btn = gr.Button("⏹️ Stop Stream")
            refresh_btn = gr.Button("🔄 Refresh")
        
        status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
        
        with gr.Row():
            with gr.Column(scale=3):
                video_display = gr.Image(label="Live Feed")
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Statistics")
                frame_num = gr.Number(label="Frames", value=0)
                vehicle_num = gr.Number(label="Vehicles", value=0)
                stream_stat = gr.Textbox(label="Stream Status", value="Stopped")
        
        with gr.Accordion("ℹ️ Info", open=False):
            gr.Markdown(f"""
            **Stream URL:** `{STREAM_URL[:60]}...`
            
            **Resolution:** 1280x720 | **FPS:** 15
            
            **Deteksi:** Kendaraan (mobil, motor, bus, truk)
            """)
        
        # Event handlers
        start_btn.click(start_stream, outputs=[status_text, video_display, frame_num, vehicle_num])
        stop_btn.click(stop_stream, outputs=[status_text, video_display, frame_num, vehicle_num])
        refresh_btn.click(refresh_frame, outputs=[video_display, frame_num, vehicle_num, stream_stat])
    
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ATCS Jogja Gradio Server - PUBLIC MODE")
    print("=" * 60)
    print(f"YOLO: {'Available' if YOLO_AVAILABLE else 'Not Available'}")
    print(f"URL: {STREAM_URL[:50]}...")
    print("=" * 60)
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # PUBLIC LINK
        show_error=True
    )
