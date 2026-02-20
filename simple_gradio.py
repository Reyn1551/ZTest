#!/usr/bin/env python3
"""
Simplified Gradio App untuk ATCS Vision
Works with or without YOLOv8 model (demo mode available)
"""

import gradio as gr
import numpy as np
import cv2
import subprocess
import threading
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if ultralytics is available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLOv8 available - full mode")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLOv8 not available - demo mode (no detection)")


class SimpleStreamCapture:
    """Simple stream capture using FFmpeg"""
    
    def __init__(self, url: str, resolution: tuple = (1280, 720), target_fps: int = 15):
        self.url = url
        self.resolution = resolution
        self.target_fps = target_fps
        self.process = None
        self.frame_buffer = queue.Queue(maxsize=100)
        self.is_running = False
        self.capture_thread = None
        
    def _build_command(self):
        w, h = self.resolution
        return [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-i', self.url,
            '-vf', f'fps={self.target_fps},scale={w}:{h}',
            '-pix_fmt', 'bgr24',
            '-f', 'rawvideo',
            '-an',
            'pipe:1'
        ]
    
    def _capture_loop(self):
        frame_size = self.resolution[0] * self.resolution[1] * 3
        
        while self.is_running:
            try:
                if self.process is None or self.process.poll() is not None:
                    self._start_process()
                    continue
                
                raw_frame = self.process.stdout.read(frame_size)
                
                if len(raw_frame) != frame_size:
                    continue
                
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.resolution[1], self.resolution[0], 3)).copy()
                
                try:
                    self.frame_buffer.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait(frame)
                    except:
                        pass
                        
            except Exception as e:
                logger.debug(f"Capture error: {e}")
                time.sleep(0.1)
    
    def _start_process(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                pass
        
        time.sleep(2)
        
        try:
            cmd = self._build_command()
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            time.sleep(1)
        except Exception as e:
            logger.error(f"FFmpeg error: {e}")
    
    def start(self):
        self.is_running = True
        self._start_process()
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Wait for frames
        for _ in range(50):
            if self.frame_buffer.qsize() > 0:
                return True
            time.sleep(0.1)
        
        return self.frame_buffer.qsize() > 0
    
    def read(self):
        try:
            return True, self.frame_buffer.get(timeout=5)
        except queue.Empty:
            return False, None
    
    def stop(self):
        self.is_running = False
        if self.process:
            self.process.terminate()
        self.frame_buffer.queue.clear()


class ATCSVisionDemo:
    """Demo application"""
    
    def __init__(self):
        # Configuration
        self.stream_url = "https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Wirosaban_View_Barat.stream/chunklist_w821725872.m3u8"
        self.resolution = (1280, 720)
        self.target_fps = 15
        
        # Components
        self.stream = None
        self.model = None
        self.is_running = False
        
        # Stats
        self.frame_count = 0
        self.detections_count = 0
        self.violations_count = 0
        self.vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        
        # Current frame
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Load model if available
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO("yolov8n.pt")
                logger.info("YOLOv8 model loaded")
            except Exception as e:
                logger.warning(f"Model load error: {e}")
    
    def start_stream(self):
        if self.is_running:
            return "Stream already running"
        
        logger.info("Starting stream...")
        
        self.stream = SimpleStreamCapture(
            self.stream_url,
            self.resolution,
            self.target_fps
        )
        
        if not self.stream.start():
            return "Failed to start stream"
        
        self.is_running = True
        
        # Start processing thread
        thread = threading.Thread(target=self._processing_loop, daemon=True)
        thread.start()
        
        return "✅ Stream started successfully"
    
    def stop_stream(self):
        if not self.is_running:
            return "Stream not running"
        
        self.is_running = False
        
        if self.stream:
            self.stream.stop()
        
        return "⏹️ Stream stopped"
    
    def _processing_loop(self):
        while self.is_running:
            try:
                ret, frame = self.stream.read()
                
                if not ret or frame is None:
                    continue
                
                self.frame_count += 1
                
                # Detection if model available
                detections = []
                if self.model is not None:
                    try:
                        results = self.model(
                            frame,
                            conf=0.5,
                            classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
                            verbose=False
                        )[0]
                        
                        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                        
                        if results.boxes is not None and len(results.boxes) > 0:
                            boxes = results.boxes.xyxy.cpu().numpy()
                            classes = results.boxes.cls.cpu().numpy().astype(int)
                            confs = results.boxes.conf.cpu().numpy()
                            
                            for box, cls, conf in zip(boxes, classes, confs):
                                x1, y1, x2, y2 = map(int, box)
                                class_name = class_names.get(cls, 'unknown')
                                
                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'class': class_name,
                                    'confidence': conf
                                })
                                
                                # Draw on frame
                                color = {
                                    'car': (255, 100, 100),
                                    'motorcycle': (100, 255, 100),
                                    'bus': (100, 100, 255),
                                    'truck': (255, 255, 100)
                                }.get(class_name, (200, 200, 200))
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                label = f"{class_name} {conf:.0%}"
                                cv2.putText(frame, label, (x1, y1-5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                    except Exception as e:
                        logger.debug(f"Detection error: {e}")
                
                # Draw header
                cv2.putText(frame, "ATCS JOGJA - AI Traffic Surveillance", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Stats
                cv2.putText(frame, f"Vehicles: {len(detections)}", (10, frame.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Update counts
                self.detections_count = len(detections)
                
                with self.frame_lock:
                    self.current_frame = frame
                    
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
    
    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_stats(self):
        return {
            'is_running': self.is_running,
            'frame_count': self.frame_count,
            'detections': self.detections_count,
            'yolo_available': YOLO_AVAILABLE
        }


# Create app instance
app = ATCSVisionDemo()


def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="ATCS Jogja - AI Traffic Surveillance") as demo:
        
        gr.Markdown("""
        # 🚦 ATCS Jogja - AI Traffic Surveillance
        
        **Speed + Direction + Helmet Detection System**
        
        Sistem pemantauan lalu lintas cerdas menggunakan CCTV ATCS Jogja dengan AI Computer Vision.
        """)
        
        # Mode indicator
        mode_text = "🟢 **Full Mode** (YOLOv8 Detection Active)" if YOLO_AVAILABLE else "🟡 **Demo Mode** (Stream Only - Install ultralytics for detection)"
        gr.Markdown(mode_text)
        
        # Controls
        with gr.Row():
            start_btn = gr.Button("▶️ Start Stream", variant="primary", size="lg")
            stop_btn = gr.Button("⏹️ Stop Stream", variant="secondary", size="lg")
            status_box = gr.Textbox(label="Status", value="Ready", interactive=False)
        
        # Main display
        with gr.Row():
            with gr.Column(scale=3):
                video_output = gr.Image(label="Live Feed", height=500)
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Statistics")
                
                with gr.Group():
                    frame_count = gr.Number(label="Frames Processed", value=0)
                    vehicles_count = gr.Number(label="Current Vehicles", value=0)
                    stream_status = gr.Textbox(label="Stream", value="Stopped")
        
        # Info
        with gr.Accordion("ℹ️ Stream Information", open=False):
            gr.Markdown(f"""
            **Stream URL:** 
            ```
            {app.stream_url}
            ```
            
            **Resolution:** {app.resolution}
            
            **Target FPS:** {app.target_fps}
            
            **Detection:** {"YOLOv8 (COCO)" if YOLO_AVAILABLE else "Not Available"}
            
            **Classes:** car, motorcycle, bus, truck
            """)
        
        # Instructions
        with gr.Accordion("📖 Instructions", open=False):
            gr.Markdown("""
            ### Cara Penggunaan
            
            1. Klik **Start Stream** untuk memulai streaming
            2. Tunggu beberapa detik untuk buffer
            3. Video akan ditampilkan dengan deteksi real-time
            
            ### Fitur
            
            - **Vehicle Detection**: Deteksi mobil, motor, bus, truk
            - **Speed Estimation**: Estimasi kecepatan (jika kalibrasi)
            - **Direction Analysis**: Analisis arah pergerakan
            - **Helmet Detection**: Deteksi helm (motor)
            
            ### Troubleshooting
            
            Jika video tidak muncul:
            - Pastikan koneksi internet stabil
            - Stream mungkin sedang tidak aktif
            - Coba refresh halaman
            """)
        
        # Event handlers
        def start_handler():
            result = app.start_stream()
            return result
        
        def stop_handler():
            result = app.stop_stream()
            return result
        
        def update_display():
            frame = app.get_frame()
            stats = app.get_stats()
            
            status = "🟢 Running" if stats['is_running'] else "🔴 Stopped"
            
            return (
                frame,
                stats['frame_count'],
                stats['detections'],
                status
            )
        
        start_btn.click(start_handler, outputs=[status_box])
        stop_btn.click(stop_handler, outputs=[status_box])
        
        # Auto-update
        demo.load(
            update_display,
            outputs=[video_output, frame_count, vehicles_count, stream_status],
            every=0.5
        )
    
    return demo


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Starting ATCS Jogja Gradio Server")
    print("=" * 50)
    print(f"YOLO Available: {YOLO_AVAILABLE}")
    print(f"Stream URL: {app.stream_url[:50]}...")
    print("=" * 50)
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
