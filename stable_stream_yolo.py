#!/usr/bin/env python3
"""
ATCS Jogja - Ultra Stable Video Streaming dengan YOLOv8 Detection
Real-time 25-30 FPS dengan buffer anti-freeze
"""

import subprocess
import numpy as np
import cv2
import threading
import queue
import time
import os
import logging
from datetime import datetime
from typing import Optional, Tuple
import gradio as gr

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================= KONFIGURASI =================
STREAM_URL = "https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Wirosaban_View_Barat.stream/chunklist_w821725872.m3u8"
WIDTH = 1280
HEIGHT = 720
BUFFER_SECONDS = 30
FALLBACK_FPS = 25.0
LOG_FILE = "stable_stream_log.txt"

# YOLO Configuration
YOLO_MODEL = "yolov8n.pt"  # Gunakan yolov8n.pt yang sudah di-download
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
CLASS_COLORS = {
    'car': (255, 100, 100),
    'motorcycle': (100, 255, 100),
    'bus': (100, 100, 255),
    'truck': (255, 255, 100)
}


# ================= LOGGING HELPER =================
def write_log(text):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {text}"
    logger.info(text)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ================= DETECT FPS =================
def detect_stream_fps(url, timeout=15):
    """Deteksi FPS asli stream dengan ffprobe"""
    write_log("Mendeteksi FPS asli stream...")
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
        '-of', 'csv=p=0',
        url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout.strip()
        write_log(f"ffprobe raw output: '{output}'")

        parts = output.replace('\n', ',').split(',')
        for part in parts:
            part = part.strip()
            if '/' in part:
                try:
                    num, den = part.split('/')
                    fps = float(num) / float(den)
                    if 1.0 < fps < 120.0:
                        write_log(f"FPS terdeteksi: {fps:.3f}")
                        return fps
                except:
                    continue
            elif part:
                try:
                    fps = float(part)
                    if 1.0 < fps < 120.0:
                        write_log(f"FPS terdeteksi (plain): {fps:.3f}")
                        return fps
                except:
                    continue
    except Exception as e:
        write_log(f"ffprobe error: {e}")

    write_log(f"Pakai fallback FPS: {FALLBACK_FPS}")
    return FALLBACK_FPS


# ================= STABLE STREAMER =================
class StableStreamer:
    """
    Streamer yang feed frame ke queue tanpa manipulasi FPS.
    Menggunakan passthrough vsync untuk motion yang natural.
    """
    def __init__(self, src, width, height, fps):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = int(BUFFER_SECONDS * fps)
        write_log(f"Buffer: {self.buffer_size} frames ({BUFFER_SECONDS}s @ {fps:.2f} FPS)")

        self.q = queue.Queue(maxsize=self.buffer_size)
        self.stopped = False
        self.proc = None
        self._lock = threading.Lock()
        self.frame_count = 0
        self.dropped_count = 0

        # vsync passthrough: output frame apa adanya
        self.cmd = [
            'ffmpeg',
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '10',
            '-i', self.src,
            '-vsync', 'passthrough',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-loglevel', 'error',
            'pipe:1'
        ]

    def start(self):
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        frame_size = self.width * self.height * 3
        while not self.stopped:
            try:
                raw = self.proc.stdout.read(frame_size)
                if len(raw) != frame_size:
                    write_log("Koneksi putus, mencoba reconnect...")
                    with self._lock:
                        if self.proc:
                            self.proc.kill()
                    time.sleep(2)
                    with self._lock:
                        self.proc = subprocess.Popen(
                            self.cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            bufsize=10**8
                        )
                    continue

                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()

                try:
                    self.q.put(frame, timeout=5.0)
                    self.frame_count += 1
                except queue.Full:
                    self.dropped_count += 1
                    if self.dropped_count % 100 == 0:
                        write_log(f"Queue penuh - {self.dropped_count} frames di-skip")
                        
            except Exception as e:
                write_log(f"Capture error: {e}")
                time.sleep(0.1)

    def read(self, timeout=10.0):
        return self.q.get(timeout=timeout)

    def queue_size(self):
        return self.q.qsize()

    def get_stats(self):
        return {
            'buffer_size': self.q.qsize(),
            'max_buffer': self.buffer_size,
            'frames_captured': self.frame_count,
            'frames_dropped': self.dropped_count,
            'fps': self.fps
        }

    def stop(self):
        self.stopped = True
        with self._lock:
            if self.proc:
                self.proc.kill()


# ================= YOLOV8 DETECTOR =================
class YOLODetector:
    """YOLOv8 Vehicle Detector"""
    
    def __init__(self, model_path=YOLO_MODEL):
        self.model = None
        self.class_names = VEHICLE_CLASSES
        self.colors = CLASS_COLORS
        self.enabled = False
        
        # Try to load YOLO
        try:
            from ultralytics import YOLO
            
            # Check if model file exists
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                write_log(f"YOLOv8 loaded from {model_path}")
            else:
                # Will auto-download
                write_log(f"Downloading YOLOv8 model: {model_path}")
                self.model = YOLO(model_path)
            
            self.enabled = True
            write_log("YOLOv8 detection enabled")
            
        except ImportError:
            write_log("ultralytics not installed - detection disabled")
        except Exception as e:
            write_log(f"YOLO load error: {e} - detection disabled")
    
    def detect(self, frame):
        """Detect vehicles in frame"""
        if not self.enabled or self.model is None:
            return frame, 0, []
        
        try:
            results = self.model(
                frame,
                conf=0.5,
                classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
                verbose=False
            )[0]
            
            detections = []
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    name = self.class_names.get(cls, 'unknown')
                    color = self.colors.get(name, (200, 200, 200))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{name} {conf:.0%}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    detections.append({
                        'class': name,
                        'confidence': float(conf),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            return frame, len(detections), detections
            
        except Exception as e:
            write_log(f"Detection error: {e}")
            return frame, 0, []


# ================= VIDEO PROCESSOR =================
class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self):
        self.streamer = None
        self.detector = YOLODetector()
        self.running = False
        self.frame_count = 0
        self.vehicle_count = 0
        self.current_frame = None
        self.current_detections = []
        self.lock = threading.Lock()
        self.process_thread = None
        self.display_fps = 0.0
        self.last_update = time.time()
        
    def start(self, stream_url=None):
        if self.running:
            return "Already running"
        
        url = stream_url or STREAM_URL
        
        # Clean old log
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        
        write_log("=== ATCS Vision - Stable Stream v3 ===")
        
        # Detect FPS
        stream_fps = detect_stream_fps(url)
        
        # Start streamer
        self.streamer = StableStreamer(url, WIDTH, HEIGHT, stream_fps)
        self.streamer.start()
        
        # Warm up buffer
        write_log(f"Warm up: mengisi buffer {BUFFER_SECONDS}s...")
        target_fill = min(self.streamer.buffer_size - 10, int(5 * stream_fps))  # Wait for 5 seconds buffer
        
        warmup_start = time.time()
        max_warmup = 30  # Max 30 seconds
        
        while True:
            q = self.streamer.queue_size()
            elapsed = time.time() - warmup_start
            
            if q >= target_fill or elapsed > max_warmup:
                break
            time.sleep(0.5)
        
        write_log(f"Buffer ready: {self.streamer.queue_size()} frames")
        
        self.running = True
        self.frame_count = 0
        self.display_fps = stream_fps
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        return f"✅ Stream started ({stream_fps:.1f} FPS, buffer: {self.streamer.queue_size()} frames)"
    
    def stop(self):
        self.running = False
        if self.streamer:
            self.streamer.stop()
        write_log("Stream stopped")
        return "⏹️ Stream stopped"
    
    def _process_loop(self):
        """Main processing loop with fixed-rate timing"""
        frame_duration = 1.0 / self.display_fps
        next_frame_time = time.time()
        health_timer = time.time()
        local_frame_count = 0
        
        while self.running:
            try:
                # Get frame from buffer
                frame = self.streamer.read(timeout=10.0)
                
                # Timing control
                now = time.time()
                wait = next_frame_time - now
                if wait > 0:
                    time.sleep(wait)
                
                # Reset if lag > 1 second
                lag = time.time() - next_frame_time
                if lag > 1.0:
                    next_frame_time = time.time() + frame_duration
                else:
                    next_frame_time += frame_duration
                
                # Detect vehicles
                frame, count, detections = self.detector.detect(frame)
                
                # Draw overlay
                self._draw_overlay(frame)
                
                # Update state
                self.frame_count += 1
                local_frame_count += 1
                self.vehicle_count = count
                
                with self.lock:
                    self.current_frame = frame
                    self.current_detections = detections
                
                # Health log every 5 seconds
                now = time.time()
                if now - health_timer >= 5.0:
                    elapsed = now - health_timer
                    fps_actual = local_frame_count / elapsed
                    q = self.streamer.queue_size()
                    write_log(f"Health → FPS: {fps_actual:.1f} | Vehicles: {count} | Buffer: {q}")
                    local_frame_count = 0
                    health_timer = now
                    
            except queue.Empty:
                write_log("STALL: Buffer empty, waiting...")
                next_frame_time = time.time()
            except Exception as e:
                write_log(f"Process error: {e}")
                time.sleep(0.1)
    
    def _draw_overlay(self, frame):
        """Draw overlay on frame"""
        now = datetime.now()
        
        # Header background
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "ATCS JOGJA - AI Traffic Surveillance", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Timestamp
        cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Stats on right side
        stats_text = f"FPS: {self.display_fps:.1f} | Buffer: {self.streamer.queue_size() if self.streamer else 0}"
        cv2.putText(frame, stats_text, (frame.shape[1] - 350, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Footer background
        cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        
        # Vehicle count
        cv2.putText(frame, f"Vehicles Detected: {self.vehicle_count}", (10, frame.shape[0] - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # YOLO status
        yolo_status = "YOLO: ON" if self.detector.enabled else "YOLO: OFF"
        color = (0, 255, 0) if self.detector.enabled else (0, 165, 255)
        cv2.putText(frame, yolo_status, (frame.shape[1] - 150, frame.shape[0] - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def get_frame(self):
        """Get current frame"""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_stats(self):
        """Get current statistics"""
        stats = {
            'running': self.running,
            'frame_count': self.frame_count,
            'vehicle_count': self.vehicle_count,
            'fps': self.display_fps,
            'yolo_enabled': self.detector.enabled
        }
        if self.streamer:
            stats.update(self.streamer.get_stats())
        return stats


# ================= GRADIO INTERFACE =================
processor = VideoProcessor()


def start_stream():
    return processor.start()


def stop_stream():
    return processor.stop()


def get_current_frame():
    """Get current frame for display"""
    frame = processor.get_frame()
    stats = processor.get_stats()
    status = "🟢 Running" if stats['running'] else "🔴 Stopped"
    yolo = "✅ ON" if stats.get('yolo_enabled', False) else "⚠️ OFF"
    
    return (
        frame,
        stats.get('frame_count', 0),
        stats.get('vehicle_count', 0),
        status,
        f"{stats.get('fps', 0):.1f}",
        stats.get('buffer_size', 0),
        yolo
    )


def create_ui():
    """Create Gradio UI"""
    
    with gr.Blocks(title="ATCS Jogja - Stable Stream v3") as demo:
        gr.Markdown("""
        # 🚦 ATCS Jogja - Ultra Stable Video Stream
        
        **Features:**
        - 🎬 Real-time 25-30 FPS smooth streaming
        - 🤖 YOLOv8 Vehicle Detection
        - 📦 Anti-freeze buffer system
        - 🔄 Auto-reconnect on disconnect
        """)
        
        # Status indicators
        with gr.Row():
            yolo_status = gr.Textbox(label="YOLO", value="⚠️ Checking...", interactive=False)
            fps_display = gr.Textbox(label="FPS", value="0.0", interactive=False)
            buffer_display = gr.Number(label="Buffer", value=0)
        
        # Controls
        with gr.Row():
            start_btn = gr.Button("▶️ Start Stream", variant="primary", size="lg")
            stop_btn = gr.Button("⏹️ Stop Stream", variant="secondary", size="lg")
        
        status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
        
        # Video display
        video_display = gr.Image(label="Live Video Feed", height=500)
        
        # Statistics
        with gr.Row():
            frame_count = gr.Number(label="Frames Processed", value=0)
            vehicle_count = gr.Number(label="Vehicles Detected", value=0)
            stream_status = gr.Textbox(label="Stream Status", value="Stopped")
        
        # Auto-refresh timer (40ms = ~25 FPS)
        timer = gr.Timer(0.04, active=True)
        
        # Event handlers
        start_btn.click(start_stream, outputs=[status_text])
        stop_btn.click(stop_stream, outputs=[status_text])
        
        # Auto-update
        timer.tick(
            get_current_frame,
            outputs=[
                video_display, 
                frame_count, 
                vehicle_count, 
                stream_status,
                fps_display,
                buffer_display,
                yolo_status
            ]
        )
        
        # Info
        with gr.Accordion("ℹ️ Information", open=False):
            gr.Markdown(f"""
            **Stream URL:** `{STREAM_URL[:60]}...`
            
            **Configuration:**
            - Resolution: {WIDTH}x{HEIGHT}
            - Buffer: {BUFFER_SECONDS} seconds
            - Model: {YOLO_MODEL}
            
            **Detected Classes:**
            - 🚗 Car
            - 🏍️ Motorcycle  
            - 🚌 Bus
            - 🚚 Truck
            """)
    
    return demo


# ================= MAIN =================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ATCS Jogja - Ultra Stable Stream v3")
    print("=" * 60)
    print(f"Model: {YOLO_MODEL}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Buffer: {BUFFER_SECONDS}s")
    print("=" * 60)
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
