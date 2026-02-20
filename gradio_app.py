"""
Gradio UI untuk ATCS Vision System
Interactive web interface untuk traffic surveillance
"""

import gradio as gr
import numpy as np
import cv2
import threading
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATCSVisionApp:
    """
    Main application class yang mengintegrasikan semua komponen
    dengan Gradio UI
    """
    
    def __init__(self):
        """Initialize application"""
        from config.settings import (
            STREAM_CONFIG, MODEL_CONFIG, SPEED_CONFIG,
            DIRECTION_CONFIG, HELMET_CONFIG, OUTPUT_CONFIG, GRADIO_CONFIG
        )
        
        self.stream_config = STREAM_CONFIG
        self.model_config = MODEL_CONFIG
        self.speed_config = SPEED_CONFIG
        self.direction_config = DIRECTION_CONFIG
        self.helmet_config = HELMET_CONFIG
        self.output_config = OUTPUT_CONFIG
        self.gradio_config = GRADIO_CONFIG
        
        # Components (lazy load)
        self._stream_capture = None
        self._detector = None
        self._analyzer = None
        self._renderer = None
        self._reporter = None
        
        # State
        self.is_running = False
        self.processing_thread = None
        self.current_frame = None
        self.current_results = []
        self.stats = {}
        
        # Locks
        self._frame_lock = threading.Lock()
        self._stats_lock = threading.Lock()
    
    @property
    def stream_capture(self):
        """Lazy load stream capture"""
        if self._stream_capture is None:
            from core.stream_capture import create_capture
            self._stream_capture = create_capture(self.stream_config)
        return self._stream_capture
    
    @property
    def detector(self):
        """Lazy load detector"""
        if self._detector is None:
            from core.detector import VehicleDetector
            self._detector = VehicleDetector(self.model_config)
        return self._detector
    
    @property
    def analyzer(self):
        """Lazy load analyzer"""
        if self._analyzer is None:
            from core.analyzer import TrafficAnalyzer
            self._analyzer = TrafficAnalyzer(
                self.speed_config,
                self.direction_config,
                self.helmet_config,
                self.model_config
            )
        return self._analyzer
    
    @property
    def renderer(self):
        """Lazy load renderer"""
        if self._renderer is None:
            from utils.visualization import DashboardRenderer
            self._renderer = DashboardRenderer(self.model_config)
        return self._renderer
    
    @property
    def reporter(self):
        """Lazy load reporter"""
        if self._reporter is None:
            from utils.csv_reporter import CSVReporter
            self._reporter = CSVReporter(self.output_config.REPORTS_DIR)
        return self._reporter
    
    def start_stream(self):
        """Start stream capture"""
        if self.is_running:
            return "Stream already running"
        
        logger.info("Starting stream...")
        
        if not self.stream_capture.start():
            return "Failed to start stream"
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        return "Stream started successfully"
    
    def stop_stream(self):
        """Stop stream capture"""
        if not self.is_running:
            return "Stream not running"
        
        logger.info("Stopping stream...")
        
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=3)
        
        self.stream_capture.stop()
        
        return "Stream stopped"
    
    def _processing_loop(self):
        """Main processing loop"""
        logger.info("Processing loop started")
        
        prev_time = time.time()
        frame_count = 0
        
        while self.is_running:
            try:
                # Read frame
                ret, frame = self.stream_capture.read()
                
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Calculate dt
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time
                
                # Detect and track
                tracks = self.detector.detect_with_tracking(frame)
                
                # Analyze
                results = self.analyzer.analyze(frame, tracks, dt)
                
                # Render
                stream_stats = self.stream_capture.get_stats()
                display = self.renderer.draw_results(frame, results, stream_stats)
                
                # Update state
                with self._frame_lock:
                    self.current_frame = display
                    self.current_results = results
                
                # Update stats
                with self._stats_lock:
                    self.stats = {
                        'stream': stream_stats,
                        'analysis': self.analyzer.get_statistics(),
                        'tracking': {
                            'active_tracks': len(tracks),
                            'frame_count': frame_count
                        }
                    }
                
                # Log to reporter
                for result in results:
                    result_dict = {
                        'track_id': result.track_id,
                        'class_name': result.class_name,
                        'class_id': result.class_id,
                        'speed_kmh': result.speed_kmh,
                        'direction': result.direction,
                        'helmet_status': result.helmet_status,
                        'is_violation': result.is_violation,
                        'is_speeding': result.is_speeding,
                        'bbox': result.bbox,
                        'confidence': result.confidence
                    }
                    self.reporter.log_vehicle(result_dict)
                    if result.is_violation:
                        self.reporter.log_violation(result_dict)
                
                frame_count += 1
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
        
        logger.info("Processing loop stopped")
    
    def get_current_frame(self):
        """Get current frame for display"""
        with self._frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_current_stats(self):
        """Get current statistics"""
        with self._stats_lock:
            return self.stats.copy()
    
    def get_violations(self):
        """Get recent violations"""
        return self.reporter.get_violations(limit=20)
    
    def generate_reports(self):
        """Generate all reports"""
        vehicle_report = self.reporter.generate_vehicle_report()
        violation_report = self.reporter.generate_violation_report()
        summary_report = self.reporter.generate_summary_report()
        esg_report = self.reporter.generate_esg_report()
        
        return {
            'vehicle_report': vehicle_report,
            'violation_report': violation_report,
            'summary_report': summary_report,
            'esg_report': esg_report
        }


def create_gradio_interface():
    """Create Gradio interface"""
    
    app = ATCSVisionApp()
    
    # Custom CSS
    custom_css = """
    .header-text {
        text-align: center;
        margin-bottom: 20px;
    }
    .stats-box {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .violation-box {
        background-color: #2d132c;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .status-running {
        color: #00ff00;
        font-weight: bold;
    }
    .status-stopped {
        color: #ff0000;
        font-weight: bold;
    }
    """
    
    # Create interface
    with gr.Blocks(css=custom_css, title="ATCS Jogja - AI Traffic Surveillance") as demo:
        
        # Header
        gr.Markdown("""
        # 🚦 ATCS Jogja - AI Traffic Surveillance System
        
        **Speed Detection | Direction Analysis | Helmet Compliance**
        
        Sistem pemantauan lalu lintas cerdas menggunakan CCTV ATCS Jogja dengan AI Computer Vision.
        """)
        
        # Status
        with gr.Row():
            status_text = gr.Textbox(label="Status", value="Stopped", interactive=False)
            start_btn = gr.Button("▶️ Start Stream", variant="primary")
            stop_btn = gr.Button("⏹️ Stop Stream", variant="secondary")
        
        # Main display
        with gr.Row():
            with gr.Column(scale=2):
                video_display = gr.Image(label="Live Feed", height=500)
            
            with gr.Column(scale=1):
                # Statistics panel
                gr.Markdown("### 📊 Statistics")
                
                with gr.Group():
                    total_vehicles = gr.Number(label="Total Vehicles", value=0)
                    current_speed = gr.Number(label="Avg Speed (km/h)", value=0)
                    violations_count = gr.Number(label="Violations", value=0)
                
                with gr.Group():
                    gr.Markdown("#### Vehicle Count")
                    cars_count = gr.Number(label="Cars", value=0)
                    motorcycles_count = gr.Number(label="Motorcycles", value=0)
                    buses_count = gr.Number(label="Buses", value=0)
                    trucks_count = gr.Number(label="Trucks", value=0)
        
        # Violations log
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚠️ Recent Violations")
                violations_display = gr.Dataframe(
                    headers=["Time", "Type", "Class", "Speed", "Helmet"],
                    datatype=["str", "str", "str", "number", "str"],
                    row_count=10,
                    col_count=(5, "fixed"),
                    label="Violations Log"
                )
        
        # Reports
        with gr.Row():
            generate_report_btn = gr.Button("📋 Generate Reports", variant="secondary")
            report_status = gr.Textbox(label="Report Status", interactive=False)
        
        # Stream info
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📡 Stream Info")
                buffer_status = gr.Textbox(label="Buffer", value="0/0", interactive=False)
                fps_status = gr.Textbox(label="FPS", value="0.0", interactive=False)
                dropped_frames = gr.Textbox(label="Dropped Frames", value="0", interactive=False)
        
        # Configuration display
        with gr.Accordion("⚙️ Configuration", open=False):
            gr.Markdown(f"""
            **Stream URL:** `{app.stream_config.M3U8_URL}`
            
            **Resolution:** {app.stream_config.RESOLUTION}
            
            **Target FPS:** {app.stream_config.TARGET_FPS}
            
            **Speed Limit:** {app.speed_config.SPEED_LIMIT} km/h
            
            **Model:** {app.model_config.VEHICLE_MODEL}
            """)
        
        # Event handlers
        def start_stream_handler():
            result = app.start_stream()
            return result
        
        def stop_stream_handler():
            result = app.stop_stream()
            return result
        
        def update_frame():
            frame = app.get_current_frame()
            stats = app.get_current_stats()
            
            # Default values
            total = 0
            avg_speed = 0
            violations = 0
            cars = 0
            motorcycles = 0
            buses = 0
            trucks = 0
            buffer_val = "0/0"
            fps_val = "0.0"
            dropped_val = "0"
            status = "Running" if app.is_running else "Stopped"
            
            if stats:
                stream_stats = stats.get('stream', {})
                analysis_stats = stats.get('analysis', {})
                tracking_stats = stats.get('tracking', {})
                
                # Stream stats
                buffer_val = f"{stream_stats.get('buffer_size', 0)}/{stream_stats.get('buffer_max', 0)}"
                fps_val = f"{stream_stats.get('fps_effective', 0):.1f}"
                dropped_val = str(stream_stats.get('dropped_frames', 0))
                
                # Analysis stats
                violations = analysis_stats.get('violations_detected', 0)
                
                # Get from current results
                results = app.current_results if hasattr(app, 'current_results') else []
                total = len(results)
                
                speeds = [r.speed_kmh for r in results if r.speed_kmh > 0]
                avg_speed = sum(speeds) / len(speeds) if speeds else 0
                
                cars = len([r for r in results if r.class_name == 'car'])
                motorcycles = len([r for r in results if r.class_name == 'motorcycle'])
                buses = len([r for r in results if r.class_name == 'bus'])
                trucks = len([r for r in results if r.class_name == 'truck'])
            
            return (
                frame,
                status,
                total,
                round(avg_speed, 1),
                violations,
                cars,
                motorcycles,
                buses,
                trucks,
                buffer_val,
                fps_val,
                dropped_val
            )
        
        def update_violations():
            violations = app.get_violations()
            
            data = []
            for v in violations[-10:]:  # Last 10
                timestamp = v.get('timestamp', '')[-8:]  # Just time
                v_types = v.get('violation_types', '')
                class_name = v.get('class_name', '')
                speed = round(v.get('speed_kmh', 0), 1)
                helmet = v.get('helmet_status', '')
                
                data.append([timestamp, v_types, class_name, speed, helmet])
            
            return data
        
        def generate_reports_handler():
            reports = app.generate_reports()
            return f"Reports generated:\n- {reports['vehicle_report']}\n- {reports['violation_report']}\n- {reports['esg_report']}"
        
        # Connect handlers
        start_btn.click(start_stream_handler, outputs=[status_text])
        stop_btn.click(stop_stream_handler, outputs=[status_text])
        generate_report_btn.click(generate_reports_handler, outputs=[report_status])
        
        # Auto-update
        demo.load(
            update_frame,
            outputs=[
                video_display,
                status_text,
                total_vehicles,
                current_speed,
                violations_count,
                cars_count,
                motorcycles_count,
                buses_count,
                trucks_count,
                buffer_status,
                fps_status,
                dropped_frames
            ],
            every=0.5  # Update every 500ms
        )
        
        demo.load(update_violations, outputs=[violations_display], every=2.0)
    
    return demo


def launch_app(share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
    """
    Launch the Gradio application
    
    Args:
        share: Create public share link
        server_name: Server name
        server_port: Server port
    """
    print("=" * 60)
    print("🚀 ATCS JOGJA - AI Traffic Surveillance System")
    print("=" * 60)
    print(f"Server: {server_name}:{server_port}")
    print(f"Share: {share}")
    print("=" * 60)
    
    demo = create_gradio_interface()
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    launch_app()
