#!/usr/bin/env python3
"""
ATCS Jogja - Computer Vision System
Speed + Direction + Helmet Detection dengan Robust M3U8 Streaming

Main entry point untuk sistem ATCS Vision

Usage:
    python main.py              # Run dengan OpenCV display
    python main.py --gradio     # Run dengan Gradio web UI
    python main.py --help       # Show help
"""

import sys
import os
import cv2
import time
import argparse
import signal
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('atcs_vision.log')
    ]
)
logger = logging.getLogger(__name__)


class ATCSVisionSystem:
    """
    Main system class yang mengintegrasikan semua komponen
    """
    
    def __init__(self, use_gradio: bool = False, headless: bool = False):
        """
        Initialize system
        
        Args:
            use_gradio: Use Gradio web UI
            headless: Run without display (for servers)
        """
        from config.settings import (
            STREAM_CONFIG, MODEL_CONFIG, SPEED_CONFIG,
            DIRECTION_CONFIG, HELMET_CONFIG, OUTPUT_CONFIG
        )
        
        self.use_gradio = use_gradio
        self.headless = headless
        
        # Config
        self.stream_config = STREAM_CONFIG
        self.model_config = MODEL_CONFIG
        self.speed_config = SPEED_CONFIG
        self.output_config = OUTPUT_CONFIG
        
        # Components
        self.stream = None
        self.detector = None
        self.analyzer = None
        self.renderer = None
        self.reporter = None
        
        # State
        self.is_running = False
        self.video_writer = None
        self.frame_count = 0
        self.prev_time = time.time()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _init_components(self):
        """Initialize all components"""
        logger.info("Initializing components...")
        
        # Stream capture
        from core.stream_capture import create_capture
        self.stream = create_capture(self.stream_config)
        
        # Detector
        from core.detector import VehicleDetector
        self.detector = VehicleDetector(self.model_config)
        self.detector.warmup()
        
        # Analyzer
        from core.analyzer import TrafficAnalyzer
        from config.settings import DIRECTION_CONFIG, HELMET_CONFIG
        self.analyzer = TrafficAnalyzer(
            self.speed_config,
            DIRECTION_CONFIG,
            HELMET_CONFIG,
            self.model_config
        )
        
        # Renderer
        from utils.visualization import DashboardRenderer
        self.renderer = DashboardRenderer(self.model_config)
        
        # Reporter
        from utils.csv_reporter import CSVReporter
        self.reporter = CSVReporter(self.output_config.REPORTS_DIR)
        
        logger.info("All components initialized")
    
    def _init_video_writer(self):
        """Initialize video writer for recording"""
        if not self.output_config.SAVE_VIDEO:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = Path(self.output_config.RECORDINGS_DIR) / f"atcs_{timestamp}.mp4"
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*self.output_config.VIDEO_CODEC)
        self.video_writer = cv2.VideoWriter(
            str(filename),
            fourcc,
            self.output_config.VIDEO_FPS,
            self.stream_config.RESOLUTION
        )
        
        logger.info(f"Recording to: {filename}")
    
    def _save_snapshot(self, frame):
        """Save snapshot"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = Path(self.output_config.SNAPSHOTS_DIR) / f"snap_{timestamp}.jpg"
        filename.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filename), frame)
        logger.info(f"Snapshot saved: {filename}")
    
    def _save_violation_screenshot(self, frame, result):
        """Save violation screenshot"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = Path(self.output_config.VIOLATIONS_DIR) / f"violation_{timestamp}.jpg"
        filename.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filename), frame)
        logger.info(f"Violation screenshot saved: {filename}")
    
    def run_opencv(self):
        """Run with OpenCV display"""
        logger.info("Starting OpenCV mode...")
        
        self._init_components()
        self._init_video_writer()
        
        logger.info("Starting stream...")
        if not self.stream.start():
            logger.error("Failed to start stream!")
            return
        
        self.is_running = True
        logger.info("System running! Press 'q' to quit, 's' to save snapshot")
        
        try:
            while self.is_running:
                # Read frame
                ret, frame = self.stream.read()
                if not ret:
                    logger.warning("Frame read failed, retrying...")
                    continue
                
                # Calculate dt
                current_time = time.time()
                dt = current_time - self.prev_time
                self.prev_time = current_time
                
                # Detect and track
                tracks = self.detector.detect_with_tracking(frame)
                
                # Analyze
                results = self.analyzer.analyze(frame, tracks, dt)
                
                # Render
                stream_stats = self.stream.get_stats()
                display = self.renderer.draw_results(frame, results, stream_stats)
                
                # Record
                if self.video_writer:
                    self.video_writer.write(display)
                
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
                        # Save violation screenshot
                        if self.output_config.SAVE_VIOLATIONS:
                            self._save_violation_screenshot(display, result)
                
                # Show
                if not self.headless:
                    cv2.imshow("ATCS Jogja - AI Surveillance", display)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self._save_snapshot(display)
                
                self.frame_count += 1
                
                # Print stats periodically
                if self.frame_count % 100 == 0:
                    stats = self.stream.get_stats()
                    logger.info(f"Frame {self.frame_count}: FPS={stats.get('fps_effective', 0):.1f}, "
                               f"Buffer={stats.get('buffer_size', 0)}, "
                               f"Dropped={stats.get('dropped_frames', 0)}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def run_gradio(self):
        """Run with Gradio web UI"""
        logger.info("Starting Gradio mode...")
        
        from gradio_app import launch_app
        from config.settings import GRADIO_CONFIG
        
        launch_app(
            share=GRADIO_CONFIG.SHARE,
            server_name=GRADIO_CONFIG.SERVER_NAME,
            server_port=GRADIO_CONFIG.SERVER_PORT
        )
    
    def stop(self):
        """Stop system"""
        logger.info("Stopping system...")
        
        self.is_running = False
        
        if self.stream:
            self.stream.stop()
        
        if self.video_writer:
            self.video_writer.release()
        
        if not self.headless:
            cv2.destroyAllWindows()
        
        # Generate final reports
        if self.reporter and self.frame_count > 0:
            logger.info("Generating final reports...")
            reports = {
                'vehicle': self.reporter.generate_vehicle_report(),
                'violation': self.reporter.generate_violation_report(),
                'summary': self.reporter.generate_summary_report(),
                'esg': self.reporter.generate_esg_report()
            }
            logger.info(f"Reports saved to {self.output_config.REPORTS_DIR}")
        
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info("System stopped")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='ATCS Jogja - AI Traffic Surveillance System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Run with OpenCV display
    python main.py --gradio     # Run with Gradio web UI
    python main.py --headless   # Run without display (for servers)
    python main.py --no-record  # Run without video recording
        """
    )
    
    parser.add_argument(
        '--gradio',
        action='store_true',
        help='Run with Gradio web UI'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without display (for servers)'
    )
    
    parser.add_argument(
        '--no-record',
        action='store_true',
        help='Disable video recording'
    )
    
    parser.add_argument(
        '--url',
        type=str,
        default=None,
        help='Override M3U8 stream URL'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Gradio server port (default: 7860)'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public share link for Gradio'
    )
    
    args = parser.parse_args()
    
    # Override config if URL provided
    if args.url:
        from config.settings import STREAM_CONFIG
        STREAM_CONFIG.M3U8_URL = args.url
        logger.info(f"Using custom URL: {args.url}")
    
    # Override recording
    if args.no_record:
        from config.settings import OUTPUT_CONFIG
        OUTPUT_CONFIG.SAVE_VIDEO = False
        logger.info("Video recording disabled")
    
    # Print banner
    print("=" * 60)
    print("🚦 ATCS JOGJA - AI Traffic Surveillance System")
    print("   Speed + Direction + Helmet Detection")
    print("=" * 60)
    
    # Create and run system
    system = ATCSVisionSystem(
        use_gradio=args.gradio,
        headless=args.headless
    )
    
    if args.gradio:
        # Override Gradio config
        from config.settings import GRADIO_CONFIG
        GRADIO_CONFIG.SERVER_PORT = args.port
        GRADIO_CONFIG.SHARE = args.share
        
        system.run_gradio()
    else:
        system.run_opencv()


if __name__ == "__main__":
    main()
