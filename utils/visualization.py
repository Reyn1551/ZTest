"""
Visualization dan Dashboard Renderer untuk ATCS Vision
Render hasil analisis dengan overlay visual yang informatif
"""

import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Color definitions (BGR format)
COLORS = {
    'car': (255, 100, 100),         # Light blue
    'motorcycle': (100, 255, 100),   # Light green
    'bus': (100, 100, 255),          # Light red
    'truck': (255, 255, 100),        # Cyan
    'text': (255, 255, 255),         # White
    'alert': (0, 0, 255),            # Red
    'safe': (0, 255, 0),             # Green
    'warning': (0, 165, 255),        # Orange
    'info': (255, 255, 255),         # White
    'speeding': (0, 0, 255),         # Red
    'helmet_ok': (0, 255, 0),        # Green
    'no_helmet': (0, 0, 255),        # Red
    'trajectory': (255, 200, 0),     # Cyan
    'background': (30, 30, 30),      # Dark gray
}

# Direction arrows
DIRECTION_ARROWS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0),
    'STOPPED': (0, 0),
    'UNKNOWN': (0, 0)
}


@dataclass
class DashboardStats:
    """Statistics for dashboard display"""
    total_vehicles: int = 0
    cars: int = 0
    motorcycles: int = 0
    buses: int = 0
    trucks: int = 0
    violations: int = 0
    speeding: int = 0
    no_helmet: int = 0
    avg_speed: float = 0.0
    max_speed: float = 0.0


class DashboardRenderer:
    """
    Render hasil analisis ke frame dengan overlay visual
    
    Features:
    - Bounding boxes dengan class-specific colors
    - Speed indicators
    - Direction arrows
    - Helmet status badges
    - Violation alerts
    - Statistics panel
    - Trajectory visualization
    """
    
    def __init__(self, model_config=None):
        """
        Initialize renderer
        
        Args:
            model_config: ModelConfig for class names
        """
        self.model_config = model_config
        self.colors = COLORS
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_small = 0.5
        self.font_scale_medium = 0.6
        self.font_scale_large = 0.8
        self.font_thickness = 2
        
        # Show options
        self.show_trajectory = True
        self.show_direction = True
        self.show_speed = True
        self.show_helmet = True
    
    def draw_results(self, 
                     frame: np.ndarray, 
                     analysis_results: List, 
                     stream_stats: Dict[str, Any] = None,
                     show_stats_panel: bool = True) -> np.ndarray:
        """
        Draw all analysis results on frame
        
        Args:
            frame: Input frame
            analysis_results: List of AnalysisResult objects
            stream_stats: Stream statistics dictionary
            show_stats_panel: Whether to show statistics panel
        
        Returns:
            Frame with overlays
        """
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Draw header
        self._draw_header(output, stream_stats)
        
        # Draw each detection
        for result in analysis_results:
            self._draw_detection(output, result)
        
        # Draw footer stats
        if show_stats_panel:
            stats = self._calculate_stats(analysis_results)
            self._draw_footer(output, stats)
        
        return output
    
    def _draw_detection(self, frame: np.ndarray, result):
        """
        Draw single detection result
        
        Args:
            frame: Frame to draw on
            result: AnalysisResult object
        """
        x1, y1, x2, y2 = map(int, result.bbox)
        class_name = result.class_name
        
        # Get class color
        color = self.colors.get(class_name, (200, 200, 200))
        
        # Check for violations - use red
        if result.is_violation:
            box_color = self.colors['alert']
            box_thickness = 3
        else:
            box_color = color
            box_thickness = 2
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
        
        # Draw label background
        label_parts = [f"ID:{result.track_id}"]
        if self.show_speed:
            label_parts.append(f"{result.speed_kmh:.0f}km/h")
        
        label = " ".join(label_parts)
        (tw, th), _ = cv2.getTextSize(label, self.font, self.font_scale_medium, 2)
        
        # Label position
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(frame, (x1, label_y - th - 5), (x1 + tw + 5, label_y), box_color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y - 3), 
                   self.font, self.font_scale_medium, (0, 0, 0), 2)
        
        # Draw speed indicator (colored based on speed)
        if self.show_speed:
            speed_color = self.colors['safe']
            if result.is_speeding:
                speed_color = self.colors['speeding']
            elif result.speed_kmh > 40:
                speed_color = self.colors['warning']
            
            # Speed bar
            bar_width = int(min(result.speed_kmh / 100, 1) * (x2 - x1))
            bar_y = y2 + 5
            cv2.rectangle(frame, (x1, bar_y), (x1 + bar_width, bar_y + 4), speed_color, -1)
        
        # Draw direction arrow
        if self.show_direction:
            self._draw_direction_arrow(frame, result)
        
        # Draw helmet status for motorcycles
        if self.show_helmet and result.class_id == 3:  # motorcycle
            self._draw_helmet_status(frame, result)
        
        # Draw trajectory
        if self.show_trajectory and len(result.trajectory) > 1:
            self._draw_trajectory(frame, result.trajectory)
    
    def _draw_direction_arrow(self, frame: np.ndarray, result):
        """Draw direction indicator"""
        x1, y1, x2, y2 = map(int, result.bbox)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        direction = result.direction
        if direction in DIRECTION_ARROWS:
            dx, dy = DIRECTION_ARROWS[direction]
            arrow_length = 30
            
            end_x = center_x + dx * arrow_length
            end_y = center_y + dy * arrow_length
            
            # Draw arrow
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                          self.colors['trajectory'], 2, tipLength=0.3)
    
    def _draw_helmet_status(self, frame: np.ndarray, result):
        """Draw helmet status badge"""
        x1, y1, x2, y2 = map(int, result.bbox)
        
        helmet_status = result.helmet_status
        helmet_conf = result.helmet_confidence
        
        if helmet_status == "HELMET":
            text = f"HELMET {helmet_conf:.0%}"
            color = self.colors['helmet_ok']
        elif helmet_status == "NO_HELMET":
            text = f"NO HELMET! {helmet_conf:.0%}"
            color = self.colors['no_helmet']
        else:
            text = "HELMET?"
            color = self.colors['warning']
        
        # Position below bbox
        text_y = y2 + 25
        
        # Background
        (tw, th), _ = cv2.getTextSize(text, self.font, self.font_scale_small, 2)
        cv2.rectangle(frame, (x1, text_y - th - 3), (x1 + tw + 5, text_y + 3), color, -1)
        cv2.putText(frame, text, (x1 + 2, text_y), 
                   self.font, self.font_scale_small, (0, 0, 0), 2)
    
    def _draw_trajectory(self, frame: np.ndarray, trajectory: List[Tuple[float, float]]):
        """Draw movement trajectory"""
        if len(trajectory) < 2:
            return
        
        # Draw trajectory line with fading
        points = [tuple(map(int, p)) for p in trajectory]
        
        for i in range(1, len(points)):
            alpha = i / len(points)  # Fade from transparent to opaque
            color = tuple(int(c * alpha) for c in self.colors['trajectory'])
            cv2.line(frame, points[i-1], points[i], color, 2)
    
    def _draw_header(self, frame: np.ndarray, stream_stats: Dict[str, Any] = None):
        """Draw header with system info"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        title = "ATCS JOGJA - AI TRAFFIC SURVEILLANCE"
        cv2.putText(frame, title, (10, 25), 
                   self.font, self.font_scale_large, (0, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 50), 
                   self.font, self.font_scale_small, self.colors['info'], 1)
        
        # Stream stats
        if stream_stats:
            buffer_info = f"Buffer: {stream_stats.get('buffer_size', 0)}/{stream_stats.get('buffer_max', 0)}"
            fps_info = f"FPS: {stream_stats.get('fps_effective', 0):.1f}"
            dropped = f"Dropped: {stream_stats.get('dropped_frames', 0)}"
            
            stats_text = f"{buffer_info} | {fps_info} | {dropped}"
            cv2.putText(frame, stats_text, (w - 400, 50), 
                       self.font, self.font_scale_small, self.colors['info'], 1)
    
    def _draw_footer(self, frame: np.ndarray, stats: DashboardStats):
        """Draw footer with statistics"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Vehicle counts
        count_parts = [
            f"TOTAL: {stats.total_vehicles}",
            f"CARS: {stats.cars}",
            f"MOTOR: {stats.motorcycles}",
            f"BUS: {stats.buses}",
            f"TRUCK: {stats.trucks}"
        ]
        count_text = " | ".join(count_parts)
        cv2.putText(frame, count_text, (10, h - 35), 
                   self.font, self.font_scale_small, self.colors['info'], 1)
        
        # Violations
        violation_parts = [
            f"VIOLATIONS: {stats.violations}",
            f"SPEEDING: {stats.speeding}",
            f"NO HELMET: {stats.no_helmet}"
        ]
        violation_text = " | ".join(violation_parts)
        cv2.putText(frame, violation_text, (10, h - 15), 
                   self.font, self.font_scale_small, self.colors['alert'], 1)
        
        # Speed stats
        speed_text = f"AVG: {stats.avg_speed:.1f} km/h | MAX: {stats.max_speed:.1f} km/h"
        cv2.putText(frame, speed_text, (w - 350, h - 25), 
                   self.font, self.font_scale_small, self.colors['info'], 1)
    
    def _calculate_stats(self, analysis_results: List) -> DashboardStats:
        """Calculate statistics from results"""
        stats = DashboardStats()
        
        if not analysis_results:
            return stats
        
        speeds = []
        
        for result in analysis_results:
            stats.total_vehicles += 1
            
            # Count by class
            if result.class_name == 'car':
                stats.cars += 1
            elif result.class_name == 'motorcycle':
                stats.motorcycles += 1
            elif result.class_name == 'bus':
                stats.buses += 1
            elif result.class_name == 'truck':
                stats.trucks += 1
            
            # Count violations
            if result.is_violation:
                stats.violations += 1
            if result.is_speeding:
                stats.speeding += 1
            if result.helmet_status == "NO_HELMET":
                stats.no_helmet += 1
            
            # Speed stats
            if result.speed_kmh > 0:
                speeds.append(result.speed_kmh)
        
        if speeds:
            stats.avg_speed = np.mean(speeds)
            stats.max_speed = np.max(speeds)
        
        return stats
    
    def draw_violation_alert(self, 
                             frame: np.ndarray, 
                             result) -> np.ndarray:
        """
        Draw prominent violation alert
        
        Args:
            frame: Input frame
            result: AnalysisResult with violation
        
        Returns:
            Frame with alert overlay
        """
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Flash effect
        flash = int(datetime.now().timestamp() * 4) % 2
        if flash:
            # Red border
            cv2.rectangle(output, (0, 0), (w, h), self.colors['alert'], 10)
            
            # Alert text
            alert_text = "VIOLATION DETECTED!"
            (tw, th), _ = cv2.getTextSize(alert_text, self.font, 1.2, 3)
            
            text_x = (w - tw) // 2
            text_y = h // 2
            
            # Background
            cv2.rectangle(output, (text_x - 20, text_y - th - 10), 
                         (text_x + tw + 20, text_y + 20), self.colors['alert'], -1)
            cv2.putText(output, alert_text, (text_x, text_y), 
                       self.font, 1.2, (255, 255, 255), 3)
            
            # Details
            details = []
            if result.is_speeding:
                details.append(f"Speeding: {result.speed_kmh:.0f} km/h")
            if result.helmet_status == "NO_HELMET":
                details.append("No Helmet")
            
            detail_text = " | ".join(details)
            cv2.putText(output, detail_text, (text_x, text_y + 40), 
                       self.font, self.font_scale_medium, (255, 255, 255), 2)
        
        return output
    
    def create_summary_image(self, 
                            frame: np.ndarray,
                            analysis_results: List,
                            title: str = "ATCS Analysis Summary") -> np.ndarray:
        """
        Create summary image with grid layout
        
        Args:
            frame: Input frame
            analysis_results: Analysis results
            title: Summary title
        
        Returns:
            Summary image
        """
        h, w = 1200, 1600
        output = np.zeros((h, w, 3), dtype=np.uint8)
        output[:] = self.colors['background']
        
        # Title
        cv2.putText(output, title, (20, 50), 
                   self.font, 1.5, (0, 255, 255), 3)
        
        # Original frame (resized)
        if frame is not None:
            frame_resized = cv2.resize(frame, (800, 450))
            output[80:530, 20:820] = frame_resized
        
        # Statistics panel
        stats = self._calculate_stats(analysis_results)
        
        # Draw stats boxes
        y_offset = 600
        
        stat_items = [
            ("Total Vehicles", stats.total_vehicles, self.colors['info']),
            ("Cars", stats.cars, self.colors['car']),
            ("Motorcycles", stats.motorcycles, self.colors['motorcycle']),
            ("Buses", stats.buses, self.colors['bus']),
            ("Trucks", stats.trucks, self.colors['truck']),
            ("Violations", stats.violations, self.colors['alert']),
        ]
        
        for i, (label, value, color) in enumerate(stat_items):
            x = 50 + (i % 3) * 250
            y = y_offset + (i // 3) * 100
            
            # Box
            cv2.rectangle(output, (x, y), (x + 200, y + 80), color, -1)
            cv2.rectangle(output, (x, y), (x + 200, y + 80), (255, 255, 255), 2)
            
            # Text
            cv2.putText(output, label, (x + 10, y + 30), 
                       self.font, self.font_scale_small, (0, 0, 0), 2)
            cv2.putText(output, str(value), (x + 10, y + 60), 
                       self.font, 1.2, (0, 0, 0), 3)
        
        return output


# Test standalone
if __name__ == "__main__":
    print("Testing Dashboard Renderer...")
    
    renderer = DashboardRenderer()
    
    # Create test frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)
    
    # Create test results
    from core.analyzer import AnalysisResult
    
    test_results = [
        AnalysisResult(
            track_id=1,
            bbox=[100, 100, 200, 200],
            class_id=3,
            class_name='motorcycle',
            confidence=0.9,
            speed_kmh=45.5,
            direction='RIGHT',
            helmet_status='NO_HELMET',
            helmet_confidence=0.85,
            is_violation=True,
            trajectory=[(100, 150), (120, 150), (140, 150), (160, 150)]
        ),
        AnalysisResult(
            track_id=2,
            bbox=[400, 300, 550, 400],
            class_id=2,
            class_name='car',
            confidence=0.95,
            speed_kmh=35.2,
            direction='DOWN',
            helmet_status='N/A',
            is_violation=False,
            trajectory=[(475, 300), (475, 320), (475, 340), (475, 360)]
        )
    ]
    
    # Render
    stream_stats = {
        'buffer_size': 45,
        'buffer_max': 150,
        'fps_effective': 14.5,
        'dropped_frames': 12
    }
    
    output = renderer.draw_results(frame, test_results, stream_stats)
    
    # Save test output
    cv2.imwrite('/home/z/my-project/atcs_vision/outputs/test_render.jpg', output)
    print("Test render saved to outputs/test_render.jpg")
