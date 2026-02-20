"""
Multi-Task Traffic Analyzer untuk ATCS Vision
Speed Estimation + Direction Detection + Helmet Compliance
"""

import logging
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result for single tracked vehicle"""
    track_id: int
    bbox: List[float]
    class_id: int
    class_name: str
    confidence: float
    
    # Speed
    speed_kmh: float = 0.0
    speed_mps: float = 0.0
    is_speeding: bool = False
    
    # Direction
    direction: str = "UNKNOWN"
    direction_detail: str = ""
    
    # Helmet (motorcycle only)
    helmet_status: str = "N/A"
    helmet_confidence: float = 0.0
    is_violation: bool = False
    
    # Tracking info
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    distance_traveled: float = 0.0
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SpeedEstimator:
    """
    Estimasi kecepatan kendaraan dari pergerakan pixel
    
    Metode:
    - Euclidean distance antar frame
    - Kalibrasi pixel-to-meter
    - Exponential Moving Average untuk smoothing
    """
    
    def __init__(self, config):
        """
        Initialize speed estimator
        
        Args:
            config: SpeedConfig instance
        """
        self.config = config
        self.pixel_to_meter = config.PIXEL_TO_METER
        self.fps = config.FPS_STREAM
        self.min_speed = config.MIN_SPEED
        self.max_speed = config.MAX_SPEED
        self.speed_limit = config.SPEED_LIMIT
        
        # History for smoothing
        self.speed_history: Dict[int, List[float]] = defaultdict(list)
        self.position_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        
        # Calibration
        self.calibration_points = config.CALIBRATION_POINTS
        self.is_calibrated = False
    
    def estimate(self, track_id: int, 
                 current_pos: Tuple[float, float],
                 dt: float = 1/30) -> Tuple[float, bool]:
        """
        Estimate speed for a tracked object
        
        Args:
            track_id: Unique track identifier
            current_pos: Current position (x, y)
            dt: Time delta between frames
        
        Returns:
            Tuple of (speed_kmh, is_speeding)
        """
        # Get previous position
        prev_positions = self.position_history[track_id]
        
        if len(prev_positions) < 1:
            self.position_history[track_id].append(current_pos)
            return 0.0, False
        
        prev_pos = prev_positions[-1]
        
        # Calculate displacement
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        
        # Euclidean distance in pixels
        pixel_distance = np.sqrt(dx**2 + dy**2)
        
        # Convert to meters
        meter_distance = pixel_distance * self.pixel_to_meter
        
        # Calculate speed (m/s)
        speed_mps = meter_distance / dt if dt > 0 else 0
        
        # Convert to km/h
        speed_kmh = speed_mps * 3.6
        
        # Add to history for smoothing
        self.speed_history[track_id].append(speed_kmh)
        if len(self.speed_history[track_id]) > self.config.SPEED_HISTORY_LENGTH:
            self.speed_history[track_id].pop(0)
        
        # Smooth using Exponential Moving Average
        smoothed_speed = self._ema_smooth(track_id)
        
        # Filter outliers
        smoothed_speed = np.clip(smoothed_speed, self.min_speed, self.max_speed)
        
        # Update position history
        self.position_history[track_id].append(current_pos)
        if len(self.position_history[track_id]) > self.config.SPEED_HISTORY_LENGTH:
            self.position_history[track_id].pop(0)
        
        # Check if speeding
        is_speeding = smoothed_speed > self.speed_limit
        
        return smoothed_speed, is_speeding
    
    def _ema_smooth(self, track_id: int) -> float:
        """Exponential Moving Average smoothing"""
        history = self.speed_history[track_id]
        if len(history) == 0:
            return 0.0
        
        alpha = self.config.SPEED_SMOOTHING
        ema = history[0]
        
        for speed in history[1:]:
            ema = alpha * speed + (1 - alpha) * ema
        
        return ema
    
    def calibrate(self, reference_distance_meters: float, 
                  point1: Tuple[int, int], 
                  point2: Tuple[int, int]):
        """
        Calibrate pixel-to-meter ratio
        
        Args:
            reference_distance_meters: Known distance between points in meters
            point1: First calibration point (x, y) in pixels
            point2: Second calibration point (x, y) in pixels
        """
        pixel_distance = np.sqrt(
            (point2[0] - point1[0])**2 + 
            (point2[1] - point1[1])**2
        )
        
        self.pixel_to_meter = reference_distance_meters / pixel_distance
        self.is_calibrated = True
        
        logger.info(f"Calibrated: 1 pixel = {self.pixel_to_meter:.4f} meters")
    
    def get_speed_stats(self, track_id: int) -> dict:
        """Get speed statistics for a track"""
        history = self.speed_history.get(track_id, [])
        
        if len(history) == 0:
            return {'avg': 0, 'max': 0, 'min': 0}
        
        return {
            'avg': np.mean(history),
            'max': np.max(history),
            'min': np.min(history),
            'current': history[-1] if history else 0
        }
    
    def clear_track(self, track_id: int):
        """Clear history for a track"""
        if track_id in self.speed_history:
            del self.speed_history[track_id]
        if track_id in self.position_history:
            del self.position_history[track_id]


class DirectionDetector:
    """
    Deteksi arah pergerakan kendaraan
    
    Directions:
    - UP: Menuju kamera
    - DOWN: Menjauhi kamera
    - LEFT: Bergerak ke kiri
    - RIGHT: Bergerak ke kanan
    - STOPPED: Berhenti
    - UNKNOWN: Tidak terdeteksi
    """
    
    def __init__(self, config):
        """
        Initialize direction detector
        
        Args:
            config: DirectionConfig instance
        """
        self.config = config
        self.directions = config.DIRECTIONS
        self.min_trajectory = config.MIN_TRAJECTORY_LENGTH
        self.movement_threshold = config.MOVEMENT_THRESHOLD
        self.horizontal_threshold = config.HORIZONTAL_THRESHOLD
    
    def detect(self, trajectory: List[Tuple[float, float]]) -> Tuple[str, str]:
        """
        Detect direction from trajectory
        
        Args:
            trajectory: List of (x, y) positions
        
        Returns:
            Tuple of (direction, direction_detail)
        """
        if len(trajectory) < self.min_trajectory:
            return "UNKNOWN", "Insufficient trajectory data"
        
        # Use last N points for direction
        recent = np.array(trajectory[-self.min_trajectory:])
        
        # Calculate movement vector
        start = recent[0]
        end = recent[-1]
        vector = end - start
        
        dx, dy = vector
        abs_dx, abs_dy = abs(dx), abs(dy)
        
        # Check if stopped
        total_movement = np.sqrt(dx**2 + dy**2)
        if total_movement < self.movement_threshold:
            return "STOPPED", "Vehicle not moving"
        
        # Determine dominant direction
        direction = "UNKNOWN"
        detail = ""
        
        if abs_dy > abs_dx * 1.5:  # Mostly vertical movement
            if dy > 0:
                direction = "DOWN"
                detail = "Menjauhi kamera"
            else:
                direction = "UP"
                detail = "Menuju kamera"
        
        elif abs_dx > self.horizontal_threshold:  # Horizontal movement
            if dx > 0:
                direction = "RIGHT"
                detail = "Bergerak ke kanan"
            else:
                direction = "LEFT"
                detail = "Bergerak ke kiri"
        
        else:  # Diagonal or small movement
            if abs_dx > abs_dy:
                direction = "RIGHT" if dx > 0 else "LEFT"
                detail = f"Diagonal movement (dx={dx:.1f}, dy={dy:.1f})"
            else:
                direction = "DOWN" if dy > 0 else "UP"
                detail = f"Diagonal movement (dx={dx:.1f}, dy={dy:.1f})"
        
        return direction, detail
    
    def get_direction_angle(self, trajectory: List[Tuple[float, float]]) -> float:
        """
        Get movement angle in degrees
        
        Args:
            trajectory: List of (x, y) positions
        
        Returns:
            Angle in degrees (0 = right, 90 = down, etc.)
        """
        if len(trajectory) < 2:
            return 0.0
        
        start = np.array(trajectory[0])
        end = np.array(trajectory[-1])
        vector = end - start
        
        angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
        return angle


class TrafficAnalyzer:
    """
    Multi-task analyzer untuk traffic analysis
    
    Combines:
    - Speed estimation
    - Direction detection
    - Helmet compliance
    """
    
    def __init__(self, speed_config, direction_config, helmet_config, model_config):
        """
        Initialize traffic analyzer
        
        Args:
            speed_config: SpeedConfig
            direction_config: DirectionConfig
            helmet_config: HelmetConfig
            model_config: ModelConfig
        """
        self.speed_estimator = SpeedEstimator(speed_config)
        self.direction_detector = DirectionDetector(direction_config)
        self.helmet_config = helmet_config
        self.model_config = model_config
        
        # Helmet detector (lazy load)
        self._helmet_detector = None
        
        # Statistics
        self.analysis_count = 0
        self.violations_detected = 0
        
        # Violation log
        self.violations: List[dict] = []
    
    @property
    def helmet_detector(self):
        """Lazy load helmet detector"""
        if self._helmet_detector is None:
            from .detector import HelmetDetector
            self._helmet_detector = HelmetDetector(self.helmet_config)
        return self._helmet_detector
    
    def analyze(self, frame: np.ndarray, 
                tracks: List[dict], 
                dt: float = 1/30) -> List[AnalysisResult]:
        """
        Analyze all tracks in frame
        
        Args:
            frame: Current frame
            tracks: List of track dictionaries from tracker
            dt: Time delta between frames
        
        Returns:
            List of AnalysisResult objects
        """
        results = []
        self.analysis_count += 1
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            class_id = track['class_id']
            confidence = track['confidence']
            center = track.get('center', self._get_center(bbox))
            trajectory = track.get('trajectory', [center])
            
            class_name = self.model_config.CLASS_NAMES.get(class_id, f"class_{class_id}")
            
            # 1. Speed estimation
            speed_kmh, is_speeding = self.speed_estimator.estimate(
                track_id, center, dt
            )
            speed_mps = speed_kmh / 3.6
            
            # 2. Direction detection
            direction, direction_detail = self.direction_detector.detect(trajectory)
            
            # 3. Helmet check (motorcycle only)
            helmet_status = "N/A"
            helmet_confidence = 0.0
            
            if class_id == self.helmet_config.MOTORCYCLE_CLASS_ID:
                helmet_status, helmet_confidence = self.helmet_detector.detect_helmet(
                    frame, bbox
                )
            
            # Check violations
            is_violation = (
                is_speeding or 
                (helmet_status == "NO_HELMET")
            )
            
            if is_violation:
                self._log_violation(track_id, class_name, speed_kmh, helmet_status)
            
            # Create result
            result = AnalysisResult(
                track_id=track_id,
                bbox=bbox,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                speed_kmh=speed_kmh,
                speed_mps=speed_mps,
                is_speeding=is_speeding,
                direction=direction,
                direction_detail=direction_detail,
                helmet_status=helmet_status,
                helmet_confidence=helmet_confidence,
                is_violation=is_violation,
                trajectory=trajectory,
                distance_traveled=self._calculate_distance(trajectory)
            )
            
            results.append(result)
        
        return results
    
    def _get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, trajectory: List[Tuple[float, float]]) -> float:
        """Calculate total distance traveled"""
        if len(trajectory) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1]
            p2 = trajectory[i]
            total += np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        
        return total * self.speed_estimator.pixel_to_meter
    
    def _log_violation(self, track_id: int, class_name: str, 
                       speed: float, helmet_status: str):
        """Log violation for reporting"""
        violation = {
            'timestamp': datetime.now().isoformat(),
            'track_id': track_id,
            'class_name': class_name,
            'speed_kmh': speed,
            'helmet_status': helmet_status,
            'type': []
        }
        
        if speed > self.speed_estimator.speed_limit:
            violation['type'].append('SPEEDING')
        if helmet_status == "NO_HELMET":
            violation['type'].append('NO_HELMET')
        
        self.violations.append(violation)
        self.violations_detected += 1
        
        # Keep only last 1000 violations
        if len(self.violations) > 1000:
            self.violations = self.violations[-1000:]
    
    def get_statistics(self) -> dict:
        """Get analysis statistics"""
        return {
            'analysis_count': self.analysis_count,
            'violations_detected': self.violations_detected,
            'active_tracks': len(self.speed_estimator.speed_history),
            'recent_violations': len([v for v in self.violations 
                                     if self._is_recent(v['timestamp'])])
        }
    
    def _is_recent(self, timestamp: str, seconds: int = 60) -> bool:
        """Check if timestamp is recent"""
        try:
            ts = datetime.fromisoformat(timestamp)
            return (datetime.now() - ts).total_seconds() < seconds
        except:
            return False
    
    def get_violations(self, limit: int = 100) -> List[dict]:
        """Get recent violations"""
        return self.violations[-limit:]
    
    def clear_track(self, track_id: int):
        """Clear all data for a track"""
        self.speed_estimator.clear_track(track_id)
    
    def reset(self):
        """Reset analyzer state"""
        self.speed_estimator.speed_history.clear()
        self.speed_estimator.position_history.clear()
        self.analysis_count = 0
        self.violations_detected = 0
        self.violations.clear()


# Test standalone
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import SPEED_CONFIG, DIRECTION_CONFIG, HELMET_CONFIG, MODEL_CONFIG
    
    print("Testing Traffic Analyzer...")
    
    analyzer = TrafficAnalyzer(SPEED_CONFIG, DIRECTION_CONFIG, HELMET_CONFIG, MODEL_CONFIG)
    
    # Simulate tracks
    tracks = [
        {
            'track_id': 1,
            'bbox': [100, 100, 200, 200],
            'class_id': 3,  # motorcycle
            'confidence': 0.9,
            'trajectory': [(100, 100), (110, 105), (120, 110), (130, 115)]
        },
        {
            'track_id': 2,
            'bbox': [300, 300, 400, 400],
            'class_id': 2,  # car
            'confidence': 0.95,
            'trajectory': [(300, 300), (300, 320), (300, 340), (300, 360)]
        }
    ]
    
    # Create dummy frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    results = analyzer.analyze(frame, tracks)
    
    for result in results:
        print(f"\nTrack {result.track_id} ({result.class_name}):")
        print(f"  Speed: {result.speed_kmh:.1f} km/h")
        print(f"  Direction: {result.direction}")
        print(f"  Helmet: {result.helmet_status}")
        print(f"  Violation: {result.is_violation}")
    
    print(f"\nStatistics: {analyzer.get_statistics()}")
