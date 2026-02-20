"""
Calibration utilities untuk ATCS Vision
Pixel-to-meter calibration dan perspective correction
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationPoint:
    """Single calibration point"""
    pixel_coords: Tuple[int, int]
    real_coords: Tuple[float, float]  # in meters
    description: str = ""


@dataclass
class CalibrationData:
    """Complete calibration data"""
    points: List[CalibrationPoint]
    pixel_to_meter: float
    perspective_matrix: Optional[np.ndarray] = None
    reference_distance: float = 0.0
    camera_height: float = 0.0
    camera_angle: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = {
            'points': [asdict(p) for p in self.points],
            'pixel_to_meter': self.pixel_to_meter,
            'reference_distance': self.reference_distance,
            'camera_height': self.camera_height,
            'camera_angle': self.camera_angle,
            'notes': self.notes
        }
        if self.perspective_matrix is not None:
            data['perspective_matrix'] = self.perspective_matrix.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationData':
        """Create from dictionary"""
        points = [CalibrationPoint(**p) for p in data.get('points', [])]
        
        calib = cls(
            points=points,
            pixel_to_meter=data.get('pixel_to_meter', 0.05),
            reference_distance=data.get('reference_distance', 0),
            camera_height=data.get('camera_height', 0),
            camera_angle=data.get('camera_angle', 0),
            notes=data.get('notes', '')
        )
        
        if 'perspective_matrix' in data:
            calib.perspective_matrix = np.array(data['perspective_matrix'])
        
        return calib


class Calibrator:
    """
    Calibration tool untuk menghitung pixel-to-meter ratio
    dan perspective transformation
    """
    
    def __init__(self, calibration_file: str = "data/calibration_data.json"):
        """
        Initialize calibrator
        
        Args:
            calibration_file: Path to calibration JSON file
        """
        self.calibration_file = Path(calibration_file)
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.calibration_data: Optional[CalibrationData] = None
        self.temp_points: List[Tuple[int, int]] = []
        
        self._load_calibration()
    
    def _load_calibration(self):
        """Load calibration from file"""
        if self.calibration_file.exists():
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                self.calibration_data = CalibrationData.from_dict(data)
                logger.info(f"Calibration loaded: {len(self.calibration_data.points)} points")
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")
    
    def save_calibration(self):
        """Save calibration to file"""
        if self.calibration_data is None:
            logger.warning("No calibration data to save")
            return
        
        with open(self.calibration_file, 'w') as f:
            json.dump(self.calibration_data.to_dict(), f, indent=2)
        
        logger.info(f"Calibration saved to {self.calibration_file}")
    
    def add_calibration_point(self, 
                              pixel_coords: Tuple[int, int],
                              real_coords: Tuple[float, float],
                              description: str = ""):
        """
        Add a calibration point
        
        Args:
            pixel_coords: (x, y) in pixels
            real_coords: (x, y) in meters
            description: Optional description
        """
        if self.calibration_data is None:
            self.calibration_data = CalibrationData(
                points=[],
                pixel_to_meter=0.05
            )
        
        point = CalibrationPoint(
            pixel_coords=pixel_coords,
            real_coords=real_coords,
            description=description
        )
        
        self.calibration_data.points.append(point)
        
        # Recalculate pixel-to-meter if we have at least 2 points
        if len(self.calibration_data.points) >= 2:
            self._calculate_pixel_to_meter()
        
        logger.info(f"Calibration point added: {pixel_coords} -> {real_coords}")
    
    def _calculate_pixel_to_meter(self):
        """Calculate average pixel-to-meter ratio from calibration points"""
        if self.calibration_data is None or len(self.calibration_data.points) < 2:
            return
        
        ratios = []
        points = self.calibration_data.points
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1 = points[i]
                p2 = points[j]
                
                # Pixel distance
                pixel_dist = np.sqrt(
                    (p2.pixel_coords[0] - p1.pixel_coords[0])**2 +
                    (p2.pixel_coords[1] - p1.pixel_coords[1])**2
                )
                
                # Real distance
                real_dist = np.sqrt(
                    (p2.real_coords[0] - p1.real_coords[0])**2 +
                    (p2.real_coords[1] - p1.real_coords[1])**2
                )
                
                if pixel_dist > 0 and real_dist > 0:
                    ratios.append(real_dist / pixel_dist)
        
        if ratios:
            self.calibration_data.pixel_to_meter = np.mean(ratios)
            logger.info(f"Calculated pixel-to-meter: {self.calibration_data.pixel_to_meter:.6f}")
    
    def calibrate_from_reference(self,
                                  point1: Tuple[int, int],
                                  point2: Tuple[int, int],
                                  real_distance: float):
        """
        Quick calibration using two reference points
        
        Args:
            point1: First point (x, y) in pixels
            point2: Second point (x, y) in pixels
            real_distance: Known distance between points in meters
        """
        pixel_distance = np.sqrt(
            (point2[0] - point1[0])**2 +
            (point2[1] - point1[1])**2
        )
        
        pixel_to_meter = real_distance / pixel_distance
        
        self.calibration_data = CalibrationData(
            points=[
                CalibrationPoint(point1, (0, 0), "Reference point 1"),
                CalibrationPoint(point2, (real_distance, 0), "Reference point 2")
            ],
            pixel_to_meter=pixel_to_meter,
            reference_distance=real_distance
        )
        
        logger.info(f"Calibrated: 1 pixel = {pixel_to_meter:.6f} meters")
        
        return pixel_to_meter
    
    def get_pixel_to_meter(self) -> float:
        """Get current pixel-to-meter ratio"""
        if self.calibration_data is None:
            return 0.05  # Default
        return self.calibration_data.pixel_to_meter
    
    def set_perspective_transform(self, 
                                   src_points: List[Tuple[int, int]],
                                   dst_points: List[Tuple[int, int]]):
        """
        Set perspective transformation matrix
        
        Args:
            src_points: Source points in image (4 points)
            dst_points: Destination points (4 points)
        """
        if len(src_points) != 4 or len(dst_points) != 4:
            raise ValueError("Need exactly 4 points for perspective transform")
        
        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(src, dst)
        
        if self.calibration_data is None:
            self.calibration_data = CalibrationData(points=[], pixel_to_meter=0.05)
        
        self.calibration_data.perspective_matrix = matrix
        logger.info("Perspective transformation matrix set")
    
    def apply_perspective(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Apply perspective transformation to a point
        
        Args:
            point: (x, y) point
        
        Returns:
            Transformed point or None if no transformation set
        """
        if self.calibration_data is None or self.calibration_data.perspective_matrix is None:
            return None
        
        pts = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.calibration_data.perspective_matrix)
        
        return tuple(transformed[0][0])
    
    def pixels_to_meters(self, pixel_distance: float) -> float:
        """
        Convert pixel distance to meters
        
        Args:
            pixel_distance: Distance in pixels
        
        Returns:
            Distance in meters
        """
        return pixel_distance * self.get_pixel_to_meter()
    
    def meters_to_pixels(self, meter_distance: float) -> float:
        """
        Convert meter distance to pixels
        
        Args:
            meter_distance: Distance in meters
        
        Returns:
            Distance in pixels
        """
        ratio = self.get_pixel_to_meter()
        if ratio == 0:
            return 0
        return meter_distance / ratio
    
    def estimate_speed(self, 
                       pixel_displacement: float, 
                       time_delta: float) -> float:
        """
        Estimate speed from pixel displacement
        
        Args:
            pixel_displacement: Displacement in pixels
            time_delta: Time in seconds
        
        Returns:
            Speed in km/h
        """
        meter_distance = self.pixels_to_meters(pixel_displacement)
        speed_mps = meter_distance / time_delta if time_delta > 0 else 0
        speed_kmh = speed_mps * 3.6
        return speed_kmh
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration information"""
        if self.calibration_data is None:
            return {
                'calibrated': False,
                'pixel_to_meter': 0.05,
                'points_count': 0
            }
        
        return {
            'calibrated': True,
            'pixel_to_meter': self.calibration_data.pixel_to_meter,
            'points_count': len(self.calibration_data.points),
            'reference_distance': self.calibration_data.reference_distance,
            'has_perspective': self.calibration_data.perspective_matrix is not None,
            'points': [asdict(p) for p in self.calibration_data.points]
        }


class InteractiveCalibrator:
    """
    Interactive calibration tool using mouse clicks
    """
    
    def __init__(self, calibrator: Calibrator):
        """
        Initialize interactive calibrator
        
        Args:
            calibrator: Calibrator instance
        """
        self.calibrator = calibrator
        self.click_points: List[Tuple[int, int]] = []
        self.current_frame: Optional[np.ndarray] = None
        self.window_name = "Calibration Tool"
        self.is_calibrating = False
    
    def start_interactive_calibration(self, frame: np.ndarray):
        """
        Start interactive calibration on a frame
        
        Args:
            frame: Frame to calibrate on
        """
        self.current_frame = frame.copy()
        self.click_points = []
        self.is_calibrating = True
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "=" * 50)
        print("CALIBRATION MODE")
        print("=" * 50)
        print("1. Click on a reference point (e.g., road marking)")
        print("2. Click on another reference point")
        print("3. Enter the real distance between them (meters)")
        print("4. Press 'c' to confirm, 'r' to reset, 'q' to quit")
        print("=" * 50)
        
        while self.is_calibrating:
            display = self.current_frame.copy()
            
            # Draw clicked points
            for i, pt in enumerate(self.click_points):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                cv2.putText(display, f"P{i+1}", (pt[0]+10, pt[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw line between points
            if len(self.click_points) == 2:
                cv2.line(display, self.click_points[0], self.click_points[1],
                        (0, 255, 255), 2)
            
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.is_calibrating = False
            elif key == ord('r'):
                self.click_points = []
            elif key == ord('c') and len(self.click_points) == 2:
                # Get distance from user
                try:
                    distance = float(input("Enter real distance (meters): "))
                    self.calibrator.calibrate_from_reference(
                        self.click_points[0],
                        self.click_points[1],
                        distance
                    )
                    self.calibrator.save_calibration()
                    print(f"Calibration saved! 1 pixel = {self.calibrator.get_pixel_to_meter():.6f} meters")
                except ValueError:
                    print("Invalid input, please enter a number")
        
        cv2.destroyWindow(self.window_name)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for clicking points"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.click_points) < 2:
                self.click_points.append((x, y))
                print(f"Point {len(self.click_points)}: ({x}, {y})")


# Test standalone
if __name__ == "__main__":
    print("Testing Calibrator...")
    
    calibrator = Calibrator("data/test_calibration.json")
    
    # Test quick calibration
    ratio = calibrator.calibrate_from_reference(
        point1=(100, 500),
        point2=(500, 500),
        real_distance=10.0  # 10 meters between points
    )
    
    print(f"Pixel to meter ratio: {ratio:.6f}")
    print(f"100 pixels = {calibrator.pixels_to_meters(100):.2f} meters")
    print(f"Speed at 200 pixels/second = {calibrator.estimate_speed(200, 1.0):.2f} km/h")
    
    calibrator.save_calibration()
    print(f"Calibration info: {calibrator.get_calibration_info()}")
