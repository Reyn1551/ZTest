"""
YOLOv8 Vehicle Detector untuk ATCS Vision
Mendukung deteksi kendaraan: car, motorcycle, bus, truck
"""

import logging
import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Get area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x1, y1, x2, y2, conf, class_id]"""
        return np.array([*self.bbox, self.confidence, self.class_id])


class VehicleDetector:
    """
    YOLOv8 wrapper untuk deteksi kendaraan
    
    Features:
    - Automatic model download
    - GPU/CPU support
    - Batch processing
    - Class filtering
    - Confidence thresholding
    """
    
    def __init__(self, config):
        """
        Initialize detector
        
        Args:
            config: ModelConfig instance
        """
        self.config = config
        self.model = None
        self.class_names = config.CLASS_NAMES
        self.vehicle_classes = config.VEHICLE_CLASSES
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            
            model_path = self.config.VEHICLE_MODEL
            
            # Try to load model
            logger.info(f"Loading YOLOv8 model: {model_path}")
            self.model = YOLO(model_path)
            
            # Set device
            device = self.config.DEVICE
            if device != "cpu" and not self._check_gpu():
                logger.warning("GPU requested but not available, using CPU")
                device = "cpu"
            
            self.model.to(device)
            logger.info(f"Model loaded successfully on {device}")
            
        except ImportError:
            logger.error("ultralytics not installed! Run: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def detect(self, frame: np.ndarray, 
               return_detections: bool = True) -> List[Detection]:
        """
        Detect vehicles in frame
        
        Args:
            frame: Input image (BGR format)
            return_detections: If True, return Detection objects; else return numpy array
        
        Returns:
            List of Detection objects or numpy array
        """
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.config.CONFIDENCE,
                iou=self.config.IOU,
                classes=self.vehicle_classes,
                verbose=False
            )[0]
            
            # Parse results
            detections = []
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = box
                    
                    class_name = self.class_names.get(cls, f"class_{cls}")
                    
                    detection = Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(conf),
                        class_id=int(cls),
                        class_name=class_name
                    )
                    detections.append(detection)
            
            if return_detections:
                return detections
            else:
                # Return numpy array format
                if detections:
                    return np.array([d.to_array() for d in detections])
                return np.empty((0, 6))
                
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return [] if return_detections else np.empty((0, 6))
    
    def detect_with_tracking(self, frame: np.ndarray) -> List[dict]:
        """
        Detect and track vehicles in single call
        Uses YOLOv8 built-in tracking (ByteTrack)
        
        Args:
            frame: Input image
        
        Returns:
            List of track dictionaries
        """
        if self.model is None:
            return []
        
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.config.CONFIDENCE,
                iou=self.config.IOU,
                classes=self.vehicle_classes,
                verbose=False
            )[0]
            
            tracks = []
            
            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                confidences = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                
                for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                    x1, y1, x2, y2 = box
                    
                    class_name = self.class_names.get(cls, f"class_{cls}")
                    
                    tracks.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'track_id': int(track_id),
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': class_name,
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                    })
            
            return tracks
            
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            return []
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID"""
        return self.class_names.get(class_id, f"class_{class_id}")
    
    def warmup(self, input_size: Tuple[int, int] = (640, 480)):
        """
        Warmup model with dummy inference
        Call this before main loop for faster first inference
        """
        logger.info("Warming up model...")
        dummy_frame = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        self.detect(dummy_frame)
        logger.info("Model warmup complete")


class HelmetDetector:
    """
    Detector untuk helm (placeholder)
    Nanti diganti dengan custom trained YOLOv8 model
    
    Untuk demo, menggunakan heuristic sederhana:
    - Check warna di region helm
    - Mock detection untuk testing
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        # TODO: Load custom helmet model when available
        # if config.HELMET_MODEL:
        #     self.model = YOLO(config.HELMET_MODEL)
    
    def detect_helmet(self, frame: np.ndarray, 
                      motorcycle_bbox: List[float]) -> Tuple[str, float]:
        """
        Check if rider is wearing helmet
        
        Args:
            frame: Full frame
            motorcycle_bbox: [x1, y1, x2, y2] of motorcycle
        
        Returns:
            Tuple of (status, confidence)
            status: "HELMET", "NO_HELMET", or "UNKNOWN"
        """
        x1, y1, x2, y2 = map(int, motorcycle_bbox[:4])
        
        # Extract helmet region (upper portion of motorcycle bbox)
        height = y2 - y1
        helmet_region_top = y1
        helmet_region_bottom = y1 + int(height * self.config.HELMET_REGION_RATIO)
        
        # Ensure valid region
        helmet_region_top = max(0, helmet_region_top)
        helmet_region_bottom = min(frame.shape[0], helmet_region_bottom)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)
        
        if helmet_region_bottom <= helmet_region_top or x2 <= x1:
            return "UNKNOWN", 0.0
        
        # Extract region
        helmet_region = frame[helmet_region_top:helmet_region_bottom, x1:x2]
        
        if helmet_region.size == 0:
            return "UNKNOWN", 0.0
        
        # Heuristic: Check for common helmet colors
        # Convert to HSV for better color detection
        try:
            hsv = cv2.cvtColor(helmet_region, cv2.COLOR_BGR2HSV)
            
            # Check for common helmet colors
            # Black/Dark helmets
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            black_mask = cv2.inRange(hsv, lower_black, upper_black)
            
            # White helmets
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Red helmets
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = red_mask1 | red_mask2
            
            # Blue helmets
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Calculate helmet-like color coverage
            total_pixels = helmet_region.shape[0] * helmet_region.shape[1]
            helmet_pixels = (cv2.countNonZero(black_mask) + 
                           cv2.countNonZero(white_mask) + 
                           cv2.countNonZero(red_mask) +
                           cv2.countNonZero(blue_mask))
            
            coverage = helmet_pixels / total_pixels if total_pixels > 0 else 0
            
            # Heuristic threshold
            if coverage > 0.3:
                return "HELMET", min(0.95, 0.6 + coverage * 0.35)
            else:
                return "NO_HELMET", 0.7
                
        except Exception as e:
            logger.debug(f"Helmet detection error: {e}")
            return "UNKNOWN", 0.0
    
    def detect_helmet_ml(self, frame: np.ndarray,
                         motorcycle_bbox: List[float]) -> Tuple[str, float]:
        """
        Detect helmet using ML model (when available)
        
        Args:
            frame: Full frame
            motorcycle_bbox: [x1, y1, x2, y2] of motorcycle
        
        Returns:
            Tuple of (status, confidence)
        """
        if self.model is None:
            # Fallback to heuristic
            return self.detect_helmet(frame, motorcycle_bbox)
        
        # TODO: Implement ML-based detection
        # 1. Crop motorcycle region
        # 2. Run helmet detection model
        # 3. Return result
        
        x1, y1, x2, y2 = map(int, motorcycle_bbox[:4])
        height = y2 - y1
        helmet_region = frame[y1:y1+int(height*0.4), x1:x2]
        
        try:
            results = self.model(helmet_region, verbose=False)[0]
            
            if results.boxes and len(results.boxes) > 0:
                # Get highest confidence detection
                confs = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                best_idx = np.argmax(confs)
                best_conf = confs[best_idx]
                best_class = int(classes[best_idx])
                
                # Assuming class 0 = helmet, class 1 = no_helmet
                status = "HELMET" if best_class == 0 else "NO_HELMET"
                return status, float(best_conf)
            
            return "UNKNOWN", 0.0
            
        except Exception as e:
            logger.debug(f"ML helmet detection error: {e}")
            return self.detect_helmet(frame, motorcycle_bbox)


# Test standalone
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import MODEL_CONFIG
    
    print("Testing Vehicle Detector...")
    
    detector = VehicleDetector(MODEL_CONFIG)
    detector.warmup()
    
    # Test with dummy image
    test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Draw some rectangles as fake objects
    cv2.rectangle(test_img, (100, 100), (300, 200), (255, 0, 0), -1)
    cv2.rectangle(test_img, (400, 300), (600, 450), (0, 255, 0), -1)
    
    detections = detector.detect(test_img)
    print(f"Detected {len(detections)} objects")
    
    for det in detections:
        print(f"  - {det.class_name}: conf={det.confidence:.2f}, bbox={det.bbox}")
