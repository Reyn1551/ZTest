"""
Vehicle Tracking dengan ByteTrack untuk ATCS Vision
Mendukung trajectory tracking untuk speed dan direction estimation
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Single track information"""
    track_id: int
    bbox: Tuple[float, float, float, float]
    class_id: int
    confidence: float
    center: Tuple[float, float]
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    speed: float = 0.0
    direction: str = "UNKNOWN"
    helmet_status: str = "N/A"
    first_seen: int = 0
    last_seen: int = 0
    age: int = 0
    hit_streak: int = 0


class VehicleTracker:
    """
    ByteTrack-style tracking untuk kendaraan
    
    Features:
    - Track persistence across frames
    - Trajectory history untuk direction
    - Track management (create, update, delete)
    - Integration dengan YOLOv8 tracking
    """
    
    def __init__(self, config=None):
        """
        Initialize tracker
        
        Args:
            config: Optional configuration
        """
        self.tracks: Dict[int, Track] = {}
        self.trajectories: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.max_history = 50  # Maximum trajectory length
        self.min_hits = 3      # Minimum detections before track is confirmed
        self.max_age = 30      # Maximum frames to keep lost track
        
        self.frame_count = 0
        self.next_id = 1
        
        # Statistics
        self.total_tracks = 0
        self.active_tracks = 0
    
    def update(self, detections: List[dict], frame_idx: Optional[int] = None) -> List[Track]:
        """
        Update tracks dengan detections baru
        
        Args:
            detections: List of detection dictionaries from detector
            frame_idx: Optional frame index
        
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        if frame_idx is None:
            frame_idx = self.frame_count
        
        # Age all tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id].age += 1
        
        # Match detections to existing tracks
        matched = set()
        
        for det in detections:
            det_center = det.get('center', self._get_center(det['bbox']))
            best_match_id = None
            best_distance = float('inf')
            
            # Find closest track
            for track_id, track in self.tracks.items():
                if track_id in matched:
                    continue
                
                # Check class match
                if track.class_id != det.get('class_id', -1):
                    continue
                
                # Calculate distance
                track_center = track.center
                distance = np.sqrt(
                    (det_center[0] - track_center[0])**2 + 
                    (det_center[1] - track_center[1])**2
                )
                
                # Threshold based on bbox size
                bbox_size = max(det['bbox'][2] - det['bbox'][0], 
                              det['bbox'][3] - det['bbox'][1])
                threshold = bbox_size * 0.5
                
                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                self._update_track(best_match_id, det, frame_idx)
                matched.add(best_match_id)
            else:
                # Create new track
                new_id = det.get('track_id', self.next_id)
                if new_id >= self.next_id:
                    self.next_id = new_id + 1
                self._create_track(new_id, det, frame_idx)
        
        # Remove old tracks
        self._remove_old_tracks()
        
        # Update statistics
        self.active_tracks = len([t for t in self.tracks.values() if t.age < 5])
        
        return self.get_active_tracks()
    
    def _get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate center of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _create_track(self, track_id: int, detection: dict, frame_idx: int):
        """Create new track"""
        bbox = detection['bbox']
        center = detection.get('center', self._get_center(bbox))
        
        track = Track(
            track_id=track_id,
            bbox=tuple(bbox),
            class_id=detection.get('class_id', -1),
            confidence=detection.get('confidence', 0),
            center=center,
            trajectory=[center],
            first_seen=frame_idx,
            last_seen=frame_idx,
            age=0,
            hit_streak=1
        )
        
        self.tracks[track_id] = track
        self.trajectories[track_id] = [center]
        self.total_tracks += 1
        
        logger.debug(f"Created track {track_id}")
    
    def _update_track(self, track_id: int, detection: dict, frame_idx: int):
        """Update existing track"""
        track = self.tracks[track_id]
        bbox = detection['bbox']
        center = detection.get('center', self._get_center(bbox))
        
        # Update track info
        track.bbox = tuple(bbox)
        track.center = center
        track.confidence = detection.get('confidence', track.confidence)
        track.last_seen = frame_idx
        track.age = 0
        track.hit_streak += 1
        
        # Update trajectory
        self.trajectories[track_id].append(center)
        if len(self.trajectories[track_id]) > self.max_history:
            self.trajectories[track_id].pop(0)
        
        track.trajectory = self.trajectories[track_id]
    
    def _remove_old_tracks(self):
        """Remove tracks that haven't been updated"""
        to_remove = [
            track_id for track_id, track in self.tracks.items()
            if track.age > self.max_age
        ]
        
        for track_id in to_remove:
            del self.tracks[track_id]
            if track_id in self.trajectories:
                del self.trajectories[track_id]
    
    def get_active_tracks(self) -> List[Track]:
        """Get list of active (confirmed) tracks"""
        return [
            track for track in self.tracks.values()
            if track.hit_streak >= self.min_hits and track.age < 5
        ]
    
    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """Get trajectory for a track"""
        return self.trajectories.get(track_id, [])
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID"""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> Dict[int, Track]:
        """Get all tracks"""
        return self.tracks
    
    def get_statistics(self) -> dict:
        """Get tracking statistics"""
        return {
            'total_tracks_created': self.total_tracks,
            'active_tracks': self.active_tracks,
            'current_tracks': len(self.tracks),
            'frame_count': self.frame_count
        }
    
    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.trajectories.clear()
        self.frame_count = 0
        self.next_id = 1
        self.total_tracks = 0
        self.active_tracks = 0


class YOLOv8Tracker:
    """
    Wrapper untuk YOLOv8 built-in tracking
    Lebih akurat menggunakan ByteTrack algorithm
    """
    
    def __init__(self, config=None):
        """
        Initialize YOLOv8 tracker
        
        Args:
            config: ModelConfig instance
        """
        self.config = config
        self.model = None
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.max_history = 50
        
        self._init_model()
    
    def _init_model(self):
        """Initialize YOLOv8 model for tracking"""
        try:
            from ultralytics import YOLO
            
            model_path = "yolov8n.pt"  # Default
            if self.config and hasattr(self.config, 'VEHICLE_MODEL'):
                model_path = self.config.VEHICLE_MODEL
            
            logger.info(f"Loading YOLOv8 for tracking: {model_path}")
            self.model = YOLO(model_path)
            
            device = "cpu"
            if self.config and hasattr(self.config, 'DEVICE'):
                device = self.config.DEVICE
            
            self.model.to(device)
            logger.info(f"Tracker model loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to init tracking model: {e}")
    
    def track(self, frame: np.ndarray, 
              classes: Optional[List[int]] = None,
              conf: float = 0.5) -> List[dict]:
        """
        Track objects in frame
        
        Args:
            frame: Input image
            classes: List of class IDs to track
            conf: Confidence threshold
        
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
                conf=conf,
                classes=classes,
                verbose=False
            )[0]
            
            tracks = []
            
            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                for box, track_id, confidence, class_id in zip(
                    boxes, track_ids, confidences, class_ids
                ):
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    # Update trajectory
                    self.track_history[track_id].append(center)
                    if len(self.track_history[track_id]) > self.max_history:
                        self.track_history[track_id].pop(0)
                    
                    tracks.append({
                        'track_id': int(track_id),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': center,
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'trajectory': list(self.track_history[track_id])
                    })
            
            return tracks
            
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            return []
    
    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """Get trajectory for a track"""
        return self.track_history.get(track_id, [])
    
    def reset(self):
        """Reset track history"""
        self.track_history.clear()


# Test standalone
if __name__ == "__main__":
    print("Testing Vehicle Tracker...")
    
    # Test simple tracker
    tracker = VehicleTracker()
    
    # Simulate detections
    detections = [
        {'bbox': [100, 100, 200, 200], 'class_id': 2, 'confidence': 0.9},
        {'bbox': [300, 300, 400, 400], 'class_id': 3, 'confidence': 0.8},
    ]
    
    tracks = tracker.update(detections)
    print(f"Frame 1: {len(tracks)} tracks")
    
    # Simulate movement
    detections = [
        {'bbox': [110, 100, 210, 200], 'class_id': 2, 'confidence': 0.9},
        {'bbox': [320, 310, 420, 410], 'class_id': 3, 'confidence': 0.8},
    ]
    
    tracks = tracker.update(detections)
    print(f"Frame 2: {len(tracks)} tracks")
    
    for track in tracks:
        print(f"  Track {track.track_id}: trajectory length = {len(track.trajectory)}")
    
    print(f"Statistics: {tracker.get_statistics()}")
