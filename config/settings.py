"""
Konfigurasi sistem ATCS Vision
Speed + Direction + Helmet Detection dengan Robust M3U8 Streaming
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class StreamConfig:
    """Konfigurasi untuk M3U8 streaming dari ATCS Jogja"""
    
    # ATCS Jogja M3U8 URL
    M3U8_URL: str = "https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Wirosaban_View_Barat.stream/chunklist_w821725872.m3u8"
    
    # Buffer settings (penting untuk anti-freeze)
    PRE_BUFFER_SECONDS: int = 5          # Buffer 5 detik sebelum play
    MAX_BUFFER_SIZE: int = 150           # Frame buffer (5 detik @ 30fps)
    FRAME_TIMEOUT: float = 10.0          # Timeout baca frame (detik)
    
    # Stream quality
    TARGET_FPS: int = 15                 # Target FPS processing
    RESOLUTION: Tuple[int, int] = (1280, 720)  # Downscale untuk performa
    
    # Reconnect settings
    RECONNECT_ATTEMPTS: int = 5
    RECONNECT_DELAY: float = 3.0
    
    # FFmpeg specific
    FFmpeg_BUFFER_SIZE: str = "50000000"  # 50MB buffer


@dataclass
class ModelConfig:
    """Konfigurasi untuk model AI"""
    
    # YOLOv8 models
    VEHICLE_MODEL: str = "yolov8n.pt"    # Deteksi kendaraan (COCO)
    HELMET_MODEL: str = None              # Custom model helm (nanti training)
    
    # Inference settings
    CONFIDENCE: float = 0.5
    IOU: float = 0.45
    DEVICE: str = "cpu"                   # "0" untuk GPU, "cpu" untuk CPU
    
    # Classes yang dipakai (COCO dataset)
    # 2: car, 3: motorcycle, 5: bus, 7: truck
    VEHICLE_CLASSES: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    
    # Class names mapping
    CLASS_NAMES: dict = field(default_factory=lambda: {
        2: 'car', 
        3: 'motorcycle', 
        5: 'bus', 
        7: 'truck'
    })


@dataclass
class SpeedConfig:
    """Konfigurasi untuk estimasi kecepatan"""
    
    # Kalibrasi (sesuaikan dengan lokasi CCTV)
    PIXEL_TO_METER: float = 0.05         # 1 pixel = 0.05 meter (contoh)
    FPS_STREAM: int = 30                 # FPS asli stream
    
    # Speed calculation
    SPEED_SMOOTHING: float = 0.7         # EMA smoothing factor
    MIN_SPEED: float = 0.0               # km/jam
    MAX_SPEED: float = 120.0             # km/jam (filter outlier)
    
    # Speed limit for violation
    SPEED_LIMIT: float = 60.0            # km/jam
    
    # History length for smoothing
    SPEED_HISTORY_LENGTH: int = 10
    
    # Reference points (sesuaikan dengan marka jalan)
    CALIBRATION_POINTS: List[Tuple[int, int]] = field(
        default_factory=lambda: [(100, 500), (500, 500)]
    )


@dataclass
class DirectionConfig:
    """Konfigurasi untuk deteksi arah"""
    
    # Thresholds
    MIN_TRAJECTORY_LENGTH: int = 5       # Minimal points untuk direction
    MOVEMENT_THRESHOLD: int = 20         # Minimal pixel movement
    HORIZONTAL_THRESHOLD: int = 50       # Threshold untuk LEFT/RIGHT
    
    # Direction labels
    DIRECTIONS: dict = field(default_factory=lambda: {
        'UP': 'Menuju kamera',
        'DOWN': 'Menjauhi kamera',
        'LEFT': 'Bergerak ke kiri',
        'RIGHT': 'Bergerak ke kanan',
        'STOPPED': 'Berhenti',
        'STRAIGHT': 'Lurus'
    })


@dataclass
class HelmetConfig:
    """Konfigurasi untuk deteksi helm"""
    
    # Only check motorcycle class
    MOTORCYCLE_CLASS_ID: int = 3
    
    # Helmet region (relative to motorcycle bbox)
    HELMET_REGION_RATIO: float = 0.4     # Upper 40% of bbox
    
    # Confidence threshold
    HELMET_CONFIDENCE: float = 0.6


@dataclass
class OutputConfig:
    """Konfigurasi output dan recording"""
    
    # Video recording
    SAVE_VIDEO: bool = True
    VIDEO_CODEC: str = "mp4v"
    VIDEO_FPS: int = 15
    
    # Violation screenshots
    SAVE_VIOLATIONS: bool = True
    VIOLATION_COOLDOWN: int = 5          # Detik antar screenshot pelanggaran
    
    # CSV Reports
    GENERATE_CSV_REPORT: bool = True
    REPORT_INTERVAL: int = 300           # Generate report tiap 5 menit
    
    # Output directories
    OUTPUT_DIR: str = "outputs"
    RECORDINGS_DIR: str = "outputs/recordings"
    VIOLATIONS_DIR: str = "outputs/violations"
    REPORTS_DIR: str = "outputs/reports"
    SNAPSHOTS_DIR: str = "outputs/snapshots"


@dataclass
class GradioConfig:
    """Konfigurasi untuk Gradio UI"""
    
    # Server settings
    SERVER_NAME: str = "0.0.0.0"
    SERVER_PORT: int = 7860
    SHARE: bool = False
    
    # UI settings
    TITLE: str = "ATCS Jogja - AI Traffic Surveillance"
    DESCRIPTION: str = """
    ### 🚦 Sistem Pemantauan Lalu Lintas Cerdas
    **Speed Detection | Direction Analysis | Helmet Compliance**
    
    Menggunakan CCTV ATCS Jogja dengan AI Computer Vision untuk:
    - 📊 Estimasi kecepatan kendaraan
    - 🧭 Deteksi arah pergerakan
    - 🪖 Pemeriksaan pemakaian helm (motor)
    """
    
    # Theme
    THEME: str = "default"
    
    # Update intervals
    STATS_UPDATE_INTERVAL: float = 1.0   # detik
    CHART_UPDATE_INTERVAL: float = 5.0   # detik


# Singleton instances
STREAM_CONFIG = StreamConfig()
MODEL_CONFIG = ModelConfig()
SPEED_CONFIG = SpeedConfig()
DIRECTION_CONFIG = DirectionConfig()
HELMET_CONFIG = HelmetConfig()
OUTPUT_CONFIG = OutputConfig()
GRADIO_CONFIG = GradioConfig()
