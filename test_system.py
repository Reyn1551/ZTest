#!/usr/bin/env python3
"""
Quick Test Script untuk ATCS Vision System
Test koneksi stream dan komponen dasar
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test semua imports"""
    print("=" * 50)
    print("1. Testing Imports...")
    print("=" * 50)
    
    try:
        import numpy as np
        print(f"   ✓ NumPy {np.__version__}")
        
        import cv2
        print(f"   ✓ OpenCV {cv2.__version__}")
        
        import ultralytics
        print(f"   ✓ Ultralytics {ultralytics.__version__}")
        
        import gradio as gr
        print(f"   ✓ Gradio {gr.__version__}")
        
        print("   All imports successful!\n")
        return True
    except ImportError as e:
        print(f"   ✗ Import error: {e}\n")
        return False


def test_config():
    """Test konfigurasi"""
    print("=" * 50)
    print("2. Testing Configuration...")
    print("=" * 50)
    
    from config.settings import STREAM_CONFIG, MODEL_CONFIG, SPEED_CONFIG
    
    print(f"   Stream URL: {STREAM_CONFIG.M3U8_URL[:50]}...")
    print(f"   Resolution: {STREAM_CONFIG.RESOLUTION}")
    print(f"   Target FPS: {STREAM_CONFIG.TARGET_FPS}")
    print(f"   Speed Limit: {SPEED_CONFIG.SPEED_LIMIT} km/h")
    print(f"   Model: {MODEL_CONFIG.VEHICLE_MODEL}")
    print("   Configuration loaded!\n")
    return True


def test_stream_connection():
    """Test koneksi ke stream"""
    print("=" * 50)
    print("3. Testing Stream Connection...")
    print("=" * 50)
    
    import subprocess
    from config.settings import STREAM_CONFIG
    
    url = STREAM_CONFIG.M3U8_URL
    
    # Test dengan FFprobe
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✓ Stream accessible!")
            print(f"   URL: {url[:60]}...")
            return True
        else:
            print("   ⚠ Stream may not be accessible (network/timeout)")
            print(f"   Error: {result.stderr.decode()[:100]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⚠ Connection timeout (stream might be slow)")
        return False
    except Exception as e:
        print(f"   ⚠ Test error: {e}")
        return False


def test_model_loading():
    """Test loading YOLOv8 model"""
    print("=" * 50)
    print("4. Testing Model Loading...")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        from config.settings import MODEL_CONFIG
        
        print(f"   Loading {MODEL_CONFIG.VEHICLE_MODEL}...")
        model = YOLO(MODEL_CONFIG.VEHICLE_MODEL)
        
        # Warmup inference
        import numpy as np
        dummy = np.zeros((640, 480, 3), dtype=np.uint8)
        _ = model(dummy, verbose=False)
        
        print("   ✓ Model loaded and warmed up!")
        print("   Classes: car, motorcycle, bus, truck\n")
        return True
        
    except Exception as e:
        print(f"   ✗ Model loading error: {e}\n")
        return False


def test_components():
    """Test semua komponen"""
    print("=" * 50)
    print("5. Testing Components...")
    print("=" * 50)
    
    try:
        from core.stream_capture import RobustStreamCapture
        print("   ✓ Stream Capture")
        
        from core.detector import VehicleDetector
        print("   ✓ Vehicle Detector")
        
        from core.tracker import VehicleTracker
        print("   ✓ Vehicle Tracker")
        
        from core.analyzer import TrafficAnalyzer
        print("   ✓ Traffic Analyzer")
        
        from utils.visualization import DashboardRenderer
        print("   ✓ Dashboard Renderer")
        
        from utils.csv_reporter import CSVReporter
        print("   ✓ CSV Reporter")
        
        print("   All components ready!\n")
        return True
        
    except Exception as e:
        print(f"   ✗ Component error: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 48 + "╗")
    print("║   ATCS VISION - SYSTEM TEST                   ║")
    print("╚" + "═" * 48 + "╝")
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Stream", test_stream_connection()))
    results.append(("Model", test_model_loading()))
    results.append(("Components", test_components()))
    
    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"   {name}: {status}")
    
    print()
    print(f"   Total: {passed}/{total} tests passed")
    print()
    
    if passed == total:
        print("   🎉 All tests passed! System ready.")
        print()
        print("   To run with Gradio UI:")
        print("   $ python main.py --gradio")
        print()
        print("   To run with OpenCV:")
        print("   $ python main.py")
    else:
        print("   ⚠ Some tests failed. Check errors above.")
    
    print()
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
