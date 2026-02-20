#!/bin/bash
# ATCS Vision - Quick Start Script

echo "=============================================="
echo "🚦 ATCS JOGJA - AI Traffic Surveillance"
echo "=============================================="

cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found!"
    exit 1
fi

echo "Python: $(python3 --version)"

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg: $(ffmpeg -version | head -1)"
else
    echo "⚠️  FFmpeg not found - stream capture may not work"
fi

# Create directories
mkdir -p outputs/{recordings,violations,reports,snapshots}

# Check for ultralytics
python3 -c "import ultralytics" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "YOLOv8: Available (Full Mode)"
else
    echo "YOLOv8: Not installed (Stream-Only Mode)"
fi

echo ""
echo "=============================================="
echo "Choose mode:"
echo "  1) Gradio Web UI (default)"
echo "  2) OpenCV Display"
echo "  3) Headless (server mode)"
echo "=============================================="
echo ""

read -p "Enter choice [1-3]: " choice

case $choice in
    2)
        echo "Starting OpenCV mode..."
        python3 main.py
        ;;
    3)
        echo "Starting headless mode..."
        python3 main.py --headless
        ;;
    *)
        echo "Starting Gradio Web UI..."
        python3 app.py
        ;;
esac
