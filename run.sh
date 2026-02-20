#!/bin/bash
# ATCS Vision - Run Script

echo "=========================================="
echo "🚦 ATCS JOGJA - AI Traffic Surveillance"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found!"
    exit 1
fi

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found, will use OpenCV fallback"
fi

# Create output directories
mkdir -p outputs/recordings outputs/violations outputs/reports outputs/snapshots

# Run based on argument
case "$1" in
    --gradio|-g)
        echo "Starting Gradio Web UI..."
        python3 main.py --gradio "${@:2}"
        ;;
    --headless|-h)
        echo "Starting in headless mode..."
        python3 main.py --headless "${@:2}"
        ;;
    --help)
        python3 main.py --help
        ;;
    *)
        echo "Starting OpenCV mode..."
        python3 main.py "$@"
        ;;
esac
