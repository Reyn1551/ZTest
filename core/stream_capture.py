"""
Robust M3U8 Stream Capture dengan FFmpeg
Anti-freeze dengan pre-buffer dan frame management untuk ATCS Jogja
"""

import subprocess
import threading
import queue
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StreamStats:
    """Statistics untuk stream"""
    frame_count: int = 0
    dropped_frames: int = 0
    buffer_size: int = 0
    reconnects: int = 0
    fps_effective: float = 0.0
    is_connected: bool = False
    last_frame_time: float = 0.0
    start_time: float = 0.0
    bytes_received: int = 0


class RobustStreamCapture:
    """
    Capture M3U8 stream dengan:
    - FFmpeg sebagai backend (lebih stabil dari OpenCV)
    - Pre-buffer untuk menghindari freeze
    - Auto-reconnect saat putus
    - Frame dropping jika processing lambat
    
    Features:
    - Thread-safe frame buffer
    - Adaptive buffer management
    - Real-time statistics
    - Graceful error handling
    """
    
    def __init__(self, config):
        """
        Initialize stream capture
        
        Args:
            config: StreamConfig instance
        """
        self.url = config.M3U8_URL
        self.resolution = config.RESOLUTION
        self.target_fps = config.TARGET_FPS
        self.max_buffer = config.MAX_BUFFER_SIZE
        self.frame_timeout = config.FRAME_TIMEOUT
        self.reconnect_attempts = config.RECONNECT_ATTEMPTS
        self.reconnect_delay = config.RECONNECT_DELAY
        
        # Frame buffer (thread-safe)
        self.frame_buffer: queue.Queue = queue.Queue(maxsize=self.max_buffer)
        
        # Process management
        self.process: Optional[subprocess.Popen] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.is_running: bool = False
        
        # Statistics
        self.stats = StreamStats()
        self._lock = threading.Lock()
        
        # Frame timing
        self._frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.033
        
    def _build_ffmpeg_command(self) -> list:
        """
        Build command FFmpeg untuk HLS streaming optimal
        
        Returns:
            List of command arguments for FFmpeg
        """
        width, height = self.resolution
        
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'warning',
            
            # Input options for HLS/M3U8
            '-fflags', 'nobuffer+fastseek+genpts',
            '-flags', 'low_delay',
            '-strict', 'experimental',
            
            # Network options
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '10',
            '-timeout', '30000000',  # 30 seconds timeout in microseconds
            
            # Input
            '-i', self.url,
            
            # Video processing
            '-vf', f'fps={self.target_fps},scale={width}:{height}:flags=fast_bilinear',
            '-pix_fmt', 'bgr24',
            
            # Output format
            '-f', 'rawvideo',
            '-an',  # No audio
            
            # Low latency
            '-flush_packets', '1',
            
            'pipe:1'
        ]
        
        return cmd
    
    def _capture_frames(self):
        """
        Thread function: Baca frame dari FFmpeg ke buffer
        Menghandle frame dropping dan buffer management
        """
        frame_size = self.resolution[0] * self.resolution[1] * 3
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        logger.info(f"Frame capture thread started. Frame size: {frame_size} bytes")
        
        while self.is_running:
            try:
                # Check if process is alive
                if self.process is None or self.process.poll() is not None:
                    logger.warning("FFmpeg process dead, attempting reconnect...")
                    if not self._reconnect():
                        time.sleep(1)
                        continue
                    consecutive_errors = 0
                
                # Read raw frame from FFmpeg stdout
                raw_frame = self.process.stdout.read(frame_size)
                
                # Validate frame size
                if len(raw_frame) != frame_size:
                    if len(raw_frame) == 0:
                        logger.debug("Empty frame received, stream may be ending")
                        consecutive_errors += 1
                        if consecutive_errors > max_consecutive_errors:
                            logger.error("Too many empty frames, reconnecting...")
                            self._reconnect()
                            consecutive_errors = 0
                        continue
                    else:
                        logger.debug(f"Incomplete frame: {len(raw_frame)}/{frame_size} bytes")
                        consecutive_errors += 1
                        continue
                
                # Reset error counter on successful read
                consecutive_errors = 0
                
                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.resolution[1], self.resolution[0], 3)).copy()
                
                # Add to buffer with overflow handling
                try:
                    self.frame_buffer.put_nowait(frame)
                    
                    with self._lock:
                        self.stats.frame_count += 1
                        self.stats.last_frame_time = time.time()
                        self.stats.bytes_received += frame_size
                        
                except queue.Full:
                    # Buffer full - drop oldest frame, add newest
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait(frame)
                        
                        with self._lock:
                            self.stats.dropped_frames += 1
                            
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Capture error: {type(e).__name__}: {e}")
                consecutive_errors += 1
                
                if consecutive_errors > max_consecutive_errors:
                    logger.error("Too many errors, reconnecting...")
                    self._reconnect()
                    consecutive_errors = 0
                    
                time.sleep(0.1)
        
        logger.info("Frame capture thread stopped")
    
    def _reconnect(self) -> bool:
        """
        Reconnect ke stream dengan retry logic
        
        Returns:
            bool: True if reconnection successful
        """
        with self._lock:
            self.stats.reconnects += 1
            attempt = self.stats.reconnects
        
        logger.info(f"Reconnect attempt {attempt}")
        
        # Kill old process
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.debug(f"Error killing process: {e}")
        
        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        
        # Wait before reconnecting
        time.sleep(self.reconnect_delay)
        
        # Start new process
        return self._start_ffmpeg()
    
    def _start_ffmpeg(self) -> bool:
        """
        Start FFmpeg process
        
        Returns:
            bool: True if started successfully
        """
        try:
            cmd = self._build_ffmpeg_command()
            logger.info(f"Starting FFmpeg for: {self.url}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8  # 100MB buffer
            )
            
            # Wait for process to initialize
            time.sleep(2)
            
            # Check if process started successfully
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                logger.error(f"FFmpeg failed to start: {stderr_output}")
                return False
            
            with self._lock:
                self.stats.is_connected = True
                
            logger.info("FFmpeg started successfully")
            return True
            
        except FileNotFoundError:
            logger.error("FFmpeg not found! Please install FFmpeg and add to PATH")
            return False
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start capture dengan pre-buffer
        
        Returns:
            bool: True if stream started successfully
        """
        logger.info("=" * 50)
        logger.info("Starting ATCS Jogja Stream Capture")
        logger.info(f"URL: {self.url}")
        logger.info(f"Resolution: {self.resolution}")
        logger.info(f"Target FPS: {self.target_fps}")
        logger.info("=" * 50)
        
        self.is_running = True
        self.stats.start_time = time.time()
        
        # Start FFmpeg
        if not self._start_ffmpeg():
            logger.error("Failed to start FFmpeg")
            return False
        
        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_frames,
            name="FrameCaptureThread",
            daemon=True
        )
        self.capture_thread.start()
        
        # Pre-buffer: wait for buffer to fill
        pre_buffer_target = max(10, self.max_buffer // 3)
        logger.info(f"Pre-buffering {pre_buffer_target} frames...")
        
        start_wait = time.time()
        max_wait = 30  # Maximum 30 seconds wait
        
        while self.frame_buffer.qsize() < pre_buffer_target:
            if time.time() - start_wait > max_wait:
                logger.warning("Pre-buffer timeout, starting with available frames")
                break
            
            if not self.is_running:
                return False
                
            time.sleep(0.1)
        
        buffer_size = self.frame_buffer.qsize()
        logger.info(f"Stream ready! Buffer: {buffer_size}/{self.max_buffer}")
        
        return buffer_size > 0 or self.stats.is_connected
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Baca frame dari buffer (blocking dengan timeout)
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        try:
            frame = self.frame_buffer.get(timeout=self.frame_timeout)
            return True, frame
        except queue.Empty:
            logger.warning("Frame timeout - buffer empty")
            return False, None
        except Exception as e:
            logger.error(f"Read error: {e}")
            return False, None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get stream statistics
        
        Returns:
            Dict with current statistics
        """
        with self._lock:
            elapsed = time.time() - self.stats.start_time if self.stats.start_time > 0 else 1
            fps = self.stats.frame_count / elapsed if elapsed > 0 else 0
            
            return {
                'frame_count': self.stats.frame_count,
                'dropped_frames': self.stats.dropped_frames,
                'buffer_size': self.frame_buffer.qsize(),
                'buffer_max': self.max_buffer,
                'reconnects': self.stats.reconnects,
                'fps_effective': fps,
                'is_connected': self.stats.is_connected,
                'bytes_received': self.stats.bytes_received,
                'elapsed_time': elapsed,
                'url': self.url
            }
    
    def is_alive(self) -> bool:
        """Check if stream is alive and has frames"""
        return self.is_running and (
            self.frame_buffer.qsize() > 0 or 
            (self.process and self.process.poll() is None)
        )
    
    def stop(self):
        """Stop capture gracefully"""
        logger.info("Stopping stream capture...")
        
        self.is_running = False
        
        # Wait for capture thread
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3)
        
        # Terminate FFmpeg
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg didn't terminate, killing...")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.debug(f"Error stopping FFmpeg: {e}")
        
        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        
        with self._lock:
            self.stats.is_connected = False
        
        stats = self.get_stats()
        logger.info(f"Stream stopped. Total frames: {stats['frame_count']}, "
                   f"Dropped: {stats['dropped_frames']}, Reconnects: {stats['reconnects']}")


class OpenCVFallbackCapture:
    """
    Fallback capture menggunakan OpenCV jika FFmpeg tidak tersedia
    Kurang stabil untuk M3U8 tapi bisa digunakan sebagai backup
    """
    
    def __init__(self, config):
        self.url = config.M3U8_URL
        self.resolution = config.RESOLUTION
        self.target_fps = config.TARGET_FPS
        self.cap = None
        self.is_running = False
        
    def start(self) -> bool:
        """Start OpenCV capture"""
        logger.info(f"Starting OpenCV fallback for: {self.url}")
        
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if not self.cap.isOpened():
            logger.error("OpenCV failed to open stream")
            return False
        
        self.is_running = True
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame"""
        if not self.cap or not self.is_running:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            # Resize if needed
            if frame.shape[:2][::-1] != self.resolution:
                frame = cv2.resize(frame, self.resolution)
            return True, frame
        
        return False, None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'is_connected': self.is_running,
            'url': self.url,
            'backend': 'OpenCV'
        }
    
    def stop(self):
        """Stop capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()


# Factory function
def create_capture(config, prefer_ffmpeg: bool = True):
    """
    Create appropriate capture instance
    
    Args:
        config: StreamConfig
        prefer_ffmpeg: If True, try FFmpeg first, fall back to OpenCV
    
    Returns:
        Capture instance
    """
    if prefer_ffmpeg:
        # Check if FFmpeg is available
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("FFmpeg detected, using RobustStreamCapture")
                return RobustStreamCapture(config)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("FFmpeg not found, falling back to OpenCV")
        except Exception as e:
            logger.warning(f"FFmpeg check failed: {e}, falling back to OpenCV")
    
    return OpenCVFallbackCapture(config)


# Test standalone
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import STREAM_CONFIG
    
    print("Testing ATCS Stream Capture...")
    print("Press 'q' to quit")
    
    cap = create_capture(STREAM_CONFIG)
    
    if cap.start():
        print("Stream started successfully!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Show stats
            stats = cap.get_stats()
            info_text = f"Frames: {stats.get('frame_count', 0)} | Buffer: {stats.get('buffer_size', 0)}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("ATCS Stream Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.stop()
        cv2.destroyAllWindows()
    else:
        print("Failed to start stream!")
