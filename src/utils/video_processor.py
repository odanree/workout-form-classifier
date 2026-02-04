"""Video processing utilities."""

import subprocess
import json
from pathlib import Path


class VideoProcessor:
    """Handle video processing with FFmpeg."""
    
    @staticmethod
    def get_duration(video_path: str) -> float:
        """Get video duration in seconds."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
    @staticmethod
    def extract_frame(video_path: str, timestamp: float, output_path: str) -> None:
        """Extract single frame at timestamp."""
        cmd = [
            "ffmpeg", "-i", video_path,
            "-ss", str(timestamp),
            "-vframes", "1",
            "-q:v", "2",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    @staticmethod
    def get_fps(video_path: str) -> float:
        """Get video frame rate."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        parts = result.stdout.strip().split("/")
        return float(parts[0]) / float(parts[1]) if len(parts) > 1 else 30.0
