#!/usr/bin/env python3
"""
Extract training frames from workout videos and generate labeled CSV files.

Creates structured training data for fine-tuning CLIP model.

Usage:
    python scripts/extract_training_frames.py --video-dir data/training_data
    python scripts/extract_training_frames.py --video-dir data/training_data --frames-per-video 30
"""

import argparse
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
import sys

from src.utils.logger import get_logger
from src.utils.video_processor import VideoProcessor

logger = get_logger(__name__)


class TrainingDataExtractor:
    """Extract frames from workout videos for training."""
    
    def __init__(self, output_dir: str = "data/labeled_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_processor = VideoProcessor()
        self.extracted_data = []
    
    def process_video_directory(
        self,
        video_dir: str,
        frames_per_video: int = 20,
        skip_first_sec: float = 1.0,
        skip_last_sec: float = 1.0
    ) -> Dict[str, int]:
        """
        Process entire video directory structure.
        
        Expected structure:
        video_dir/
        ├── squat/
        │   ├── good_form/
        │   │   ├── squat_good_001.mp4
        │   │   └── ...
        │   └── bad_form/
        │       ├── squat_bad_001.mp4
        │       └── ...
        ├── deadlift/
        │   ├── good_form/
        │   └── bad_form/
        └── ...
        
        Args:
            video_dir: Root video directory
            frames_per_video: How many frames to extract per video
            skip_first_sec: Skip initial seconds (setup/breathing)
            skip_last_sec: Skip final seconds (racking)
        
        Returns:
            Summary dict with counts
        """
        video_dir = Path(video_dir)
        summary = {}
        
        # Iterate exercises (squat, deadlift, etc.)
        for exercise_dir in video_dir.iterdir():
            if not exercise_dir.is_dir():
                continue
            
            exercise = exercise_dir.name
            logger.info(f"\nProcessing exercise: {exercise}")
            
            exercise_frames = {}
            
            # Iterate good_form and bad_form
            for form_type in ["good_form", "bad_form"]:
                form_dir = exercise_dir / form_type
                if not form_dir.exists():
                    logger.warning(f"  Skipping {form_type} (not found)")
                    continue
                
                logger.info(f"  {form_type}:")
                
                video_files = list(form_dir.glob("*.mp4")) + list(form_dir.glob("*.mov"))
                logger.info(f"    Found {len(video_files)} videos")
                
                form_frames = []
                
                for video_path in video_files:
                    try:
                        frames = self._extract_frames_from_video(
                            str(video_path),
                            exercise,
                            form_type,
                            frames_per_video,
                            skip_first_sec,
                            skip_last_sec
                        )
                        form_frames.extend(frames)
                        logger.info(f"    ✓ {video_path.name} - {len(frames)} frames")
                    except Exception as e:
                        logger.error(f"    ✗ {video_path.name} - {str(e)}")
                
                exercise_frames[form_type] = form_frames
            
            # Save CSV for this exercise
            self._save_exercise_csv(exercise, exercise_frames)
            
            total_frames = sum(len(frames) for frames in exercise_frames.values())
            summary[exercise] = total_frames
            logger.info(f"  Total frames: {total_frames}")
        
        # Save metadata
        self._save_metadata(summary)
        
        return summary
    
    def _extract_frames_from_video(
        self,
        video_path: str,
        exercise: str,
        form_type: str,
        frames_per_video: int,
        skip_first_sec: float,
        skip_last_sec: float
    ) -> List[Dict]:
        """Extract specific number of frames from a video."""
        duration = self.video_processor.get_duration(video_path)
        fps = self.video_processor.get_fps(video_path)
        
        # Calculate skip frames
        skip_first_frames = int(skip_first_sec * fps)
        skip_last_frames = int(skip_last_sec * fps)
        total_frames = int(duration * fps)
        
        usable_frames = total_frames - skip_first_frames - skip_last_frames
        if usable_frames <= 0:
            raise ValueError(f"Video too short or skip values too large")
        
        # Calculate frame indices to extract
        frame_indices = [
            skip_first_frames + int(i * usable_frames / frames_per_video)
            for i in range(frames_per_video)
        ]
        
        # Extract frames
        frames_data = []
        frame_output_dir = self.output_dir / f"frames_{exercise}_{form_type}"
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, frame_num in enumerate(frame_indices):
            timestamp = frame_num / fps
            frame_path = frame_output_dir / f"{Path(video_path).stem}_{idx:03d}.jpg"
            
            try:
                self.video_processor.extract_frame(video_path, timestamp, str(frame_path))
                
                frames_data.append({
                    "frame_path": str(frame_path.relative_to(self.output_dir)),
                    "exercise": exercise,
                    "form_type": form_type,
                    "source_video": Path(video_path).name,
                    "timestamp": timestamp,
                    "description": ""  # To be filled by human annotation
                })
            except Exception as e:
                logger.warning(f"Failed to extract frame {frame_num}: {str(e)}")
        
        return frames_data
    
    def _save_exercise_csv(self, exercise: str, exercise_frames: Dict[str, List]) -> None:
        """Save training data for one exercise as CSV."""
        csv_path = self.output_dir / f"{exercise}_training_data.csv"
        
        all_frames = []
        for form_type, frames in exercise_frames.items():
            all_frames.extend(frames)
        
        if not all_frames:
            logger.warning(f"  No frames to save for {exercise}")
            return
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame_path", "exercise", "form_type", "source_video",
                "timestamp", "description", "confidence", "annotator_notes"
            ])
            writer.writeheader()
            
            for frame in all_frames:
                frame["confidence"] = ""  # To be filled during training
                frame["annotator_notes"] = ""
                writer.writerow(frame)
        
        logger.info(f"  Saved: {csv_path}")
    
    def _save_metadata(self, summary: Dict[str, int]) -> None:
        """Save extraction metadata."""
        metadata = {
            "total_frames": sum(summary.values()),
            "exercises": summary,
            "output_dir": str(self.output_dir),
            "frames_per_video": 20,  # Would need to pass this through
            "timestamp": str(Path(self.output_dir).name)
        }
        
        metadata_path = self.output_dir / "_extraction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\nMetadata saved: {metadata_path}")
        logger.info(f"Total frames extracted: {metadata['total_frames']}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract training frames from workout videos"
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing videos (exercise/good_form or bad_form structure)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/labeled_frames",
        help="Output directory for extracted frames and CSVs"
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=20,
        help="Number of frames to extract per video"
    )
    parser.add_argument(
        "--skip-first",
        type=float,
        default=1.0,
        help="Skip first N seconds of video (setup)"
    )
    parser.add_argument(
        "--skip-last",
        type=float,
        default=1.0,
        help="Skip last N seconds of video (racking)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_dir):
        logger.error(f"Video directory not found: {args.video_dir}")
        sys.exit(1)
    
    extractor = TrainingDataExtractor(args.output_dir)
    summary = extractor.process_video_directory(
        args.video_dir,
        frames_per_video=args.frames_per_video,
        skip_first_sec=args.skip_first,
        skip_last_sec=args.skip_last
    )
    
    print("\n" + "="*60)
    print("Extraction Summary:")
    print("="*60)
    for exercise, count in summary.items():
        print(f"  {exercise}: {count} frames")
    print(f"  TOTAL: {sum(summary.values())} frames")
    print("="*60)
    print(f"\nNext step: Annotate descriptions in CSV files")
    print(f"Files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
