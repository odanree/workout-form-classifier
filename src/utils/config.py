"""Configuration management."""

import json
import os


class Config:
    """Load and manage configuration."""
    
    DEFAULTS = {
        "detection_threshold": 1.0,
        "min_scene_length": 0.5,
        "min_form_confidence": 0.5,
        "output_formats": ["json"],
        "exercise_definitions": {
            "squat": {
                "good_form_keywords": [
                    "knees tracking over toes",
                    "chest up",
                    "neutral spine",
                    "full depth",
                    "controlled movement"
                ],
                "bad_form_keywords": [
                    "knee valgus",
                    "chest collapsed",
                    "rounded back",
                    "shallow depth",
                    "bouncing"
                ]
            },
            "deadlift": {
                "good_form_keywords": [
                    "neutral spine",
                    "shoulders over bar",
                    "chest up",
                    "smooth pull",
                    "controlled descent"
                ],
                "bad_form_keywords": [
                    "rounded back",
                    "hips too high",
                    "jerky movement",
                    "bar drifting",
                    "early hip extension"
                ]
            },
            "bench_press": {
                "good_form_keywords": [
                    "elbows at 45 degrees",
                    "full range of motion",
                    "stable shoulders",
                    "controlled descent",
                    "clean bar path"
                ],
                "bad_form_keywords": [
                    "elbows flared",
                    "partial range",
                    "shoulder instability",
                    "bouncing",
                    "bar drifting"
                ]
            }
        }
    }
    
    @classmethod
    def load(cls, config_path: str = "config/workflow_config.json"):
        """Load configuration from file, or return defaults."""
        if os.path.exists(config_path):
            with open(config_path) as f:
                return {**cls.DEFAULTS, **json.load(f)}
        return cls.DEFAULTS
