# 💪 Workout Form Classifier

ML-powered workout form analyzer using fine-tuned CLIP for real-time form feedback and form score reporting.

**Status**: MVP development 🚀

---

## 🎯 What It Does

Analyzes gym videos frame-by-frame to detect workout form quality:

- ✅ **Good Form Detection**: Automatically identifies perfect form moments
- ❌ **Bad Form Detection**: Flags form breakdowns with specific issues
- 📊 **Form Scoring**: 0-100 score per set with actionable feedback
- 🎬 **Video Analysis**: Batch process workout videos or real-time camera feed
- 📱 **Web Dashboard**: Upload videos, get instant form reports

---

## 📋 Pipeline Architecture

```
Workout Video
    ↓
[0] Scene Detection      → Detect set boundaries (PySceneDetect)
    ↓
[1] Frame Extraction     → Extract key frames from each set
    ↓
[2] CLIP Classification  → Classify frames (Good vs Bad Form)
    ↓
[3] Filtering & Scoring  → Aggregate scores, detect issues
    ↓
[4] Human Review (opt)   → Preview server for verification
    ↓
[5] Report Generation    → JSON report + recommendations
    ↓
Report Output (JSON/PDF/Web)
```

---

## 🏋️ Supported Exercises (MVP)

- [ ] Squat (good/bad form variants)
- [ ] Deadlift (good/bad form variants)
- [ ] Bench Press (good/bad form variants)
- [ ] Barbell Row (good/bad form variants)
- [ ] Overhead Press (good/bad form variants)

*(Expandable to any exercise - just add training data)*

---

## 🚀 Quick Start

### Installation

```bash
# Clone repo
git clone https://github.com/odanree/workout-form-classifier
cd workout-form-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download/train CLIP model (first run)
python scripts/setup_model.py
```

### Usage

```bash
# Analyze a workout video
python src/batch_runner.py "path/to/workout_video.mp4" --exercise squat

# With custom settings
python src/batch_runner.py "workout.mp4" \
  --exercise deadlift \
  --detection_threshold 1.0 \
  --min_scene_length 0.5

# Generate report as JSON
python src/batch_runner.py "workout.mp4" --output json

# Generate report as PDF (with charts)
python src/batch_runner.py "workout.mp4" --output pdf
```

### Output Example

```json
{
  "video": "squat_session.mp4",
  "exercise": "Squat",
  "duration": 342.5,
  "sets": [
    {
      "set_number": 1,
      "reps": 5,
      "form_score": 94,
      "frames_analyzed": 156,
      "good_form_frames": 147,
      "bad_form_frames": 9,
      "issues": [
        {
          "frame": 23,
          "timestamp": "0:15",
          "issue": "Knee valgus - knees caving inward",
          "severity": "medium"
        }
      ]
    }
  ],
  "overall_form_score": 91,
  "recommendations": [
    "Excellent knee alignment consistency - keep current form",
    "Focus on depth - ensure you reach parallel or below",
    "Smooth eccentric - avoid dropping weight quickly"
  ]
}
```

---

## 📁 Project Structure

```
workout-form-classifier/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore
├── .env.example
│
├── src/
│   ├── __init__.py
│   ├── batch_runner.py               # Main orchestrator
│   ├── 0_detect_scenes.py            # Scene detection (PySceneDetect)
│   ├── 1_extract_frames.py           # Extract frames from scenes
│   ├── 2_classify_frames.py          # CLIP classification
│   ├── 3_filter_and_score.py         # Filtering & form scoring
│   ├── 4_preview_server.py           # Web preview (http://127.0.0.1:8767)
│   ├── 5_generate_report.py          # Report generation
│   └── utils/
│       ├── config.py                 # Configuration management
│       ├── logger.py                 # Logging utilities
│       ├── video_processor.py        # FFmpeg wrappers
│       └── clip_utils.py             # CLIP model utilities
│
├── models/
│   ├── clip_finetuned_workout.pt    # Fine-tuned CLIP model (download)
│   └── exercise_configs.json         # Exercise class definitions
│
├── data/
│   ├── training_data/
│   │   ├── squat/
│   │   │   ├── good_form/            # Good form videos
│   │   │   └── bad_form/             # Bad form videos
│   │   ├── deadlift/
│   │   ├── bench_press/
│   │   └── ...
│   │
│   ├── labeled_frames/               # Frame annotations for training
│   │   ├── squat_good_form.csv
│   │   └── squat_bad_form.csv
│   │
│   └── sample_videos/                # Demo videos
│       ├── squat_demo.mp4
│       └── deadlift_demo.mp4
│
├── scripts/
│   ├── setup_model.py                # Download/initialize CLIP model
│   ├── collect_training_data.py      # Video collection helper
│   ├── label_frames.py               # Label extracted frames
│   ├── finetune_clip.py              # Train fine-tuned CLIP model
│   ├── evaluate_model.py             # Model evaluation metrics
│   └── generate_dummy_data.py        # Create test data
│
├── web/                              # React/Next.js frontend
│   ├── package.json
│   ├── pages/
│   │   ├── index.js                  # Landing page
│   │   ├── upload.js                 # Video upload page
│   │   └── report.js                 # Form report viewer
│   ├── components/
│   │   ├── VideoUploader.js
│   │   ├── FormScoreCard.js
│   │   ├── SetDetailCard.js
│   │   └── RecommendationList.js
│   └── styles/
│       └── globals.css
│
├── backend/                          # FastAPI backend
│   ├── main.py                       # API server
│   ├── routes/
│   │   ├── upload.py                 # Video upload endpoint
│   │   ├── analyze.py                # Analysis endpoint
│   │   └── report.py                 # Report retrieval
│   ├── tasks/
│   │   └── process_video.py          # Celery task for video processing
│   └── models/
│       └── schemas.py                # Pydantic schemas
│
├── tests/
│   ├── test_scene_detection.py
│   ├── test_clip_classification.py
│   ├── test_form_scoring.py
│   └── test_api.py
│
├── docs/
│   ├── SETUP.md                      # Detailed setup instructions
│   ├── TRAINING.md                   # How to train custom models
│   ├── DATA_COLLECTION.md            # Data collection guide
│   ├── ARCHITECTURE.md               # Technical deep-dive
│   └── API_REFERENCE.md              # API documentation
│
└── config/
    ├── workflow_config.json          # Pipeline configuration
    ├── exercise_definitions.json     # Exercise class descriptions
    └── form_feedback.json            # Form issue templates
```

---

## 🧠 How Form Classification Works

### Fine-Tuned CLIP Model

We train CLIP on pairs of images + descriptions:

```python
# Training data example
{
  "image": "squat_frame_001.jpg",
  "description": "perfect squat form with knees tracking over toes chest up neutral spine full depth",
  "label": "good_form"
}

{
  "image": "squat_frame_042.jpg",
  "description": "poor squat form with knees caving inward valgus chest collapsed rounded back shallow depth",
  "label": "bad_form"
}
```

CLIP then generalizes to new workout videos, scoring form quality per frame.

### Classification Pipeline

1. **Scene Detection** → Identifies set boundaries (rest periods = scene changes)
2. **Frame Extraction** → Sample frames during active lifting (skip pauses)
3. **CLIP Scoring** → Score each frame (0-100 form quality)
4. **Aggregation** → Per-set form score via majority voting
5. **Issue Detection** → Flag frames where form broke down

---

## 📊 MVP Features

- [x] Repo structure with clear separation of concerns
- [ ] Scene detection (PySceneDetect integration)
- [ ] Frame extraction pipeline
- [ ] CLIP fine-tuning on squat data
- [ ] Form classification (good vs bad)
- [ ] Form scoring algorithm
- [ ] Web preview server
- [ ] Report generation (JSON → PDF)
- [ ] Web dashboard (React)
- [ ] API backend (FastAPI)
- [ ] Celery task queue for background processing

---

## 🔧 Technologies

- **ML**: PyTorch, Hugging Face Transformers (CLIP)
- **Video**: OpenCV, FFmpeg, PySceneDetect
- **Backend**: FastAPI, Celery, Redis
- **Frontend**: React/Next.js, TypeScript
- **Deploy**: Docker, Vercel (frontend), Railway/Render (backend)

---

## 📈 Roadmap

### Phase 1 (MVP) - 2-3 weeks
- ✅ Repo structure
- [ ] Squat form classification (good/bad)
- [ ] Web upload + report generation
- [ ] Form scoring algorithm

### Phase 2 - 1-2 months
- [ ] Support 5 exercises (squat, deadlift, bench, row, OHP)
- [ ] Real-time camera feed processing
- [ ] Advanced form issue detection (granular feedback)
- [ ] User dashboard with workout history

### Phase 3 - Launch
- [ ] Mobile app (React Native)
- [ ] Gym API integrations
- [ ] Personal trainer dashboard
- [ ] SaaS pricing model

---

## 🚀 Deployment

### Local Development
```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload

# Terminal 2: Frontend
cd web
npm run dev

# Terminal 3: Celery worker (optional)
celery -A tasks worker --loglevel=info
```

### Production (Vercel + Railway)
```bash
# Deploy frontend to Vercel
vercel deploy

# Deploy backend to Railway
railway up
```

---

## 📚 Documentation

See `docs/` folder for:
- **SETUP.md** - Installation & configuration
- **TRAINING.md** - How to fine-tune CLIP on your own data
- **DATA_COLLECTION.md** - Strategies for collecting labeled video data
- **ARCHITECTURE.md** - Technical deep-dive into the pipeline
- **API_REFERENCE.md** - Full API documentation

---

## 💡 Use Cases

1. **Personal Training App** - Coach reviews form automatically
2. **Gym Memberships** - Mirror/tablet feedback during workouts
3. **Fitness Influencers** - Auto-generate form breakdown content
4. **Physical Therapy** - Monitor rehabilitation exercise form
5. **Sports Teams** - Strength & conditioning performance analysis

---

## 🤝 Contributing

Contributions welcome! Areas to help:

- [ ] Collect labeled training data for each exercise
- [ ] Build web UI components
- [ ] Optimize CLIP inference speed
- [ ] Add more exercises
- [ ] Write tests

---

## 📝 License

MIT - Feel free to use for your projects!

---

## 🎯 Next Steps

1. **Setup**: Follow [SETUP.md](docs/SETUP.md)
2. **Collect Data**: Use [DATA_COLLECTION.md](docs/DATA_COLLECTION.md) for training data strategy
3. **Train Model**: Run `scripts/finetune_clip.py`
4. **Test Pipeline**: `python src/batch_runner.py sample_videos/squat_demo.mp4`
5. **Build Web UI**: Contribute to `web/` folder
6. **Deploy**: Push to Vercel + Railway

---

**Built with ❤️ by Danh Le**  
Portfolio: [danhle.net](https://danhle.net)  
GitHub: [@odanree](https://github.com/odanree)
