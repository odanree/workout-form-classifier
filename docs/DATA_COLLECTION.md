# Data Collection Strategy

## Overview

Fine-tuned CLIP needs **labeled video examples** of good vs bad form for each exercise. This guide walks through efficient data collection.

---

## Phase 1: Initial Dataset (2-3 days)

### Step 1: Source Videos

**Best Sources:**
1. **YouTube** - Search for form tutorials + common mistakes
2. **Your Own Videos** - Film gym sessions (with permission)
3. **Stock Footage** - Pexels, Pixabay for workout videos
4. **Reddit** - r/fitness form check videos
5. **TikTok** - Fitness creators showing form

**Video Requirements:**
- **Duration:** 5-30 seconds per clip (one set/exercise)
- **Quality:** 720p+ (higher is better for detail)
- **Angle:** Multiple angles (front, side) best
- **Lighting:** Good lighting to see form clearly
- **Audio:** Irrelevant (we extract frames only)

### Step 2: Organize Videos

Create folder structure:

```
data/training_data/
├── squat/
│   ├── good_form/
│   │   ├── squat_good_001.mp4
│   │   ├── squat_good_002.mp4
│   │   └── ...
│   └── bad_form/
│       ├── squat_bad_001.mp4
│       ├── squat_bad_002.mp4
│       └── ...
├── deadlift/
│   ├── good_form/
│   └── bad_form/
└── bench_press/
    ├── good_form/
    └── bad_form/
```

**Target: 20-30 videos per category** (60+ total to start)

### Step 3: Extract Frames

```bash
python scripts/extract_training_frames.py --video-dir data/training_data
```

This creates `data/labeled_frames/` with:

```
labeled_frames/
├── squat_good_form.csv      # frames + labels
├── squat_bad_form.csv
├── frames_squat_good/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── frames_squat_bad/
    ├── frame_0001.jpg
    └── ...
```

---

## Phase 2: Annotation (1 day)

### Step 1: Review & Label Frames

```bash
# Launch frame review tool (requires web UI)
python scripts/label_frames_ui.py data/labeled_frames
```

**For Each Frame:**
- ✅ Confirm it's correctly labeled (good/bad form)
- ❌ Delete blurry or unclear frames
- 💬 Add specific form notes if relevant

### Step 2: Create Form Descriptions

For each frame, add a detailed text description:

**Good Form Examples:**
```
"perfect squat form with knees tracking over toes chest elevated neutral spine full depth below parallel smooth controlled movement"

"textbook deadlift form with neutral spine shoulders over bar chest elevated controlled hip extension smooth pull"
```

**Bad Form Examples:**
```
"poor squat form with knees caving inward chest collapsed rounded lower back shallow depth above parallel jerky movement"

"bad deadlift with rounded lower back hips too high shoulders behind bar jerky pulling motion"
```

**Template:**
```
[exercise] [good/bad] form with [specific observations] [key issues]
```

Save in CSV format:

```
frame_path,label,description
frames_squat_good/frame_0001.jpg,good_form,"perfect squat form with knees tracking over toes chest elevated neutral spine full depth"
frames_squat_bad/frame_0042.jpg,bad_form,"poor squat form with knees caving inward chest collapsed rounded lower back shallow depth"
```

---

## Phase 3: Training (1 day)

### Run Fine-Tuning

```bash
python scripts/finetune_clip.py \
  --data-dir data/labeled_frames \
  --output models/clip_finetuned_workout.pt \
  --epochs 5 \
  --batch-size 16
```

### Evaluate Model

```bash
python scripts/evaluate_model.py --model models/clip_finetuned_workout.pt
```

Expected accuracy: **80-95%** if data quality is good

---

## Phase 4: Iteration (Ongoing)

### Expand Datasets

After MVP works:
- Add more exercises (row, OHP, etc.)
- Collect edge cases (partial reps, recovery positions)
- Record videos from different gym angles/lighting

### Improve Model

- Re-train periodically with new data
- Fix false positives (frames incorrectly labeled)
- Add "unclear" category for borderline cases

---

## Data Collection Tips

### ✅ DO:
- Collect multiple angles (front, side, rear)
- Include different body types/heights
- Film in different lighting conditions
- Get 2-3 clean "good form" examples per movement
- Get 2-3 "bad form" examples showing common mistakes

### ❌ DON'T:
- Use only one camera angle (generalization suffers)
- Mix exercises in same video
- Include warm-up or cool-down footage
- Have excessive motion blur
- Use extreme slow-mo or fast-mo (unrealistic)

---

## YouTube Sources (Curated List)

### Good Form Tutorials:
- **Athlean-X** - Form-focused training
- **Jeff Nippard** - Science-backed form cues
- **Stronger by Science** - Detailed form breakdowns
- **Calgary Barbell** - Powerlifting technique

### Common Mistakes:
- **Reddit Form Check** videos
- **TikTok Fitness Fails**
- **Alan Thrall's Form Series** - Common errors explained

---

## Dataset Size Guidelines

| Training Data | Expected Accuracy | Notes |
|---|---|---|
| 10 videos per category | ~70% | Overfitting risk |
| 30 videos per category | ~85% | Good balance |
| 100+ videos per category | ~92% | Excellent (requires effort) |

**MVP Recommendation:** Start with 30-50 videos per category (good/bad), expand to 100+ after initial launch.

---

## Automating Collection

### YouTube Downloader

```bash
# Install
pip install yt-dlp

# Download video
yt-dlp "https://www.youtube.com/watch?v=..." -o "data/training_data/squat/good_form/%(title)s.mp4"
```

### Batch Frame Extraction

```bash
python scripts/extract_training_frames.py \
  --video-dir data/training_data \
  --frames-per-video 10
```

---

## File Organization Template

Create this structure now (before collecting):

```
data/
├── training_data/
│   ├── squat/
│   │   ├── good_form/
│   │   │   ├── .gitkeep
│   │   │   └── README.md (with source links)
│   │   └── bad_form/
│   │       ├── .gitkeep
│   │       └── README.md
│   ├── deadlift/
│   ├── bench_press/
│   ├── barbell_row/
│   └── overhead_press/
│
├── labeled_frames/
│   ├── squat_good_form.csv
│   ├── squat_bad_form.csv
│   ├── frames_squat_good/ (frames extracted)
│   └── frames_squat_bad/
│
└── sample_videos/
    ├── squat_demo.mp4 (for testing)
    └── deadlift_demo.mp4
```

---

## Next Steps

1. **Start collecting** - Aim for 30-50 videos per exercise by end of week
2. **Extract frames** - Run scripts to create labeled frame sets
3. **Annotate** - Add detailed descriptions to frames
4. **Train** - Fine-tune CLIP model
5. **Evaluate** - Test on new unseen workout videos
6. **Iterate** - Add more data for weak areas

---

## Questions?

- Check `docs/TRAINING.md` for fine-tuning details
- See `scripts/extract_training_frames.py` for frame extraction code
- Email: dtle82@gmail.com

Happy collecting! 💪

---

Built with ❤️ by Danh Le
