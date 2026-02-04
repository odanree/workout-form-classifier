# CLIP Fine-Tuning Guide

## Overview

This guide walks through fine-tuning a pre-trained CLIP model to classify workout form (good vs bad) for multiple exercises.

**Why CLIP?**
- Pre-trained on 400M image-text pairs → Understands visual concepts
- Transfer learning → Works well with limited labeled data (50-100 videos per exercise)
- Multimodal → Learns from both images AND descriptive text
- Fast inference → Can classify frames in real-time

---

## Phase 1: Prepare Training Data

### Step 1: Collect Videos

Gather ~30-50 workout videos per exercise category:

```
data/training_data/
├── squat/
│   ├── good_form/
│   │   ├── squat_good_athleanx_01.mp4    ← Youtube tutorial
│   │   ├── squat_good_myGym_01.mp4       ← Your own video
│   │   ├── squat_good_reddit_01.mp4      ← Reddit form check
│   │   └── ... (20-30 more)
│   └── bad_form/
│       ├── squat_bad_commonMistakes.mp4
│       └── ... (20-30 more)
├── deadlift/
│   ├── good_form/
│   └── bad_form/
└── ... (bench_press, row, ohp)
```

**Video sources:**
- YouTube: Athlean-X, Jeff Nippard, Stronger by Science
- Reddit: r/fitness form checks
- Your gym: Record with permission
- Stock footage: Pexels, Pixabay

### Step 2: Extract Frames

Convert videos → labeled frames + CSV:

```bash
python scripts/extract_training_frames.py \
  --video-dir data/training_data \
  --frames-per-video 25 \
  --skip-first 1.0 \
  --skip-last 1.0
```

**Output:**
```
data/labeled_frames/
├── squat_training_data.csv
├── deadlift_training_data.csv
├── frames_squat_good/
│   ├── squat_good_athleanx_01_000.jpg
│   ├── squat_good_athleanx_01_001.jpg
│   └── ...
├── frames_squat_bad/
└── ...
```

**CSV Format:**
```
frame_path,exercise,form_type,source_video,timestamp,description,confidence,annotator_notes
frames_squat_good/squat_good_01_000.jpg,squat,good_form,squat_good_01.mp4,0.5,"",""
frames_squat_good/squat_good_01_001.jpg,squat,good_form,squat_good_01.mp4,1.0,"",""
```

### Step 3: Annotate Frames

Add text descriptions to each frame. This is **CRITICAL** for CLIP fine-tuning.

#### Option A: Manual Annotation (Best Quality)

1. Open `squat_training_data.csv` in spreadsheet editor
2. For each row, watch frame and fill in `description`:

**Good Form Examples:**
```
"perfect squat form with knees tracking over toes throughout movement chest elevated shoulders retracted neutral spine full depth below parallel smooth controlled descent and ascent no bouncing"

"good squat with knees mostly aligned chest up neutral spine near-full depth controlled movement minor inconsistencies acceptable"

"squat with knee valgus knees caving inward during ascent loss of knee alignment form breakdown"
```

**Template:**
```
[exercise] with [key observations] [issues if any]
```

See `config/exercise_definitions.json` for example descriptions per class.

#### Option B: Semi-Automated (Faster)

Use AI to generate initial descriptions (then manually review):

```bash
# Requires Claude API key
python scripts/generate_descriptions_ai.py --csv-file squat_training_data.csv
```

Then manually review and refine.

### Step 4: Verify Data Quality

Check for obvious issues:

```bash
python scripts/validate_training_data.py --data-dir data/labeled_frames
```

This will report:
- Missing descriptions
- Duplicate frames
- Invalid image files
- Class imbalance

---

## Phase 2: Fine-Tune CLIP

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt

# Additional (if not already installed)
pip install tqdm pillow
```

### Step 2: Run Fine-Tuning

```bash
python scripts/finetune_clip.py \
  --data-dir data/labeled_frames \
  --output models/clip_finetuned_workout.pt \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 1e-5
```

**Parameters:**
- `--epochs`: Number of training passes (5-10 typical, more = better but slower)
- `--batch-size`: Images per batch (16-32 typical, higher = faster but more GPU memory)
- `--learning-rate`: How fast model learns (1e-5 for fine-tuning, don't change much)
- `--model`: Base CLIP model (`openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`, etc.)

**Training Tips:**
- Start with `epochs=5, batch_size=16` to test quickly
- If overfitting: increase epochs more, add description variety
- If underfitting: use larger batch_size, more epochs, bigger model

### Step 3: Monitor Training

Output shows per-epoch metrics:

```
Epoch 1/5 - Loss: 0.2341
  Val Loss: 0.1923 - Accuracy: 0.85
Epoch 2/5 - Loss: 0.1812
  Val Loss: 0.1654 - Accuracy: 0.89
...
Model saved: models/clip_finetuned_workout.pt
```

**Good signs:**
- Loss decreases steadily
- Validation accuracy increases
- Final val accuracy > 80%

**Warning signs:**
- Loss oscillating or increasing → learning rate too high
- Accuracy plateaus early → need more training data
- Massive overfitting (train loss << val loss) → need more diverse data

---

## Phase 3: Evaluate & Test

### Step 1: Evaluate on Test Set

```bash
python scripts/evaluate_model.py \
  --model models/clip_finetuned_workout.pt \
  --data-dir data/labeled_frames
```

Outputs per-exercise metrics:
```
Squat:
  Accuracy: 0.92
  Precision: 0.90
  Recall: 0.94

Deadlift:
  Accuracy: 0.88
  Precision: 0.86
  Recall: 0.90

...
```

### Step 2: Test on New Videos

Analyze a fresh workout video with your fine-tuned model:

```bash
python src/batch_runner.py "new_squat_video.mp4" \
  --exercise squat
```

Output:
```json
{
  "video": "new_squat_video.mp4",
  "exercise": "squat",
  "sets": [
    {
      "set_number": 1,
      "form_score": 87,
      "good_form_frames": 42,
      "bad_form_frames": 8,
      "issues": [
        {
          "timestamp": "0:23",
          "issue": "Shallow depth on reps 3-4"
        }
      ]
    }
  ],
  "overall_score": 87,
  "recommendations": ["Watch depth consistency"]
}
```

### Step 3: Manual Review

Check a few predictions manually:

1. Open results folder
2. Look at frames marked "bad_form"
3. Verify they actually have form issues
4. If not, model needs more training data

---

## Advanced: Multi-Class Classification

For more granular feedback, train with multiple form classes:

### Current Structure (Binary):
```json
{
  "squat": {
    "good_form": "...",
    "bad_form": "..."
  }
}
```

### Multi-Class Structure (Granular):
```json
{
  "squat": {
    "classes": {
      "perfect_form": "description",
      "good_form": "description",
      "partial_depth": "description",
      "knee_valgus": "description",
      "chest_collapse": "description",
      "bouncing": "description"
    }
  }
}
```

To use multi-class:

```bash
python scripts/finetune_clip.py \
  --data-dir data/labeled_frames \
  --multi-class \
  --output models/clip_finetuned_workout_multiclass.pt
```

This gives you:
- 85% accuracy instead of 92% (multi-class is harder)
- Much more detailed feedback ("knee valgus" vs just "bad form")

**Trade-off:** More specific feedback but requires more training data.

---

## Troubleshooting

### ❌ CUDA Out of Memory

```bash
# Reduce batch size
python scripts/finetune_clip.py \
  --batch-size 8  # was 16
```

Or use CPU (slower):
```bash
# In finetune_clip.py, change device to CPU
device = "cpu"
```

### ❌ Model Overfitting (train acc >> val acc)

```bash
# Need more diverse training data:
# - More videos per category
# - Different angles (front, side, rear)
# - Different gym environments
# - Different body types

# Or reduce model capacity:
python scripts/finetune_clip.py \
  --model openai/clip-vit-base-patch32  # (use smaller model)
```

### ❌ Low Accuracy (< 75%)

```bash
# Likely issue: Bad training descriptions
# Solution: Review CSV descriptions for quality

# Or: Not enough training data
# Solution: Collect 50+ videos per exercise

# Or: Learning rate wrong
python scripts/finetune_clip.py \
  --learning-rate 5e-6  # Lower
```

### ❌ "No training samples loaded"

```bash
# CSV descriptions are empty
# Solution: Fill in the description column in CSV before running
```

---

## Performance Benchmarks

**Expected results with good training data (50+ videos per category):**

| Model | Accuracy | Speed | GPU Memory |
|---|---|---|---|
| CLIP ViT-B/32 | 88% | Fast | 8GB |
| CLIP ViT-L/14 | 92% | Medium | 16GB |
| CLIP ViT-g/14 | 95% | Slow | 32GB |

**Recommended for MVP:** CLIP ViT-B/32 (balanced accuracy/speed)

---

## Production Workflow

### 1. Train Model
```bash
python scripts/finetune_clip.py --data-dir data/labeled_frames --epochs 10
```

### 2. Evaluate
```bash
python scripts/evaluate_model.py --model models/clip_finetuned_workout.pt
```

### 3. Deploy
Update `src/utils/clip_utils.py`:
```python
model_path = "models/clip_finetuned_workout.pt"
self.model.load_state_dict(torch.load(model_path))
```

### 4. Use in Pipeline
```bash
python src/batch_runner.py workout.mp4 --exercise squat
```

---

## Next Steps

1. ✅ Collect 30-50 videos per exercise
2. ✅ Extract frames: `scripts/extract_training_frames.py`
3. ✅ Annotate descriptions in CSV
4. ✅ Fine-tune: `scripts/finetune_clip.py`
5. ✅ Evaluate: `scripts/evaluate_model.py`
6. ✅ Deploy in production

---

## Resources

- CLIP Paper: https://arxiv.org/abs/2103.00020
- Hugging Face: https://huggingface.co/docs/transformers/tasks/zero_shot_image_classification
- PyTorch: https://pytorch.org/tutorials/

---

Built with ❤️ by Danh Le
