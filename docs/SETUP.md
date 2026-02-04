# Setup Guide

## Installation

### Prerequisites
- Python 3.9+
- FFmpeg + FFprobe
- CUDA 11.8+ (optional, for GPU acceleration)

### Step 1: Clone & Navigate

```bash
git clone https://github.com/odanree/workout-form-classifier
cd workout-form-classifier
```

### Step 2: Create Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install FFmpeg

**macOS** (Homebrew):
```bash
brew install ffmpeg
```

**Linux** (Ubuntu/Debian):
```bash
sudo apt-get install ffmpeg
```

**Windows** (Chocolatey):
```bash
choco install ffmpeg
```

### Step 5: Download/Setup Models

```bash
python scripts/setup_model.py
```

This downloads the base CLIP model and prepares it for fine-tuning.

---

## Quick Start

### Basic Usage

```bash
python src/batch_runner.py path/to/workout_video.mp4 --exercise squat
```

### With Options

```bash
python src/batch_runner.py workout.mp4 \
  --exercise deadlift \
  --threshold 1.0 \
  --output ./results \
  --skip-preview
```

### Output

Results saved to `./results/` (or specified `--output` dir):

```
results/
├── form_report.json          # Main analysis results
├── scenes_detected.json      # Scene boundaries
├── scene_frames/             # Extracted frames
│   ├── scene_0001/
│   ├── scene_0002/
│   └── ...
└── _metadata.json            # Execution metadata
```

---

## Configuration

Edit `config/workflow_config.json`:

```json
{
  "detection_threshold": 1.0,
  "min_scene_length": 0.5,
  "min_form_confidence": 0.5,
  "skip_preview": false,
  "output_formats": ["json"]
}
```

- **detection_threshold**: Scene cut sensitivity (lower = more scenes detected)
- **min_scene_length**: Minimum scene duration in seconds
- **min_form_confidence**: Minimum classification confidence to flag issue
- **skip_preview**: Skip Step 4 (web preview server)
- **output_formats**: Generate JSON, PDF, CSV reports

---

## Troubleshooting

### FFmpeg Not Found
```bash
# Verify FFmpeg is installed:
ffmpeg -version
ffprobe -version

# Add to PATH if needed:
# macOS/Linux: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
```

### CUDA/GPU Issues
```bash
# Force CPU mode (slower but works everywhere):
# Edit src/utils/clip_utils.py, line 10:
# Change: device = "cuda" if torch.cuda.is_available() else "cpu"
# To: device = "cpu"
```

### Out of Memory
```bash
# Reduce batch size in config
# Or process shorter videos first for testing
```

### Model Download Fails
```bash
# Manual download:
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

---

## Next Steps

1. **Collect Training Data** - See [DATA_COLLECTION.md](../docs/DATA_COLLECTION.md)
2. **Fine-tune CLIP** - See [TRAINING.md](../docs/TRAINING.md)
3. **Build Web UI** - Start with frontend scaffolding in `web/`
4. **Deploy** - See deployment guide below

---

## Development Setup

### Run Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/
```

### Local API Development

```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload

# Terminal 2: Frontend (if building web UI)
cd web
npm run dev
```

---

## Production Deployment

### Build Docker Image

```bash
docker build -t workout-form-classifier .
docker run -p 8000:8000 workout-form-classifier
```

### Deploy to Railway/Render

```bash
# Backend
railway up

# Frontend
vercel deploy
```

---

## Support

- Issues: [GitHub Issues](https://github.com/odanree/workout-form-classifier/issues)
- Docs: See `docs/` folder for detailed guides
- Email: dtle82@gmail.com

---

Built with ❤️ by Danh Le
