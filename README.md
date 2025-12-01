# Drone Surveillance System - YOLOv8 Backend

A high-performance FastAPI backend for real-time aerial drone surveillance that detects and classifies humans as "Soldier" or "Civilian" using YOLOv8.

## Features

- **Real-time Detection**: YOLOv8-powered human classification
- **Live Video Stream**: MJPEG streaming endpoint with bounding boxes
- **Real-time Statistics**: JSON endpoint for detection counts
- **Auto-restart**: Video automatically loops when finished
- **Visual Overlays**: Color-coded bounding boxes (Green=Civilian, Red=Soldier)

## Prerequisites

- Python 3.12+
- `uv` package manager (by Astral)
- `best.pt` (YOLOv8 model file) in project root
- `drone_test.mp4` (test video) in project root

## Setup Instructions (using uv)

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on macOS/Linux with Homebrew:

```bash
brew install uv
```

### 2. Navigate to Project Directory

```bash
cd /home/jetrossneri/Projects/CSC126Final
```

### 3. Install Dependencies

```bash
uv sync
```

This will create a virtual environment and install all dependencies from `pyproject.toml`.

### 4. Ensure Required Files are Present

Make sure these files exist in the project root:

- `best.pt` - Your trained YOLOv8 model
- `drone_test.mp4` - Test video file

### 5. Run the Server

```bash
uv run python main.py
```

The server will start on `http://0.0.0.0:8000`

## API Endpoints

### 1. Root Endpoint

```
GET /
```

Returns API information and available endpoints.

### 2. Video Feed (Streaming)

```
GET /video_feed
```

Returns MJPEG video stream with real-time detection and bounding boxes.

**Usage in Browser:**

```html
<img src="http://localhost:8000/video_feed" />
```

### 3. Statistics

```
GET /stats
```

Returns JSON with current detection counts.

**Response Example:**

```json
{
  "soldier": 2,
  "civilian": 5,
  "total": 7
}
```

## Class Detection

- **Class 0 - Civilian**: Green bounding box (RGB: 0, 255, 0)
- **Class 1 - Soldier**: Red bounding box (RGB: 0, 0, 255)

## Development Commands

### Install a new package

```bash
uv add <package-name>
```

### Remove a package

```bash
uv remove <package-name>
```

### Run with auto-reload (development)

```bash
uv run python main.py
```

### Direct uvicorn command (alternative)

```bash
uv run uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

## Project Structure

```
CSC126Final/
├── app/
│   ├── __init__.py
│   └── app.py          # Main FastAPI application
├── main.py             # Entry point
├── pyproject.toml      # Project dependencies (uv)
├── best.pt             # YOLOv8 model (required)
├── drone_test.mp4      # Test video (required)
└── README.md
```

## Technical Details

- **Framework**: FastAPI
- **Server**: Uvicorn with standard features
- **Model**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV + cvzone
- **CORS**: Enabled for all origins (development)
- **Threading**: Thread-safe statistics using locks
- **Video Format**: MJPEG streaming
- **Frame Rate**: ~30 FPS

## Troubleshooting

### Video file not found

Ensure `drone_test.mp4` is in the project root directory.

### Model file not found

Ensure `best.pt` is in the project root directory.

### Port already in use

Change the port in `main.py`:

```python
uvicorn.run("app.app:app", host="0.0.0.0", port=8001, reload=True)
```

## License

This project is for educational purposes (CSC126 Final Project).
