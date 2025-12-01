# Setup and Run Commands (using uv)

## Quick Start Commands

### 1. Install uv (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Navigate to project directory

```bash
cd /home/jetrossneri/Projects/CSC126Final
```

### 3. Install all dependencies

```bash
uv sync
```

### 4. Run the server

```bash
uv run python main.py
```

## Alternative Commands

### Run directly with uvicorn

```bash
uv run uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

### Add a new dependency

```bash
uv add <package-name>
```

### Check installed packages

```bash
uv pip list
```

## Access the API

After running the server:

- API Docs: http://localhost:8000/docs
- Video Feed: http://localhost:8000/video_feed
- Statistics: http://localhost:8000/stats
- Root: http://localhost:8000/

## Testing Endpoints

### Test statistics endpoint

```bash
curl http://localhost:8000/stats
```

### Test root endpoint

```bash
curl http://localhost:8000/
```

### Open video feed in browser

```bash
xdg-open http://localhost:8000/video_feed  # Linux
# or just visit http://localhost:8000/video_feed in your browser
```
