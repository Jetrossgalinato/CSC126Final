from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import cvzone
from ultralytics import YOLO
import threading
import time

app = FastAPI(title="Drone Surveillance System")

# CORS Middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO('best.pt')

# Class labels
class_names = {0: "Civilian", 1: "Soldier"}
class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # Green for Civilian, Red for Soldier

# Global statistics
stats = {
    "civilian_feed": {"soldier": 0, "civilian": 0},
    "soldier_feed": {"soldier": 0, "civilian": 0}
}
# Sets to track unique IDs
seen_ids = {
    "civilian_feed": {"soldier": set(), "civilian": set()},
    "soldier_feed": {"soldier": set(), "civilian": set()}
}
stats_lock = threading.Lock()

# Video sources
VIDEO_SOURCE_CIVILIAN = "drone_civilian.mp4"
VIDEO_SOURCE_SOLDIER = "drone_soldier.mp4"


def generate_frames(video_source: str, feed_type: str):
    """Generate video frames with real-time detection and classification"""
    global stats, seen_ids
    
    while True:
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_source}")
            break
        
        while True:
            ret, frame = cap.read()
            
            # Restart video when it ends
            if not ret:
                break
            
            # Run YOLOv8 tracking
            results = model.track(frame, persist=True, verbose=False)
            
            # Temporary counters for current frame (for display)
            current_frame_soldier = 0
            current_frame_civilian = 0
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0]) if box.conf is not None else 0.0
                    class_id = int(box.cls[0])
                    
                    # Get track ID if available
                    track_id = int(box.id[0]) if box.id is not None else None
                    
                    # Only process if class is valid (0 or 1)
                    if class_id in class_names:
                        class_name = class_names[class_id]
                        color = class_colors[class_id]
                        
                        # Update counters and seen IDs
                        if class_id == 1:  # Soldier
                            current_frame_soldier += 1
                            if track_id is not None:
                                seen_ids[feed_type]["soldier"].add(track_id)
                        else:  # Civilian
                            current_frame_civilian += 1
                            if track_id is not None:
                                seen_ids[feed_type]["civilian"].add(track_id)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Draw label with cvzone
                        label_text = f"{class_name} {confidence:.2f}"
                        if track_id is not None:
                            label_text += f" ID:{track_id}"
                            
                        cvzone.putTextRect(
                            frame, 
                            label_text, 
                            (max(0, x1), max(35, y1 - 10)),
                            scale=1,
                            thickness=2,
                            colorR=color,
                            colorT=(255, 255, 255),
                            offset=5
                        )
            
            # Update global stats with thread safety
            with stats_lock:
                stats[feed_type]["soldier"] = len(seen_ids[feed_type]["soldier"])
                stats[feed_type]["civilian"] = len(seen_ids[feed_type]["civilian"])
            
            # Add overall statistics overlay
            overlay_text = ""
            if feed_type == "civilian_feed":
                overlay_text = f"Civilians: {current_frame_civilian}"
            elif feed_type == "soldier_feed":
                overlay_text = f"Soldiers: {current_frame_soldier}"
            else:
                overlay_text = f"Soldiers: {current_frame_soldier} | Civilians: {current_frame_civilian}"

            cvzone.putTextRect(
                frame,
                overlay_text,
                (10, 30),
                scale=1.5,
                thickness=2,
                colorR=(0, 0, 0),
                colorT=(255, 255, 255),
                offset=10
            )
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to control frame rate
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Drone Surveillance System API",
        "endpoints": {
            "/video_feed/civilian": "Streaming video with civilian detection",
            "/video_feed/soldier": "Streaming video with soldier detection",
            "/stats": "Current detection statistics"
        }
    }


@app.get("/video_feed/civilian")
async def video_feed_civilian():
    """
    Stream processed civilian video with real-time detection
    Returns MJPEG stream
    """
    return StreamingResponse(
        generate_frames(VIDEO_SOURCE_CIVILIAN, "civilian_feed"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video_feed/soldier")
async def video_feed_soldier():
    """
    Stream processed soldier video with real-time detection
    Returns MJPEG stream
    """
    return StreamingResponse(
        generate_frames(VIDEO_SOURCE_SOLDIER, "soldier_feed"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video_feed")
async def video_feed():
    """
    Stream processed video with real-time human detection and classification
    Returns MJPEG stream (defaults to civilian feed for backward compatibility)
    """
    return StreamingResponse(
        generate_frames(VIDEO_SOURCE_CIVILIAN, "civilian_feed"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/stats")
async def get_stats():
    """
    Get real-time detection statistics
    Returns JSON with current counts of soldiers and civilians
    """
    with stats_lock:
        total_soldier = stats["civilian_feed"]["soldier"] + stats["soldier_feed"]["soldier"]
        total_civilian = stats["civilian_feed"]["civilian"] + stats["soldier_feed"]["civilian"]
        return {
            "soldier": total_soldier,
            "civilian": total_civilian,
            "total": total_soldier + total_civilian
        }


@app.post("/reset_stats")
async def reset_stats():
    """
    Reset all detection statistics and tracking IDs
    """
    global stats, seen_ids
    with stats_lock:
        stats = {
            "civilian_feed": {"soldier": 0, "civilian": 0},
            "soldier_feed": {"soldier": 0, "civilian": 0}
        }
        seen_ids = {
            "civilian_feed": {"soldier": set(), "civilian": set()},
            "soldier_feed": {"soldier": set(), "civilian": set()}
        }
    return {"message": "Stats reset successfully"}


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    cv2.destroyAllWindows()
