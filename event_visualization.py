"""
event_visualization.py — WIUT Hackathon 2026, Computer Vision track.

Renders bounding boxes, track IDs, zone overlays, event labels, and active event alerts
directly onto video frames.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

import solution
import zones

# Fallback definitions in case solution.py does not define class IDs or hardware devices
DEVICE = getattr(solution, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TRACKED_CLS_IDS = getattr(solution, "TRACKED_CLS_IDS", [0, 2, 3, 5, 7])
VEHICLE_CLS_IDS = getattr(solution, "VEHICLE_CLS_IDS", [2, 3, 5, 7])
PERSON_CLS_ID = getattr(solution, "PERSON_CLS_ID", 0)
OBSTACLE_CANDIDATE_CLS_IDS = getattr(solution, "OBSTACLE_CANDIDATE_CLS_IDS", [80])

# Color Palette for Event Banners & Class Bounding Boxes (BGR format)
COLOR_PALETTE = {
    "vehicle": (255, 178, 50),     # Cyan / Blue-Orange
    "person": (50, 255, 120),      # Bright Green
    "obstacle": (0, 165, 255),     # Orange
    "fire_smoke": (0, 0, 255),     # Red
    "default_box": (200, 200, 200),# Gray
    "zone_line": (255, 255, 0),    # Yellow
    "stop_line": (0, 0, 255),      # Red
    "alert_bg": (0, 0, 200),       # Bright Red for Alerts
    "alert_text": (255, 255, 255)  # White
}


def draw_zones_overlay(frame):
    """Draws defined zones (Crosswalks, Stop Lines, Red Light Zone, Lanes) on frame."""
    overlay = frame.copy()

    # 1. Crosswalks (Green transparent fill)
    if hasattr(zones, "ALL_CROSSWALKS"):
        for cw in zones.ALL_CROSSWALKS:
            poly = np.asarray(cw, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (0, 200, 0))
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

    # 2. Intersection Zone (Blue outline)
    if hasattr(zones, "INTERSECTION_ZONE"):
        poly = np.asarray(zones.INTERSECTION_ZONE, dtype=np.int32)
        cv2.polylines(frame, [poly], isClosed=True, color=(255, 200, 0), thickness=2)

    # 3. Stop Lines (Thick Red lines)
    if hasattr(zones, "STOP_LINES"):
        stop_lines = zones.STOP_LINES
        if isinstance(stop_lines, np.ndarray) and stop_lines.ndim == 2:
            stop_lines = [stop_lines]
        for line in stop_lines:
            poly = np.asarray(line, dtype=np.int32)
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 0, 255), thickness=3)

    # 4. Red Light Bulb Area (Magenta circle/polygon)
    if hasattr(zones, "RED_LIGHT_ZONE"):
        poly = np.asarray(zones.RED_LIGHT_ZONE, dtype=np.int32)
        cv2.polylines(frame, [poly], isClosed=True, color=(255, 0, 255), thickness=2)

    # Blend transparent overlays (alpha=0.2)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)


def draw_active_events_hud(frame, active_labels, current_sec):
    """Renders top banner showing current timestamp and active event alerts."""
    h, w = frame.shape[:2]

    # Header bar background
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)

    time_str = f"Time: {current_sec:.2f}s"
    cv2.putText(frame, time_str, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if not active_labels:
        cv2.putText(frame, "STATUS: NORMAL", (220, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        alert_str = "ALERTS: " + ", ".join(sorted(set(active_labels))).upper()
        # Red background highlight for active alerts
        cv2.rectangle(frame, (210, 8), (w - 10, 42), COLOR_PALETTE["alert_bg"], -1)
        cv2.putText(frame, alert_str, (220, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PALETTE["alert_text"], 2)


def render_visualized_video(video_path: str, output_path: str):
    """Main rendering routine: runs detection pipeline and draws boxes & HUD."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at path: '{video_path}'")

    print(f"Step 1/2: Running event detection pipeline on '{video_path}'...")
    events = solution.detect_events(video_path)
    print(f"Detected {len(events)} total events.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video stream: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load YOLO model for visualization tracking
    tracker_model = YOLO("yolov8n.pt")
    tracker_model.to(DEVICE)

    # Safe retrieval of auxiliary fire model if loaded in solution.py
    fire_model = getattr(solution, "_fire_smoke_model", None)
    if fire_model is None and hasattr(solution, "_PRELOADED_MODELS"):
        fire_model = solution._PRELOADED_MODELS.get("fire", None)

    frame_idx = 0
    print("Step 2/2: Rendering output video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps

        # Find active event names at current timestamp
        active_events = [
            label for start, end, label in events if start <= t_sec <= end
        ]

        # 1. Draw static zone overlays
        draw_zones_overlay(frame)

        # 2. Run tracker on current frame for bounding box rendering
        results = tracker_model.track(
            frame,
            persist=True,
            verbose=False,
            imgsz=480,
            classes=TRACKED_CLS_IDS
        )

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                results[0].boxes.id.cpu().numpy().astype(int)
                if results[0].boxes.id is not None
                else [None] * len(boxes)
            )

            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = map(int, box)

                # Determine base class color and label
                if cls_id in VEHICLE_CLS_IDS:
                    base_color = COLOR_PALETTE["vehicle"]
                    label_name = "Vehicle"
                elif cls_id == PERSON_CLS_ID:
                    base_color = COLOR_PALETTE["person"]
                    label_name = "Pedestrian"
                elif cls_id in OBSTACLE_CANDIDATE_CLS_IDS:
                    base_color = COLOR_PALETTE["obstacle"]
                    label_name = "Obstacle"
                else:
                    base_color = COLOR_PALETTE["default_box"]
                    label_name = f"Class_{cls_id}"

                # Construct dynamic caption with active event tag if events exist
                id_str = f" #{track_id}" if track_id is not None else ""

                if active_events:
                    # Render bounding box in alert color and append event name
                    box_color = COLOR_PALETTE["alert_bg"]
                    event_tag = f" [{', '.join(sorted(set(active_events))).upper()}]"
                    caption = f"{label_name}{id_str}{event_tag}"
                    text_color = COLOR_PALETTE["alert_text"]
                    box_thickness = 3
                else:
                    box_color = base_color
                    caption = f"{label_name}{id_str}"
                    text_color = (0, 0, 0)
                    box_thickness = 2

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

                # Draw caption badge directly on bounding box
                (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                badge_y1 = max(0, y1 - 22)
                badge_y2 = max(22, y1)

                cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 8, badge_y2), box_color, -1)
                cv2.putText(
                    frame,
                    caption,
                    (x1 + 4, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    1,
                    cv2.LINE_AA
                )

        # 3. Fire/Smoke detection bounding box render
        if fire_model is not None and "fire_smoke" in active_events:
            fire_res = fire_model.predict(frame, verbose=False, conf=0.25, imgsz=480)
            if fire_res and fire_res[0].boxes is not None:
                for fbox in fire_res[0].boxes.xyxy.cpu().numpy():
                    fx1, fy1, fx2, fy2 = map(int, fbox)
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), COLOR_PALETTE["fire_smoke"], 3)

                    fire_caption = "FIRE / SMOKE [EVENT: FIRE_SMOKE]"
                    (ftw, fth), _ = cv2.getTextSize(fire_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (fx1, max(0, fy1 - 25)), (fx1 + ftw + 6, max(25, fy1)), COLOR_PALETTE["fire_smoke"], -1)
                    cv2.putText(
                        frame,
                        fire_caption,
                        (fx1 + 3, max(18, fy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

        # 4. Render HUD Active Alerts Banner
        draw_active_events_hud(frame, active_events, t_sec)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"Render progress: frame {frame_idx}/{total_frames} ({t_sec:.1f}s)")

    cap.release()
    out.release()
    print(f"Visualization complete. Saved output to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize computer vision event predictions.")
    parser.add_argument(
        "--video",
        type=str,
        default=os.path.join("samples", "C3902.MP4"),
        help="Path to input video file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_visualized.mp4",
        help="Path for output video."
    )
    args = parser.parse_args()

    render_visualized_video(args.video, args.output)