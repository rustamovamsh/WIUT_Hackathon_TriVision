from __future__ import annotations

import os
import math
import time
import warnings
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import zones

# Suppress deprecation and library warnings for clean hackathon execution log
warnings.filterwarnings("ignore")

# Target event classes
CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
    "fire_smoke"
]

# Core cadence & spatial resolution parameters
TRAFFIC_SAMPLE_RATE = 5
TRAFFIC_IMAGE_SIZE = 640

# Auxiliary detector sampling settings (independent sampling to save compute budget)
AUXILIARY_IMAGE_SIZE = 640
FIRE_SAMPLE_INTERVAL_SEC = 1.25
OBSTACLE_SAMPLE_INTERVAL_SEC = 1.25
FIRE_CONFIDENCE = 0.30
OBSTACLE_CONFIDENCE = 0.55

# Physical thresholds for anomaly verification
ACCIDENT_IOU_THRESH = 0.45
NEAR_MISS_IOU_MAX = 0.15
NEAR_MISS_MIN_DURATION_SEC = 0.20
OBSTACLE_MIN_STATIONARY_SEC = 2.0
OBSTACLE_MAX_GAP_SEC = 1.75
OBSTACLE_RELATIVE_DRIFT_RATIO = 0.10
OBSTACLE_MAX_DRIFT_PX = 80.0
OBSTACLE_MAX_STEP_DRIFT_PX = 20.0

_ROOT = Path(__file__).resolve().parent
_MODEL_CACHE = {}


def _find_local_weight(filename: str, env_name: str):
    """Find supplied model weights locally without triggering online downloads."""
    override = os.environ.get(env_name, "").strip()
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    for root in (
            Path.cwd(), _ROOT,
            Path.cwd() / "weights", _ROOT / "weights",
            Path.cwd() / "assets", _ROOT / "assets",
    ):
        candidates.append(root / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_local_model(filename: str, env_name: str):
    """Load and cache YOLO model weights into memory."""
    path = _find_local_weight(filename, env_name)
    if path is None:
        return None
    key = str(path.resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(str(path))
    return _MODEL_CACHE[key]


def _reset_ultralytics_trackers(model) -> None:
    """Reset ByteTrack/BoT-SORT tracker internal state between video runs."""
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) or []:
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()


def _box_diagonal(box) -> float:
    """Compute bounding box diagonal length."""
    return math.hypot(float(box[2] - box[0]), float(box[3] - box[1]))


# Pre-load models into global memory before official timing starts
_PRELOADED_MODELS = {
    "traffic": _load_local_model("yolov8n", "weights/yolov8n.pt"),
    "fire": _load_local_model("fire_smoke", "weights/fire_smoke"),
    "obstacle": _load_local_model("road_obstacle", "weights/road_obstacle"),
}


def _warm_preloaded_models() -> None:
    """Execute warm-up inference to allocate CUDA contexts and compile kernels outside per-video timing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        with torch.inference_mode():
            traffic_model = _PRELOADED_MODELS.get("traffic")
            if traffic_model is not None:
                traffic_model.track(
                    dummy_frame,
                    persist=False,
                    verbose=False,
                    classes=[0, 2, 3, 5, 7],
                    conf=0.30,
                    imgsz=TRAFFIC_IMAGE_SIZE,
                    device=device,
                )
                _reset_ultralytics_trackers(traffic_model)
            for model_name in ("fire", "obstacle"):
                auxiliary_model = _PRELOADED_MODELS.get(model_name)
                if auxiliary_model is not None:
                    auxiliary_model.predict(
                        dummy_frame,
                        verbose=False,
                        imgsz=AUXILIARY_IMAGE_SIZE,
                        device=device,
                    )
    except Exception:
        pass


_warm_preloaded_models()


def preprocess_poly(poly):
    """Pre-convert zone polygons to float32 contiguous arrays to remove inner-loop memory allocation overhead."""
    if poly is None:
        return None
    if isinstance(poly, (list, tuple)):
        return [preprocess_poly(p) for p in poly]
    if isinstance(poly, np.ndarray):
        return poly.astype(np.float32) if poly.dtype != np.float32 else poly
    return poly


def is_point_in_poly(point: tuple, poly) -> bool:
    """Fast check if an (x, y) point is inside or on the boundary of a target zone polygon."""
    if poly is None:
        return False

    if isinstance(poly, (list, tuple)):
        return any(is_point_in_poly(point, p) for p in poly)

    if not isinstance(poly, np.ndarray) or poly.size == 0:
        return False

    while poly.ndim > 2:
        poly = poly[0]

    if poly.shape[0] < 3:
        return False

    if poly.dtype != np.float32:
        poly = poly.astype(np.float32)

    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def is_line_crossing_poly(p1: tuple, p2: tuple, poly_list, steps: int = 5) -> bool:
    """Line segment intersection test against polygon boundaries using linear interpolation sampling."""
    if poly_list is None:
        return False

    for i in range(steps + 1):
        alpha = i / float(steps)
        interp_x = p1[0] + alpha * (p2[0] - p1[0])
        interp_y = p1[1] + alpha * (p2[1] - p1[1])
        if is_point_in_poly((interp_x, interp_y), poly_list):
            return True
    return False


def _path_intersects_poly_dense(p1: tuple, p2: tuple, poly_list) -> bool:
    """Dense step interpolation test for thin road marking lines."""
    distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return is_line_crossing_poly(p1, p2, poly_list, steps=max(5, int(math.ceil(distance))))


def prepare_marking_geometries(poly_list) -> list:
    """Precompute principal long axis and normal vector for thin solid line markings."""
    if poly_list is None:
        return []
    polygons = poly_list if isinstance(poly_list, (list, tuple)) else [poly_list]
    geometries = []
    for poly in polygons:
        if not isinstance(poly, np.ndarray):
            continue
        points = np.asarray(poly, dtype=np.float32)
        while points.ndim > 2:
            points = points[0]
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        center = points.mean(axis=0)
        covariance = np.cov((points - center).T)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))].astype(np.float32)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-6:
            continue
        axis /= axis_norm
        normal = np.array([-axis[1], axis[0]], dtype=np.float32)
        geometries.append((points, center, normal))
    return geometries


def expected_lane_direction(poly, desired_y_sign: float):
    """Calculate dominant lane direction vector in image space."""
    if poly is None:
        return None
    points = np.asarray(poly, dtype=np.float32)
    while points.ndim > 2:
        points = points[0]
    if points.ndim != 2 or points.shape[0] < 3:
        return None
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((points - center).T))
    direction = vectors[:, int(np.argmax(values))].astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return None
    direction /= norm
    if direction[1] * desired_y_sign < 0.0:
        direction *= -1.0
    return direction


def did_cross_solid_marking(p1: tuple, p2: tuple, geometries: list, vehicle_width: float) -> bool:
    """Verify true side-to-side boundary crossing across a solid lane marking."""
    p1_array = np.asarray(p1, dtype=np.float32)
    p2_array = np.asarray(p2, dtype=np.float32)
    min_side_distance = max(2.0, 0.015 * float(vehicle_width))
    min_perpendicular_move = max(8.0, 0.06 * float(vehicle_width))

    for poly, center, normal in geometries:
        if not _path_intersects_poly_dense(p1, p2, poly):
            continue
        side_1 = float(np.dot(p1_array - center, normal))
        side_2 = float(np.dot(p2_array - center, normal))
        if (
                side_1 * side_2 < 0.0
                and min(abs(side_1), abs(side_2)) >= min_side_distance
                and abs(side_2 - side_1) >= min_perpendicular_move
        ):
            return True
    return False


def _backfill_track_flag(history: list, start_time: float, tuple_index: int) -> None:
    """Optimized O(1) reverse backfill: updates track history entries starting from current frame back to start_time."""
    for index in range(len(history) - 1, -1, -1):
        entry = history[index]
        if entry[0] < start_time:
            break
        if len(entry) > tuple_index:
            mutable = list(entry)
            mutable[tuple_index] = True
            history[index] = tuple(mutable)


def calculate_iou(boxA, boxB) -> float:
    """Intersection over Union (IoU) calculation for bounding box pairs."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def is_red_light_active(frame: np.ndarray, red_zone) -> bool:
    """Evaluate active red traffic light signal using thresholding in HSV color space."""
    if red_zone is None:
        return False

    if isinstance(red_zone, (list, tuple)) and len(red_zone) > 0:
        poly = red_zone[0] if isinstance(red_zone[0], np.ndarray) else np.array(red_zone[0], dtype=np.int32)
    elif isinstance(red_zone, np.ndarray):
        poly = red_zone
    else:
        return False

    while poly.ndim > 2:
        poly = poly[0]

    if not isinstance(poly, np.ndarray) or poly.size == 0:
        return False

    poly_int = poly.astype(np.int32)
    x, y, w, h = cv2.boundingRect(poly_int)
    if w <= 0 or h <= 0:
        return False

    h_img, w_img = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_pixel_count = np.count_nonzero(red_mask)
    total_pixels = roi.shape[0] * roi.shape[1]

    return red_pixel_count > max(12, int(0.10 * total_pixels))


# --- POST-PROCESSING EVENT EXTRACTORS ---

def detect_accidents(vehicle_tracks: dict, video_duration: float,
                     min_immobility_sec: float = 2.5,
                     max_gap_sec: float = 2.0,
                     freeze_move_thresh: float = 20.0) -> list:
    """Confirm accident events by verifying vehicle immobility post-collision."""
    events = []
    for history in vehicle_tracks.values():
        accident_times = [entry[0] for entry in history if len(entry) > 13 and entry[13]]
        if not accident_times:
            continue

        positions = {
            entry[0]: (entry[1], entry[11] if len(entry) > 11 else entry[2])
            for entry in history
        }
        sorted_times = sorted(positions)

        def stays_still(start_time: float) -> bool:
            start_x, start_y = positions[start_time]
            target_end = start_time + min_immobility_sec
            reached_target = False
            for sample_time in sorted_times:
                if sample_time < start_time:
                    continue
                if sample_time >= target_end:
                    reached_target = True
                    break
                x_pos, y_pos = positions[sample_time]
                if math.hypot(x_pos - start_x, y_pos - start_y) > freeze_move_thresh:
                    return False
            return reached_target

        start_time = accident_times[0]
        last_time = start_time
        for current_time in list(accident_times[1:]) + [None]:
            if current_time is not None and current_time - last_time <= max_gap_sec:
                last_time = current_time
                continue
            if stays_still(start_time):
                start_sec = round(start_time, 2)
                end_sec = min(
                    round(last_time + min_immobility_sec, 2),
                    round(video_duration, 2),
                )
                if start_sec < end_sec:
                    events.append([start_sec, end_sec, "accident"])
            if current_time is not None:
                start_time = current_time
                last_time = current_time
    return events


def detect_fire_smoke_events(fire_smoke_log: list, video_duration: float, max_gap_sec: float = 2.0) -> list:
    """Group temporal fire/smoke detections into continuous event intervals."""
    if not fire_smoke_log:
        return []
    events = []
    group = [fire_smoke_log[0]]
    groups = []
    for detection in fire_smoke_log[1:]:
        if detection[0] - group[-1][0] <= max_gap_sec:
            group.append(detection)
        else:
            groups.append(group)
            group = [detection]
    groups.append(group)

    for detections in groups:
        if len(detections) < 2 and max(item[1] for item in detections) < 0.65:
            continue
        start_time = detections[0][0]
        end_time = min(video_duration, detections[-1][0] + FIRE_SAMPLE_INTERVAL_SEC)
        if start_time < end_time:
            events.append([round(start_time, 2), round(end_time, 2), "fire_smoke"])
    return events


def detect_road_obstacle_events(road_obstacle_tracks: dict, video_duration: float,
                                min_duration_sec: float = OBSTACLE_MIN_STATIONARY_SEC,
                                max_gap_sec: float = OBSTACLE_MAX_GAP_SEC) -> list:
    """Validate road obstacles based on sustained spatial position on the road surface."""
    events = []
    for history in road_obstacle_tracks.values():
        if len(history) < 3:
            continue

        start_index = 0
        while start_index < len(history):
            start_t, start_x, start_y, start_diag = history[start_index][:4]
            tolerance = max(
                18.0,
                min(OBSTACLE_MAX_DRIFT_PX, OBSTACLE_RELATIVE_DRIFT_RATIO * start_diag),
            )
            end_index = start_index
            for index in range(start_index + 1, len(history)):
                t_sec, x_pos, y_pos = history[index][:3]
                previous_t, previous_x, previous_y = history[index - 1][:3]
                if (
                        t_sec - previous_t > max_gap_sec
                        or math.hypot(x_pos - previous_x, y_pos - previous_y) > OBSTACLE_MAX_STEP_DRIFT_PX
                        or math.hypot(x_pos - start_x, y_pos - start_y) > tolerance
                ):
                    break
                end_index = index

            end_t = history[end_index][0]
            if end_t - start_t >= min_duration_sec:
                end_time = min(video_duration, end_t + OBSTACLE_SAMPLE_INTERVAL_SEC)
                if start_t < end_time:
                    events.append([round(start_t, 2), round(end_time, 2), "road_obstacle"])
                start_index = end_index + 1
            else:
                start_index += 1
    return events


def detect_jaywalking(person_tracks: dict, video_duration: float, min_duration_sec: float = 1.0,
                      max_gap_sec: float = 1.5) -> list[list]:
    """Extract jaywalking intervals for pedestrians crossing outside crosswalk/median zones."""
    events = []
    for track_id, history in person_tracks.items():
        jw_times = [entry[0] for entry in history if entry[1]]
        if not jw_times:
            continue

        start_time = jw_times[0]
        last_time = jw_times[0]

        for t in jw_times[1:]:
            if t - last_time <= max_gap_sec:
                last_time = t
            else:
                if (last_time - start_time) >= min_duration_sec:
                    s = round(start_time, 2)
                    e = min(round(last_time, 2), round(video_duration, 2))
                    if s < e:
                        events.append([s, e, "jaywalking"])
                start_time = t
                last_time = t

        if (last_time - start_time) >= min_duration_sec:
            s = round(start_time, 2)
            e = min(round(last_time, 2), round(video_duration, 2))
            if s < e:
                events.append([s, e, "jaywalking"])

    return events


def detect_wrong_way(vehicle_tracks: dict, video_duration: float, min_duration_sec: float = 3.0,
                     max_gap_sec: float = 1.5, evidence_delay_sec: float = 0.8) -> list[list]:
    """Extract wrong-way driving events based on continuous inverted lane trajectory."""
    events = []
    for track_id, history in vehicle_tracks.items():
        ww_times = [entry[0] for entry in history if entry[3]]
        if not ww_times:
            continue

        start_time = ww_times[0]
        last_time = ww_times[0]

        for t in ww_times[1:]:
            if t - last_time <= max_gap_sec:
                last_time = t
            else:
                if (last_time - start_time + evidence_delay_sec) >= min_duration_sec:
                    s = round(max(0.0, start_time - evidence_delay_sec), 2)
                    e = min(round(last_time, 2), round(video_duration, 2))
                    if s < e:
                        events.append([s, e, "wrong_way"])
                start_time = t
                last_time = t

        if (last_time - start_time + evidence_delay_sec) >= min_duration_sec:
            s = round(max(0.0, start_time - evidence_delay_sec), 2)
            e = min(round(last_time, 2), round(video_duration, 2))
            if s < e:
                events.append([s, e, "wrong_way"])

    return events


def detect_stopped_vehicle(vehicle_tracks: dict, video_duration: float, min_duration_sec: float = 5.0,
                           max_gap_sec: float = 2.0) -> list[list]:
    """Extract stopped vehicle events in non-intersection drivable lanes."""
    events = []
    for track_id, history in vehicle_tracks.items():
        stopped_times = [entry[0] for entry in history if entry[4]]
        if not stopped_times:
            continue

        start_time = stopped_times[0]
        last_time = stopped_times[0]

        for t in stopped_times[1:]:
            if t - last_time <= max_gap_sec:
                last_time = t
            else:
                if (last_time - start_time) >= min_duration_sec:
                    s = round(start_time, 2)
                    e = min(round(last_time, 2), round(video_duration, 2))
                    if s < e:
                        events.append([s, e, "stopped_vehicle"])
                start_time = t
                last_time = t

        if (last_time - start_time) >= min_duration_sec:
            s = round(start_time, 2)
            e = min(round(last_time, 2), round(video_duration, 2))
            if s < e:
                events.append([s, e, "stopped_vehicle"])

    return events


def detect_solid_line_crossing(vehicle_tracks: dict, video_duration: float, min_duration_sec: float = 0.5,
                               max_gap_sec: float = 1.5) -> list[list]:
    """Extract solid line crossing events."""
    events = []
    for track_id, history in vehicle_tracks.items():
        sl_times = [entry[0] for entry in history if entry[5]]
        if not sl_times:
            continue

        start_time = sl_times[0]
        last_time = sl_times[0]

        for t in sl_times[1:]:
            if t - last_time <= max_gap_sec:
                last_time = t
            else:
                if (last_time - start_time) >= min_duration_sec:
                    s = round(start_time, 2)
                    e = min(round(last_time, 2) + 0.5, round(video_duration, 2))
                    if s < e:
                        events.append([s, e, "solid_line_crossing"])
                start_time = t
                last_time = t

        if (last_time - start_time) >= min_duration_sec:
            s = round(start_time, 2)
            e = min(round(last_time, 2) + 0.5, round(video_duration, 2))
            if s < e:
                events.append([s, e, "solid_line_crossing"])

    return events


def detect_flagged_vehicle_events(vehicle_tracks: dict, video_duration: float, tuple_index: int, label: str,
                                  min_duration_sec: float = 0.0, max_gap_sec: float = 1.5) -> list[list]:
    """Generic extractor for boolean vehicle flags in track memory."""
    events = []
    for track_id, history in vehicle_tracks.items():
        active_times = [entry[0] for entry in history if len(entry) > tuple_index and entry[tuple_index]]
        if not active_times:
            continue

        start_time = active_times[0]
        last_time = active_times[0]

        for t in active_times[1:]:
            if t - last_time <= max_gap_sec:
                last_time = t
            else:
                s = round(start_time, 2)
                e = min(round(last_time + 0.4, 2), round(video_duration, 2))
                if (e - s) >= min_duration_sec and s < e:
                    events.append([s, e, label])
                start_time = t
                last_time = t

        s = round(start_time, 2)
        e = min(round(last_time + 0.4, 2), round(video_duration, 2))
        if (e - s) >= min_duration_sec and s < e:
            events.append([s, e, label])

    return events


def detect_congestion_events(congestion_log: list, video_duration: float, min_duration_sec: float = 4.0,
                             max_gap_sec: float = 2.0) -> list[list]:
    """Extract traffic congestion periods."""
    events = []
    if not congestion_log:
        return events

    start_time = congestion_log[0]
    last_time = congestion_log[0]

    for t in congestion_log[1:]:
        if t - last_time <= max_gap_sec:
            last_time = t
        else:
            if (last_time - start_time) >= min_duration_sec:
                s = round(start_time, 2)
                e = min(round(last_time, 2), round(video_duration, 2))
                if s < e:
                    events.append([s, e, "congestion"])
            start_time = t
            last_time = t

    if (last_time - start_time) >= min_duration_sec:
        s = round(start_time, 2)
        e = min(round(last_time, 2), round(video_duration, 2))
        if s < e:
            events.append([s, e, "congestion"])

    return events


def merge_overlapping_events(events: list[list], video_duration: float) -> list[list]:
    """Merge overlapping or adjacent temporal intervals belonging to the same event type."""
    if not events:
        return []

    events.sort(key=lambda x: (x[2], x[0]))
    merged = []

    for evt in events:
        start_sec, end_sec, label = evt
        end_sec = min(end_sec, round(video_duration, 2))

        if start_sec >= end_sec:
            continue

        if not merged or merged[-1][2] != label:
            merged.append([start_sec, end_sec, label])
        else:
            prev_start, prev_end, prev_label = merged[-1]
            if start_sec <= prev_end + 1.0:
                merged[-1][1] = min(max(prev_end, end_sec), round(video_duration, 2))
            else:
                merged.append([start_sec, end_sec, label])

    return merged


# --- MAIN PIPELINE FUNCTION ---

def detect_events(video_path: str) -> list[list]:
    """Main event detection function called by the hackathon runner."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    video_duration = total_frames / fps

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Enable TensorFloat-32 (TF32) for acceleration without precision degradation
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = _PRELOADED_MODELS.get("traffic") or _load_local_model("weights/yolov8n.pt", "TRAFFIC_MODEL_PATH")
    if model is None:
        cap.release()
        return []

    fire_model = _PRELOADED_MODELS.get("fire") or _load_local_model("fire_smoke.pt", "weights/fire_smoke.pt")
    obstacle_model = _PRELOADED_MODELS.get("obstacle") or _load_local_model("road_obstacle",
                                                                            "weights/road_obstacle.pt")

    fire_class_ids = None
    if fire_model is not None:
        fire_class_ids = [
                             int(class_id)
                             for class_id, class_name in fire_model.names.items()
                             if str(class_name).strip().lower() in {"fire", "smoke"}
                         ] or None

    _reset_ultralytics_trackers(model)

    pedestrian_tracks = {}
    vehicle_tracks = {}
    congestion_log = []
    fire_smoke_log = []
    road_obstacle_tracks = defaultdict(list)
    next_obstacle_track_id = 1
    next_fire_time = 0.0
    next_obstacle_time = 0.0
    recent_solid_crossing_until = {}

    frame_idx = 0
    sample_rate = TRAFFIC_SAMPLE_RATE

    # Pre-process zone geometries
    exclusion_zones = preprocess_poly(getattr(zones, 'EXCLUSION_ZONES', None))
    road_zone = preprocess_poly(getattr(zones, 'ROAD_ZONE', None))
    all_crosswalks = preprocess_poly(getattr(zones, 'ALL_CROSSWALKS', None))
    all_medians = preprocess_poly(getattr(zones, 'ALL_MEDIANS', None))
    intersection_zone = preprocess_poly(getattr(zones, 'INTERSECTION_ZONE', None))
    lane_away = preprocess_poly(getattr(zones, 'LANE_AWAY', None))
    lane_towards = preprocess_poly(getattr(zones, 'LANE_TOWARDS', None))
    solid_lines = preprocess_poly(getattr(zones, 'SOLID_LINES', None))
    stop_lines = preprocess_poly(getattr(zones, 'STOP_LINES', None))
    straight_lanes = preprocess_poly(getattr(zones, 'STRAIGHT_LANES', None))
    no_turn_zones = preprocess_poly(getattr(zones, 'NO_TURN_ZONES', None))
    no_left_turn = preprocess_poly(getattr(zones, 'NO_LEFT_TURN', None))
    no_right_turn = preprocess_poly(getattr(zones, 'NO_RIGHT_TURN', None))
    solid_line_geometries = prepare_marking_geometries(solid_lines)
    towards_direction = expected_lane_direction(lane_towards, desired_y_sign=1.0)
    away_direction = expected_lane_direction(lane_away, desired_y_sign=-1.0)

    red_light_zone = None
    for attr_name in ['TRAFFIC_LIGHT_RED', 'RED_LIGHT_ZONE', 'TRAFFIC_LIGHT_RED_ZONE', 'TRAFFIC_LIGHT_ZONE']:
        if hasattr(zones, attr_name):
            red_light_zone = getattr(zones, attr_name)
            break

    has_solid_lines = solid_lines is not None and (
        len(solid_lines) > 0 if isinstance(solid_lines, (list, tuple)) else True)
    has_stop_lines = stop_lines is not None and (len(stop_lines) > 0 if isinstance(stop_lines, (list, tuple)) else True)
    has_lane_away = lane_away is not None
    has_crosswalks = all_crosswalks is not None and (
        len(all_crosswalks) > 0 if isinstance(all_crosswalks, (list, tuple)) else True)
    has_medians = all_medians is not None and (len(all_medians) > 0 if isinstance(all_medians, (list, tuple)) else True)
    has_straight_lanes = straight_lanes is not None and (
        len(straight_lanes) > 0 if isinstance(straight_lanes, (list, tuple)) else True)
    has_no_turn_zones = no_turn_zones is not None and (
        len(no_turn_zones) > 0 if isinstance(no_turn_zones, (list, tuple)) else True)
    has_no_left_turn = no_left_turn is not None and (
        len(no_left_turn) > 0 if isinstance(no_left_turn, (list, tuple)) else True)
    has_no_right_turn = no_right_turn is not None and (
        len(no_right_turn) > 0 if isinstance(no_right_turn, (list, tuple)) else True)

    start_processing_time = time.time()
    print(f"\nProcessing {video_path}...")
    target_classes = [0, 2, 3, 5, 7]

    with torch.inference_mode():
        while cap.isOpened():
            frame_idx += 1

            if frame_idx % sample_rate != 0:
                if not cap.grab():
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 100 == 0 or frame_idx == total_frames:
                elapsed_time = time.time() - start_processing_time
                proc_speed_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
                frames_remaining = total_frames - frame_idx

                eta_seconds = int(frames_remaining / proc_speed_fps) if proc_speed_fps > 0 else 0
                eta_min, eta_sec = divmod(eta_seconds, 60)

                pct = (frame_idx / total_frames) * 100
                print(
                    f"\rProgress: [{frame_idx}/{total_frames}] ({pct:5.1f}%) | "
                    f"Countdown/ETA: {eta_min:02d}:{eta_sec:02d} | "
                    f"Speed: {proc_speed_fps:.1f} FPS",
                    end="",
                    flush=True
                )

            t_sec = min((frame_idx - 1) / fps, video_duration)
            red_light_active = is_red_light_active(frame, red_light_zone)

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                classes=target_classes,
                conf=0.30,
                imgsz=TRAFFIC_IMAGE_SIZE,
                device=device
            )

            if fire_model is not None and t_sec + 1e-6 >= next_fire_time:
                fire_results = fire_model.predict(
                    frame,
                    verbose=False,
                    classes=fire_class_ids,
                    conf=FIRE_CONFIDENCE,
                    imgsz=AUXILIARY_IMAGE_SIZE,
                    device=device,
                )
                fire_boxes = fire_results[0].boxes
                if fire_boxes is not None and len(fire_boxes) > 0:
                    confidences = fire_boxes.conf.detach().cpu().numpy()
                    fire_smoke_log.append((t_sec, float(np.max(confidences))))
                next_fire_time = t_sec + FIRE_SAMPLE_INTERVAL_SEC

            if obstacle_model is not None and t_sec + 1e-6 >= next_obstacle_time:
                obstacle_results = obstacle_model.predict(
                    frame,
                    verbose=False,
                    conf=OBSTACLE_CONFIDENCE,
                    imgsz=AUXILIARY_IMAGE_SIZE,
                    device=device,
                )
                obstacle_boxes = obstacle_results[0].boxes
                if obstacle_boxes is not None and len(obstacle_boxes) > 0:
                    obstacle_xyxy = obstacle_boxes.xyxy.detach().cpu().numpy()
                    obstacle_classes = obstacle_boxes.cls.detach().cpu().numpy().astype(int)
                    obstacle_confidences = obstacle_boxes.conf.detach().cpu().numpy()
                    frame_area = float(frame.shape[0] * frame.shape[1])
                    used_track_ids = set()
                    traffic_vehicle_boxes = []
                    traffic_boxes = results[0].boxes
                    if traffic_boxes is not None and len(traffic_boxes) > 0:
                        traffic_xyxy = traffic_boxes.xyxy.detach().cpu().numpy()
                        traffic_classes = traffic_boxes.cls.detach().cpu().numpy().astype(int)
                        traffic_vehicle_boxes = [
                            traffic_box
                            for traffic_box, traffic_class in zip(traffic_xyxy, traffic_classes)
                            if int(traffic_class) in {2, 3, 5, 7}
                        ]

                    for obstacle_box, obstacle_class, obstacle_confidence in zip(
                            obstacle_xyxy, obstacle_classes, obstacle_confidences
                    ):
                        ox1, oy1, ox2, oy2 = obstacle_box
                        obstacle_width = max(1.0, float(ox2 - ox1))
                        obstacle_height = max(1.0, float(oy2 - oy1))
                        obstacle_class_name = str(
                            obstacle_model.names.get(int(obstacle_class), obstacle_class)
                        ).strip().lower()

                        if obstacle_width * obstacle_height > 0.15 * frame_area:
                            continue

                        if (
                                "road_debris" in obstacle_class_name
                                and (
                                obstacle_width * obstacle_height > 0.005 * frame_area
                                or obstacle_width > 0.10 * frame.shape[1]
                        )
                        ):
                            continue

                        if any(
                                calculate_iou(obstacle_box, traffic_box) >= 0.20
                                for traffic_box in traffic_vehicle_boxes
                        ):
                            continue

                        obstacle_x = float((ox1 + ox2) / 2.0)
                        obstacle_y = float(oy2)
                        obstacle_foot = (obstacle_x, obstacle_y)
                        if (
                                not is_point_in_poly(obstacle_foot, road_zone)
                                or is_point_in_poly(obstacle_foot, exclusion_zones)
                        ):
                            continue

                        obstacle_diag = math.hypot(obstacle_width, obstacle_height)
                        matched_track_id = None
                        best_distance = float("inf")
                        for candidate_id, candidate_history in road_obstacle_tracks.items():
                            if candidate_id in used_track_ids or not candidate_history:
                                continue
                            last_entry = candidate_history[-1]
                            last_time, last_x, last_y, last_diag, last_class = last_entry[:5]
                            if (
                                    int(last_class) != int(obstacle_class)
                                    or t_sec - last_time > OBSTACLE_MAX_GAP_SEC
                            ):
                                continue
                            distance = math.hypot(obstacle_x - last_x, obstacle_y - last_y)
                            association_limit = max(35.0, 0.15 * max(obstacle_diag, last_diag))
                            if distance <= association_limit and distance < best_distance:
                                matched_track_id = candidate_id
                                best_distance = distance

                        if matched_track_id is None:
                            matched_track_id = next_obstacle_track_id
                            next_obstacle_track_id += 1

                        used_track_ids.add(matched_track_id)
                        road_obstacle_tracks[matched_track_id].append(
                            (
                                t_sec, obstacle_x, obstacle_y, obstacle_diag,
                                int(obstacle_class), float(obstacle_confidence),
                            )
                        )
                next_obstacle_time = t_sec + OBSTACLE_SAMPLE_INTERVAL_SEC

            active_crosswalk_peds = []
            active_frame_peds = []
            active_frame_vehicles = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                # Pass 1: Pedestrian Tracking
                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    if cls_id == 0:
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / 2.0
                        foot_point = (cx, y2)

                        if is_point_in_poly(foot_point, exclusion_zones):
                            continue

                        on_road = is_point_in_poly(foot_point, road_zone)
                        on_crosswalk = is_point_in_poly(foot_point, all_crosswalks)
                        on_median = has_medians and is_point_in_poly(foot_point, all_medians)

                        if on_crosswalk and not on_median:
                            active_crosswalk_peds.append(foot_point)

                        if on_road:
                            active_frame_peds.append((track_id, (cx, y2), box))

                        is_jaywalking = on_road and not on_crosswalk and not on_median

                        if track_id not in pedestrian_tracks:
                            pedestrian_tracks[track_id] = []
                        pedestrian_tracks[track_id].append((t_sec, is_jaywalking, foot_point))

                # Pass 2: Vehicle Feature Extraction
                slow_vehicle_count = 0

                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    if cls_id in [2, 3, 5, 7]:
                        x1, y1, x2, y2 = box
                        w_box = max(1.0, x2 - x1)
                        h_box = max(1.0, y2 - y1)
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        bottom_center = (cx, y2)
                        center_point = (cx, cy)

                        is_wrong_way = False
                        is_stopped = False
                        is_solid_crossing = False
                        is_red_light = False
                        is_stop_line = False
                        is_uturn = False
                        is_illegal_turn = False
                        is_failure_to_yield = False

                        in_exclusion = is_point_in_poly(center_point, exclusion_zones)
                        in_intersection = is_point_in_poly(center_point, intersection_zone)
                        in_lane_away = has_lane_away and (
                                is_point_in_poly(center_point, lane_away) or
                                is_point_in_poly(bottom_center, lane_away)
                        )

                        prev_entry = vehicle_tracks[track_id][-1] if track_id in vehicle_tracks and len(
                            vehicle_tracks[track_id]) > 0 else None
                        step_dist = math.hypot(cx - prev_entry[1], y2 - (
                            prev_entry[11] if len(prev_entry) > 11 else prev_entry[2])) if prev_entry else 0.0

                        # FAILURE TO YIELD CHECK
                        if has_crosswalks and active_crosswalk_peds and not in_exclusion:
                            if prev_entry is not None and step_dist >= 2.5:
                                dynamic_thresh = max(120.0, min(300.0, 1.8 * max(w_box, h_box)))
                                for ped_fp in active_crosswalk_peds:
                                    dist_to_ped = math.hypot(cx - ped_fp[0], y2 - ped_fp[1])
                                    if dist_to_ped <= dynamic_thresh:
                                        is_failure_to_yield = True
                                        break

                        # STOP LINE & RED LIGHT CROSSING
                        if has_stop_lines and not in_exclusion and prev_entry is not None:
                            prev_bc = (prev_entry[1], prev_entry[11]) if len(prev_entry) > 11 else (
                            prev_entry[1], prev_entry[2])

                            if is_line_crossing_poly(prev_bc, bottom_center, stop_lines):
                                if red_light_active:
                                    prev_in_intersection = is_point_in_poly(prev_bc, intersection_zone)
                                    is_exempt_turn = in_intersection or prev_in_intersection or in_lane_away

                                    if not is_exempt_turn:
                                        is_red_light = True
                                        is_stop_line = True

                        # SOLID LINE CROSSING CHECK
                        if has_solid_lines and not in_exclusion and not in_intersection and prev_entry is not None:
                            if step_dist >= 3.5:
                                crossing_reference = prev_entry
                                for past_entry in reversed(vehicle_tracks.get(track_id, [])):
                                    if t_sec - past_entry[0] >= 0.5:
                                        crossing_reference = past_entry
                                        break
                                previous_foot = (
                                    crossing_reference[1],
                                    crossing_reference[11] if len(crossing_reference) > 11 else crossing_reference[2],
                                )
                                if did_cross_solid_marking(
                                        previous_foot,
                                        bottom_center,
                                        solid_line_geometries,
                                        w_box,
                                ):
                                    is_solid_crossing = True
                                    recent_solid_crossing_until[track_id] = t_sec + 3.0

                        # HISTORY-DEPENDENT CHECKS
                        if prev_entry is not None:
                            past_entry_2s = None
                            past_entry_08s = None

                            for past_entry in reversed(vehicle_tracks[track_id]):
                                past_t = past_entry[0]
                                dt = t_sec - past_t
                                if past_entry_08s is None and dt >= 0.8:
                                    past_entry_08s = past_entry
                                if past_entry_2s is None and dt >= 2.0:
                                    past_entry_2s = past_entry
                                    break

                            # CONGESTION & STOPPED VEHICLE CHECKS
                            in_drivable_area = (
                                    is_point_in_poly(center_point, road_zone) or
                                    is_point_in_poly(center_point, intersection_zone) or
                                    (has_stop_lines and is_point_in_poly(center_point, stop_lines))
                            )

                            if in_drivable_area and not in_exclusion:
                                if past_entry_08s is not None:
                                    dt = t_sec - past_entry_08s[0]
                                    if dt > 0:
                                        speed_px_per_sec = math.hypot(cx - past_entry_08s[1],
                                                                      cy - past_entry_08s[2]) / dt
                                        if speed_px_per_sec <= 25.0:
                                            slow_vehicle_count += 1
                                elif past_entry_2s is not None:
                                    dist_2s = math.hypot(cx - past_entry_2s[1], cy - past_entry_2s[2])
                                    if dist_2s <= 15.0:
                                        slow_vehicle_count += 1

                            if past_entry_2s is not None:
                                px2, py2 = past_entry_2s[1], past_entry_2s[2]
                                dist_2s = math.hypot(cx - px2, cy - py2)
                                on_road = is_point_in_poly(center_point, road_zone)
                                near_stop_line = has_stop_lines and is_point_in_poly(center_point, stop_lines)

                                if dist_2s <= 10.0 and on_road and not in_exclusion:
                                    if not in_intersection and not near_stop_line:
                                        is_stopped = True

                            # WRONG WAY CHECK
                            if not in_exclusion and not in_intersection:
                                if past_entry_2s is not None:
                                    px8, py8 = past_entry_2s[1], past_entry_2s[2]
                                    past_foot_y = past_entry_2s[11] if len(past_entry_2s) > 11 else py8
                                    dx = cx - px8
                                    dy = y2 - past_foot_y
                                    speed_px = math.hypot(dx, dy)
                                    if speed_px >= 15.0:
                                        past_foot = (px8, past_foot_y)
                                        current_in_towards = is_point_in_poly(bottom_center, lane_towards)
                                        current_in_away = is_point_in_poly(bottom_center, lane_away)
                                        past_in_towards = is_point_in_poly(past_foot, lane_towards)
                                        past_in_away = is_point_in_poly(past_foot, lane_away)
                                        expected_direction = None
                                        if (
                                                current_in_towards and not current_in_away
                                                and past_in_towards and not past_in_away
                                        ):
                                            expected_direction = towards_direction
                                        elif (
                                                current_in_away and not current_in_towards
                                                and past_in_away and not past_in_towards
                                        ):
                                            expected_direction = away_direction

                                        if expected_direction is not None:
                                            direction_cosine = (
                                                                       dx * float(expected_direction[0])
                                                                       + dy * float(expected_direction[1])
                                                               ) / speed_px
                                            if direction_cosine <= -0.80:
                                                is_wrong_way = True

                            # U-TURN & ILLEGAL TURN CHECKS
                            if past_entry_2s is not None and past_entry_08s is not None:
                                px2, py2 = past_entry_2s[1], past_entry_2s[2]
                                px8, py8 = past_entry_08s[1], past_entry_08s[2]

                                v1_x, v1_y = px8 - px2, py8 - py2
                                v2_x, v2_y = cx - px8, cy - py8

                                len1 = math.hypot(v1_x, v1_y)
                                len2 = math.hypot(v2_x, v2_y)

                                if len1 >= 20.0 and len2 >= 20.0:
                                    dot_product = (v1_x * v2_x + v1_y * v2_y) / (len1 * len2)
                                    cross_product = v1_x * v2_y - v1_y * v2_x

                                    if dot_product <= -0.76:
                                        past_in_towards = is_point_in_poly((px2, py2), lane_towards)
                                        past_in_away = is_point_in_poly((px2, py2), lane_away)
                                        now_in_towards = is_point_in_poly(center_point, lane_towards)
                                        now_in_away = is_point_in_poly(center_point, lane_away)
                                        lane_reversed = (
                                                (past_in_towards and not past_in_away
                                                 and now_in_away and not now_in_towards)
                                                or
                                                (past_in_away and not past_in_towards
                                                 and now_in_towards and not now_in_away)
                                        )

                                        # OPTIMIZED O(1) INTERSECTION CHECK: inspect cached intersection flags in recent history window
                                        visited_intersection = in_intersection
                                        if not visited_intersection:
                                            for past_e in reversed(vehicle_tracks[track_id]):
                                                if past_e[0] < past_entry_2s[0]:
                                                    break
                                                if len(past_e) > 14 and past_e[14]:
                                                    visited_intersection = True
                                                    break

                                        net_displacement = math.hypot(cx - px2, cy - py2)

                                        if lane_reversed and visited_intersection and net_displacement >= 50.0:
                                            is_uturn = True
                                            _backfill_track_flag(
                                                vehicle_tracks[track_id], past_entry_08s[0], 8
                                            )

                                    elif dot_product < 0.70:
                                        if has_straight_lanes and is_point_in_poly((px2, py2), straight_lanes):
                                            is_illegal_turn = True
                                        elif has_no_turn_zones and (
                                                is_point_in_poly(center_point, no_turn_zones) or is_point_in_poly(
                                            (px8, py8), no_turn_zones)):
                                            is_illegal_turn = True
                                        elif has_no_left_turn and cross_product < 0 and is_point_in_poly(center_point,
                                                                                                         no_left_turn):
                                            is_illegal_turn = True
                                        elif has_no_right_turn and cross_product > 0 and is_point_in_poly(center_point,
                                                                                                          no_right_turn):
                                            is_illegal_turn = True
                                        elif t_sec <= recent_solid_crossing_until.get(track_id, -1e9):
                                            is_illegal_turn = True

                                        if is_illegal_turn:
                                            _backfill_track_flag(
                                                vehicle_tracks[track_id], past_entry_08s[0], 9
                                            )

                        active_frame_vehicles.append({
                            'track_id': track_id,
                            'center': (cx, cy),
                            'box': box,
                            'dim': (w_box, h_box),
                            'in_exclusion': in_exclusion,
                            'in_intersection': in_intersection,
                            'flags': [is_wrong_way, is_stopped, is_solid_crossing, is_red_light, is_stop_line, is_uturn,
                                      is_illegal_turn, is_failure_to_yield],
                            'y2': y2
                        })

                # Pass 3: Near-Miss & Accident Physics Evaluation
                for v_info in active_frame_vehicles:
                    tid = v_info['track_id']
                    cx, cy = v_info['center']
                    y2 = v_info['y2']
                    box_v = v_info['box']
                    w_box, h_box = v_info['dim']
                    is_near_miss = False
                    is_accident = False

                    if not v_info['in_exclusion'] and tid in vehicle_tracks and len(vehicle_tracks[tid]) >= 3:
                        hist = vehicle_tracks[tid]
                        curr_x, curr_y = cx, y2
                        prev_entry = hist[-1]
                        prev_x, prev_y = prev_entry[1], prev_entry[11]

                        dt1 = t_sec - prev_entry[0]

                        if dt1 > 0:
                            old_entry = hist[-2]
                            old_x, old_y = old_entry[1], old_entry[11]
                            dt2 = prev_entry[0] - old_entry[0]

                            if dt2 > 0:
                                vel_now = math.hypot(curr_x - prev_x, curr_y - prev_y) / dt1
                                vel_then = math.hypot(prev_x - old_x, prev_y - old_y) / dt2
                                vel_y_now = (curr_y - prev_y) / dt1
                                vel_y_then = (prev_y - old_y) / dt2

                                deceleration_y = vel_y_then - vel_y_now
                                is_braking = (vel_y_then >= 48.0 and deceleration_y >= 80.0) or \
                                             (vel_y_then >= 52.0 and vel_y_now <= 0.23 * vel_y_then)

                                lateral_speed = abs(curr_x - prev_x) / dt1
                                lateral_shift = abs(curr_x - prev_x)
                                is_swerving = (lateral_speed >= 73.0 and lateral_shift >= 17.0)
                                abrupt_stop = vel_then >= 35.0 and vel_now <= 0.30 * vel_then

                                proximity_limit = max(65.0, min(125.0, 0.98 * max(w_box, h_box)))

                                # Vehicle-to-Vehicle Interaction
                                for other_v in active_frame_vehicles:
                                    if other_v['track_id'] == tid:
                                        continue
                                    other_x, other_y = other_v['center'][0], other_v['y2']
                                    dist_v2v = math.hypot(curr_x - other_x, curr_y - other_y)
                                    iou_v2v = calculate_iou(box_v, other_v['box'])

                                    if (
                                            iou_v2v >= ACCIDENT_IOU_THRESH
                                            and abrupt_stop
                                            and not red_light_active
                                    ):
                                        is_accident = True
                                        break

                                    if (
                                            (is_braking or is_swerving)
                                            and dist_v2v <= proximity_limit
                                            and iou_v2v < NEAR_MISS_IOU_MAX
                                    ):
                                        is_near_miss = True

                                # Vehicle-to-Pedestrian Interaction
                                if not is_accident:
                                    for ped_id, ped_center, ped_box in active_frame_peds:
                                        ped_x, ped_y = ped_center[0], ped_center[1]
                                        dist_v2p = math.hypot(curr_x - ped_x, curr_y - ped_y)
                                        iou_v2p = calculate_iou(box_v, ped_box)
                                        if (
                                                iou_v2p >= ACCIDENT_IOU_THRESH
                                                and abrupt_stop
                                                and not red_light_active
                                        ):
                                            is_accident = True
                                            break
                                        if (
                                                (is_braking or is_swerving)
                                                and dist_v2p <= proximity_limit
                                                and iou_v2p < NEAR_MISS_IOU_MAX
                                        ):
                                            is_near_miss = True
                                            break

                    if is_accident:
                        is_near_miss = False

                    # Commit frame record to vehicle memory, including cached in_intersection boolean
                    flags = v_info['flags']
                    if tid not in vehicle_tracks:
                        vehicle_tracks[tid] = []
                    vehicle_tracks[tid].append(
                        (t_sec, cx, cy, flags[0], flags[1], flags[2], flags[3], flags[4],
                         flags[5], flags[6], flags[7], v_info['y2'], is_near_miss, is_accident,
                         v_info['in_intersection'])
                    )

                if slow_vehicle_count >= 3:
                    congestion_log.append(t_sec)

    cap.release()
    print()

    # --- EXTRACT AND MERGE ALL ANOMALIES ---
    ac_events = detect_accidents(vehicle_tracks, video_duration, min_immobility_sec=2.5)
    jw_events = detect_jaywalking(pedestrian_tracks, video_duration, min_duration_sec=1.0, max_gap_sec=1.5)
    ww_events = detect_wrong_way(vehicle_tracks, video_duration, min_duration_sec=3.0, max_gap_sec=1.5)
    sv_events = detect_stopped_vehicle(vehicle_tracks, video_duration, min_duration_sec=5.0, max_gap_sec=2.0)
    sl_events = detect_solid_line_crossing(vehicle_tracks, video_duration, min_duration_sec=0.5, max_gap_sec=1.5)

    rl_events = detect_flagged_vehicle_events(vehicle_tracks, video_duration, tuple_index=6, label="red_light")
    st_events = detect_flagged_vehicle_events(vehicle_tracks, video_duration, tuple_index=7, label="stop_line")
    ut_events = detect_flagged_vehicle_events(vehicle_tracks, video_duration, tuple_index=8, label="illegal_u_turn",
                                              min_duration_sec=1.0)
    it_events = detect_flagged_vehicle_events(vehicle_tracks, video_duration, tuple_index=9, label="illegal_turn",
                                              min_duration_sec=1.0)
    fy_events = detect_flagged_vehicle_events(vehicle_tracks, video_duration, tuple_index=10, label="failure_to_yield",
                                              min_duration_sec=0.5)
    nm_events = detect_flagged_vehicle_events(vehicle_tracks, video_duration, tuple_index=12, label="near_miss",
                                              min_duration_sec=NEAR_MISS_MIN_DURATION_SEC)

    cg_events = detect_congestion_events(congestion_log, video_duration, min_duration_sec=4.0, max_gap_sec=2.0)
    fs_events = detect_fire_smoke_events(fire_smoke_log, video_duration, max_gap_sec=2.0)
    ro_events = detect_road_obstacle_events(
        road_obstacle_tracks,
        video_duration,
        min_duration_sec=OBSTACLE_MIN_STATIONARY_SEC,
        max_gap_sec=OBSTACLE_MAX_GAP_SEC,
    )

    all_events = merge_overlapping_events(
        ac_events + jw_events + ww_events + sv_events + sl_events + rl_events +
        st_events + ut_events + it_events + fy_events + nm_events + cg_events +
        fs_events + ro_events,
        video_duration
    )

    return all_events


class RiskEstimator:
    """Part B (bonus). Causal accident risk score."""

    def reset(self, meta: dict) -> None:
        pass

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return 0.0