"""
TraffiQ — Traffic Analysis System
=================================
  1. ByteTrack multi-object tracking  (via supervision)
  2. Flask live dashboard             (real-time stats in browser)
  3. Incident detection               (stopped-vehicle alerts)
  4. CSV data logging                 (analytics-ready output)
  5. Auto lane mapping                (from vehicle trajectories, Y/N confirm)
  6. Auto speed calibration           (camera geometry learned from car sizes)

Run:   python main.py
Dash:  http://localhost:5050
"""

import cv2
import numpy as np
import time
from datetime import datetime
from collections import defaultdict, deque
import threading
import webbrowser
from math import hypot

from ultralytics import YOLO
import supervision as sv

# Imports from modularized files
from config import (
    VIDEO_PATH, MODEL_NAME, CONF_THRESHOLD, TRACK_CLASSES,
    SPEED_LIMIT_KMPH, SPEEDER_LOG_FILE,
    EMERGENCY_CLASSES, EMERGENCY_SPEED_KMH, LOG_FILE, FLASK_PORT,
    AMBULANCE_BASE_MODEL, AMBULANCE_MODEL, SPEED_SMOOTHING, DETECT_IMGSZ,
    AUTO_LANE_FRAMES, DASH_CYCLE_M
)
from state import shared_state, state_lock, history_buf, session_stats
from loggers import CSVLogger, SpeedCameraLogger
from analytics import (
    IncidentDetector, LaneTrendTracker, TrafficPredictor,
    FlowRateTracker, los_grade, _SKLEARN_OK,
    LaneLineCounter, SignalController, AmbulanceDetector, AutoCalibrator,
    heading, point_in_lane
)
from lane_setup import (
    WINDOW as SETUP_WINDOW, map_lanes, lane_dividers, confirm_lanes, draw_lanes_manually,
    show_progress, lane_anchor
)
from web_app import run_flask


# ──────────────────────────────────────────────
# DETECTION + TRACKING
# ──────────────────────────────────────────────
def make_tracker(fps: float):
    # ByteTrack gets low-confidence boxes too (they only extend existing tracks; new tracks
    # still need conf_threshold) — measured ~2x more vehicles tracked across the count line.
    return sv.ByteTrack(track_activation_threshold=CONF_THRESHOLD, lost_track_buffer=60,
                        minimum_matching_threshold=0.9, frame_rate=round(fps))


def track(model, tracker, frame, conf_threshold):
    """YOLO vehicles → ByteTrack. Returns sv.Detections with tracker_id."""
    results = model(frame, conf=min(0.1, conf_threshold), imgsz=DETECT_IMGSZ, verbose=False)[0]
    tracker.track_activation_threshold = conf_threshold   # dashboard slider → tracker
    tracker.det_thresh = conf_threshold + 0.1              # (ByteTrack derives this in __init__)
    mask = np.array([model.names[int(c)] in TRACK_CLASSES for c in results.boxes.cls], dtype=bool)
    if not mask.any():
        return tracker.update_with_detections(sv.Detections.empty())
    b = results.boxes[mask]
    return tracker.update_with_detections(sv.Detections(
        xyxy=b.xyxy.cpu().numpy(), confidence=b.conf.cpu().numpy(),
        class_id=b.cls.cpu().numpy().astype(int)))


def auto_setup(cap, model, fps, calibrator):
    """Warm-up pass: track vehicles for AUTO_LANE_FRAMES frames, feed the speed
    calibrator, and map lanes from the trajectories. Returns (lanes, dirs)."""
    tracker = make_tracker(fps)
    tracks = defaultdict(list)      # tid → [(x_bottom, y_bottom, width)]
    road = []                       # sampled grey frames → traffic-free median background
    car_obs = defaultdict(list)     # tid → [(x_bottom, y_bottom, y_top, width)] cars only
    for k in range(AUTO_LANE_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        t = track(model, tracker, frame, CONF_THRESHOLD)
        for (x1, y1, x2, y2), tid, cls in zip(t.xyxy, t.tracker_id, t.class_id):
            xb = (x1 + x2) / 2
            tracks[int(tid)].append((xb, y2, x2 - x1))
            if model.names[int(cls)] == "car":
                car_obs[int(tid)].append((xb, y2, y1, x2 - x1))
        if k % 10 == 0:
            road.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            show_progress(frame, k, AUTO_LANE_FRAMES)
    for obs in car_obs.values():    # each car's overall motion tells the calibrator its orientation
        if len(obs) >= 5:
            dx, dy = obs[-1][0] - obs[0][0], obs[-1][1] - obs[0][1]
            for xb, yb, yt, w in obs:
                calibrator.add(xb, yb, yt, w, dx, dy)
    calibrator.fit()
    lanes, dirs = map_lanes(tracks)
    if road and lanes:              # dashed lane markings give the most reliable depth scale
        background = np.median(np.stack(road), axis=0).astype(np.uint8)
        calibrator.fit_focal_from_dashes(background, lane_dividers(lanes), DASH_CYCLE_M)
    print(f"[SETUP] {len(tracks)} tracks observed, {len(lanes)} lanes mapped. {calibrator.summary()}")
    return lanes, dirs


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read video. Make sure {VIDEO_PATH} exists.")
        cap.release()
        return
    first_frame = frame.copy()
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO(MODEL_NAME)
    calibrator = AutoCalibrator(frame.shape)

    # ──────────────────────────────────────────────
    # LANE SETUP: auto-map → "Are the lanes correctly mapped?" → else draw manually
    # ──────────────────────────────────────────────
    cv2.namedWindow(SETUP_WINDOW)
    lanes, lane_dirs = auto_setup(cap, model, src_fps, calibrator)
    answer = confirm_lanes(first_frame, lanes, lane_dirs) if lanes else "n"
    if not lanes:
        print("[SETUP] Could not auto-map lanes (too little moving traffic) — draw them manually.")
    if answer == "q":
        cap.release(); cv2.destroyAllWindows(); return
    if answer == "n":
        lanes, lane_dirs = draw_lanes_manually(first_frame), None
        if lanes is None:
            cap.release(); cv2.destroyAllWindows(); return
    cv2.destroyWindow(SETUP_WINDOW)
    if not lanes:  # no lanes drawn -> whole frame is one lane (timer mode would crash otherwise)
        h_, w_ = first_frame.shape[:2]
        lanes = [np.array([(0, 0), (w_ - 1, 0), (w_ - 1, h_ - 1), (0, h_ - 1)], np.int32)]
        print("No lanes drawn — using the full frame as Lane 1.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Instantiate ML predictor (one model per lane)
    predictor = TrafficPredictor(lane_ids=list(range(1, len(lanes) + 1)))
    print(f"[ML] TrafficPredictor initialised for {len(lanes)} lane(s).")
    print("     Model warms up after ~60 s of video (20 snapshots).")
    if not _SKLEARN_OK:
        print("[ML] scikit-learn missing — predictions will fall back to last count.")

    # ──────────────────────────────────────────────
    # INIT TRACKING + TOOLS
    # ──────────────────────────────────────────────
    tracker = make_tracker(src_fps)   # fresh IDs for the real run

    incident_detector = IncidentDetector()
    trend_tracker    = LaneTrendTracker()   # predictive optimisation
    logger = CSVLogger(LOG_FILE)
    speeder_logger = SpeedCameraLogger(SPEEDER_LOG_FILE)

    # Video clock: speeds/incidents use video time, not wall time, so slow
    # inference doesn't deflate speeds (e.g. 10 FPS processing of a 30 FPS clip).
    incident_ids_seen: set = set()
    line_counter = LaneLineCounter(lanes, lane_dirs)
    signal = SignalController(len(lanes))
    ambulance_detector = AmbulanceDetector(AMBULANCE_BASE_MODEL, AMBULANCE_MODEL)

    # Speed tracking (ByteTrack gives persistent IDs)
    speed_history = defaultdict(lambda: deque(maxlen=8))  # track_id → [(x_bottom, y_bottom, t), ...]
    speed_ema: dict = {}           # track_id → smoothed km/h
    heatmap = np.zeros((int(cap.get(4)), int(cap.get(3))), dtype=np.float32)
    flow_tracker  = FlowRateTracker()
    vehicle_last_lane: dict = {}   # track_id → last known lane_id (fallback for speed cam)
    # Note: history_buf imported from state.py

    mode = "lanes"
    frame_id = 0
    fps_timer = time.time()

    # ──────────────────────────────────────────────
    # START FLASK IN BACKGROUND
    # ──────────────────────────────────────────────
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"📊 Live dashboard → http://localhost:{FLASK_PORT}")
    # Give Flask a moment to start, then open the browser automatically
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{FLASK_PORT}")).start()

    # ──────────────────────────────────────────────
    # HELPER: lane for a ground point (bbox bottom-centre — where the tyres are)
    # ──────────────────────────────────────────────
    def get_lane(x, y):
        for i, poly in enumerate(lanes, 1):
            if point_in_lane(poly, x, y):
                return i
        return None

    # ──────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video — loop back to the start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break  # truly unreadable, give up
        frame_id += 1
        video_t = frame_id / src_fps

        # ── FPS ──
        now = time.time()
        fps = 1.0 / max(now - fps_timer, 1e-9)
        fps_timer = now

        # ── LOAD DYNAMIC CONTROLS FROM SHARED STATE ──
        with state_lock:
            mode = shared_state["mode"]
            speed_limit = shared_state["speed_limit"]
            incident_timeout = shared_state["incident_timeout"]
            conf_threshold = shared_state["conf_threshold"]

        # ── YOLO DETECT + BYTETRACK ──
        tracked = track(model, tracker, frame, conf_threshold)
        ambulance_ids = ambulance_detector.update(frame, frame_id, tracked)  # before annotations are drawn

        # ── PER-FRAME ACCUMULATORS ──
        lane_counts = {i+1: {"car":0,"bus":0,"truck":0,"motorbike":0} for i in range(len(lanes))}
        active_ids  = set()
        active_incidents      = []
        frame_speeders        = []
        frame_wrong_way: set  = set()
        frame_tailgating = []  # list of tailgating event dicts (populated if tailgating detection is added)
        queue_counts: dict    = {}     # lane_id → stopped-vehicle count
        emergency_lane_this_frame = None
        speeds: dict          = {}     # track_id → km/h this frame

        for i in range(len(tracked)):
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            track_id = int(tracked.tracker_id[i])
            cls_id   = int(tracked.class_id[i])
            label    = model.names[cls_id]
            if label == "motorcycle":   # COCO name → our count key
                label = "motorbike"
            cx, cy   = (x1+x2)//2, (y1+y2)//2
            gx, gy   = cx, y2                  # ground point: bbox bottom-centre (lanes + speed)
            active_ids.add(track_id)
            lane_id = in_lane = get_lane(gx, gy)
            if lane_id:
                vehicle_last_lane[track_id] = lane_id   # remember last confirmed lane
            else:
                lane_id = vehicle_last_lane.get(track_id)  # fall back to last known

            # Lane count
            if lane_id and label in lane_counts[lane_id]:
                lane_counts[lane_id][label] += 1

            # Speed: ground-point displacement → metres via the auto-calibrated camera model
            speed_history[track_id].append((gx, gy, video_t))
            speed_kmh = 0.0
            direction = ""
            dx = dy = 0.0
            moving = False
            if len(speed_history[track_id]) >= 5:   # fewer samples = bbox jitter reads as speed
                p1 = speed_history[track_id][0]
                p2 = speed_history[track_id][-1]
                dt = p2[2] - p1[2]
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                if dt > 0:
                    g = [calibrator.to_ground(x, y) for x, y, _ in speed_history[track_id]]
                    ts = [t for *_, t in speed_history[track_id]]
                    if all(g):
                        # median of per-frame steps: one bbox jump (occlusion, ID swap) can't spike it
                        raw_kmh = float(np.median([hypot(b[0]-a[0], b[1]-a[1]) / (t2 - t1)
                                                   for a, b, t1, t2 in zip(g, g[1:], ts, ts[1:])])) * 3.6
                        prev = speed_ema.get(track_id, raw_kmh)
                        if raw_kmh < 300:                  # physically impossible readings are dropped
                            speed_ema[track_id] = prev + SPEED_SMOOTHING * (raw_kmh - prev)
                        speed_kmh = speed_ema.get(track_id, 0.0)
                        moving = raw_kmh > 3               # judged on the raw reading, even if capped
                    else:                                  # not calibrated yet: judge motion in px
                        moving = hypot(dx, dy) / dt > 15
                    if moving:
                        direction = heading(dx, dy)
                if label == "car" and frame_id % 5 == 0:  # keep refining the calibration
                    calibrator.add(gx, gy, y1, x2 - x1, dx, dy)
            speeds[track_id] = speed_kmh
            if in_lane:
                line_counter.update(in_lane, track_id, gx, gy, video_t, dx, dy)
            if lane_id and not moving:
                queue_counts[lane_id] = queue_counts.get(lane_id, 0) + 1

            # ── SPEED CAMERA ──
            if speed_kmh > speed_limit:
                event = speeder_logger.log(frame_id, track_id, lane_id, speed_kmh, label)
                if event:
                    frame_speeders.append(event)
                # Police-style red alert box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,220), 3)
                cv2.rectangle(frame, (x1, y1-28), (x1+220, y1), (0,0,220), -1)
                cv2.putText(frame, f"SPEEDING  {int(speed_kmh)} km/h", (x1+4, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2)

            # ── EMERGENCY VEHICLE DETECTION ──
            if track_id in ambulance_ids:
                if lane_id:
                    emergency_lane_this_frame = lane_id
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 4)
                cv2.putText(frame, "AMBULANCE", (x1, y2+22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            elif (ambulance_detector.model is None and label in EMERGENCY_CLASSES
                    and speed_kmh > EMERGENCY_SPEED_KMH and lane_id):   # fallback heuristic
                emergency_lane_this_frame = lane_id

            # Incident detection
            is_incident = lane_id and incident_detector.update(track_id, cx, cy, lane_id, timeout=incident_timeout, now=video_t)
            if is_incident:
                if track_id not in incident_ids_seen:
                    incident_ids_seen.add(track_id)
                    session_stats.total_incidents += 1
                still_since = incident_detector.history[track_id]["still_since"]
                duration = round(video_t - still_since, 1)
                active_incidents.append({
                    "track_id": track_id, "lane": lane_id,
                    "cx": cx, "cy": cy, "duration": duration
                })
                # Red highlight
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                cv2.putText(frame, f"INCIDENT! {duration}s", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            elif speed_kmh <= speed_limit:  # don't overwrite speeding box
                # Normal annotation with speed
                color = (0,255,0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                speed_txt = f"{int(speed_kmh)}km/h {direction}" if speed_kmh > 2 else label
                cv2.putText(frame, f"ID{track_id} {speed_txt}", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

            # ── FLOW RATE recording ──
            if lane_id:
                flow_tracker.record(lane_id, track_id)



            # Heatmap update (accumulate — cv2.circle would just overwrite with 1)
            heatmap[max(cy-12, 0):cy+12, max(cx-12, 0):cx+12] += 1

            # Lane label on vehicle
            if lane_id:
                cv2.putText(frame, f"L{lane_id}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        incident_detector.cleanup(active_ids)
        if frame_id % 300 == 0:   # prune state for long-gone tracks (unbounded growth on looped video)
            for d in (speed_history, speed_ema, vehicle_last_lane, speeder_logger.logged_ids):
                for tid in [t for t in d if t not in active_ids]:
                    del d[tid]
            line_counter.prune(active_ids)
            ambulance_detector.prune(active_ids)
            calibrator.fit()      # keep adapting the speed calibration
        total_vehicles = sum(sum(c.values()) for c in lane_counts.values())

        # ── SESSION STATS + LOS + FLOW (computed once per frame, after vehicle loop) ──
        session_stats["all_ids"].update(active_ids)
        if total_vehicles > session_stats["peak_count"]:
            session_stats["peak_count"] = total_vehicles
            session_stats["peak_time"]  = datetime.now().strftime("%H:%M:%S")
        if frame_id % 90 == 0:   # history snapshot every ~3 s
            history_buf.append({
                "t": datetime.now().strftime("%H:%M:%S"),
                "lanes": {str(k): sum(v.values()) for k, v in lane_counts.items()}
            })
            # ── Feed ML predictor with this snapshot ──
            for _lid, _cnts in lane_counts.items():
                predictor.update(_lid, sum(_cnts.values()))
        lane_los_out  = {str(k): los_grade(sum(v.values()))[0] for k, v in lane_counts.items()}
        lane_flow_out = {str(k): flow_tracker.rate(k)          for k  in lane_counts}

        # ── TREND TRACKER: feed this frame's counts ──
        for lane_id, counts in lane_counts.items():
            trend_tracker.update(lane_id, sum(counts.values()))

        # ── MODES ──
        if mode == "lanes":
            for i, poly in enumerate(lanes, 1):
                total = sum(lane_counts[i].values())
                color = (0,255,0) if total < 5 else (0,255,255) if total < 15 else (0,0,255)
                cv2.polylines(frame, [poly], True, color, 2)
                cv2.putText(frame, f"Lane {i} ({total})", lane_anchor(poly, frame.shape),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                p_a, p_b = line_counter.segment(i)
                cv2.line(frame, p_a, p_b, (255,0,255), 2)   # count line
            y = 60
            for i, c in lane_counts.items():
                t = sum(c.values())
                status = "CLEAR" if t < 5 else "MODERATE" if t < 15 else "CONGESTED"
                dirs = " ".join(f"{d}:{n}" for d, n in sorted(line_counter.counts[i].items())) or "-"
                cv2.putText(frame, f"Lane {i}: {c['car']}C {c['bus']}B {c['truck']}T {c['motorbike']}M | {status} | crossed {dirs}",
                            (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                y += 22

        elif mode == "heatmap":
            blur = cv2.GaussianBlur(heatmap, (0,0), 25)
            norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.55, colored, 0.45, 0)

        elif mode == "speed":
            for i in range(len(tracked)):
                speed = speeds.get(int(tracked.tracker_id[i]), 0.0)
                x1, y1 = map(int, tracked.xyxy[i][:2])
                color = (0,255,0) if speed < 40 else (0,165,255) if speed < 80 else (0,0,255)
                cv2.putText(frame, f"{int(speed)} km/h", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        # ── SIGNAL CONTROLLER (runs every frame; drawn in timer mode) ──
        # Demand blends current count (55%) with ML-predicted count in ~15 s (45%)
        demand = {l: 0.55 * sum(c.values()) + 0.45 * predictor.predict(l) for l, c in lane_counts.items()}
        trends = {l: trend_tracker.trend(l) for l in lane_counts}
        signal_plan = signal.step(video_t, demand, trends, line_counter.last_cross, emergency_lane_this_frame)

        if mode == "timer":
            for i, poly in enumerate(lanes, 1):
                lane_total  = sum(lane_counts[i].values())
                trend_arrow = trend_tracker.label_ascii(i)
                ax, ay = lane_anchor(poly, frame.shape)
                if i == signal_plan["green"]:
                    yellow = signal_plan["phase"] == "yellow"
                    color = (0, 220, 255) if yellow else (0, 255, 0)
                    text = f"YELLOW ({signal_plan['remaining']}s)" if yellow else f"GO (~{signal_plan['remaining']}s)"
                    cv2.polylines(frame, [poly], True, color, 4)
                    cv2.putText(frame, f"{text} [{lane_total}v] {trend_arrow}",
                                (ax, ay - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    _ml_go = f"ML~{int(predictor.predict(i))}v" if predictor.is_ready() else "ML:warmup"
                    cv2.putText(frame, _ml_go, (ax, ay),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 160), 1)
                else:
                    box_color = (0, 160, 255) if lane_total < 5 else (0, 0, 220) if lane_total < 15 else (0, 0, 180)
                    nxt = " NEXT" if i == signal_plan["next"] else ""
                    cv2.polylines(frame, [poly], True, box_color, 2)
                    cv2.putText(frame, f"RED (~{signal_plan['waits'].get(i, 0)}s){nxt} [{lane_total}v] {trend_arrow}",
                                (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.putText(frame, f"Signal: last change = {signal_plan['reason']}", (20, frame.shape[0] - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # ── INCIDENT OVERLAYS ──
        for inc in active_incidents:
            cv2.putText(frame, f"[!] INCIDENT L{inc['lane']}", (20, frame.shape[0]-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # ── HUD BAR ──
        dash_url = f"http://localhost:{FLASK_PORT}"
        hud_text = f"MODE: {mode.upper()}  |  Vehicles: {total_vehicles}  |  FPS: {fps:.1f}  |  Incidents: {len(active_incidents)}  |  {calibrator.summary()}  |  Dashboard: {dash_url}"
        hud_bar = np.zeros((46, frame.shape[1], 3), dtype=np.uint8)
        cv2.putText(hud_bar, hud_text, (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        frame = np.vstack((hud_bar, frame))

        # ── UPDATE SHARED STATE ──
        with state_lock:
            shared_state["lane_counts"]       = {str(k): dict(v) for k, v in lane_counts.items()}
            shared_state["vehicle_count"]     = total_vehicles
            shared_state["incidents"]         = active_incidents
            shared_state["fps"]               = round(fps, 1)
            shared_state["frame_id"]          = frame_id
            shared_state["emergency_active"]  = emergency_lane_this_frame is not None
            shared_state["emergency_lane"]    = emergency_lane_this_frame
            shared_state["lane_trends"]       = {str(k): trend_tracker.label(k) for k in lane_counts}
            shared_state["lane_los"]          = lane_los_out
            shared_state["lane_flow"]         = lane_flow_out
            shared_state["lane_queue"]        = {str(k): queue_counts.get(k, 0) for k in lane_counts}
            shared_state["wrong_way"]         = list(frame_wrong_way)
            shared_state["tailgating"]        = list(frame_tailgating)[:5]  # cap at 5
            shared_state["lane_predictions"]  = {str(k): round(predictor.predict(k), 1) for k in lane_counts}
            shared_state["ml_ready"]          = predictor.is_ready()
            shared_state["lane_directions"]   = {str(k): dict(v) for k, v in line_counter.counts.items()}
            shared_state["signal"]            = {**signal_plan, "waits": {str(k): v for k, v in signal_plan["waits"].items()}}
            shared_state["ambulance_ids"]     = sorted(ambulance_ids)
            shared_state["ambulance_model"]   = ambulance_detector.model is not None
            # Keep last 10 speeding events
            if frame_speeders:
                shared_state["speeders"] = (shared_state["speeders"] + frame_speeders)[-10:]
                session_stats.speeders_count = speeder_logger.unique_speeders   # vehicles, not log rows

        # ── LOG EVERY 30 FRAMES ──
        if frame_id % 30 == 0:
            logger.log(frame_id, lane_counts, active_incidents)

        # ── DISPLAY ──
        cv2.imshow("Traffic Analysis  [L/H/S/T]  Q=quit", frame)
        key = cv2.waitKey(30) & 0xFF   # 30ms gives reliable key capture
        if key == ord("l"):
            with state_lock: shared_state["mode"] = "lanes"
        elif key == ord("h"):
            with state_lock: shared_state["mode"] = "heatmap"
        elif key == ord("s"):
            with state_lock: shared_state["mode"] = "speed"
        elif key == ord("t"):
            with state_lock: shared_state["mode"] = "timer"
        elif key == ord("q") or key == 27:   # Q or Esc
            break
        # Check if window was closed (WND_PROP_AUTOSIZE is reliable cross-platform)
        try:
            if cv2.getWindowProperty("Traffic Analysis  [L/H/S/T]  Q=quit",
                                      cv2.WND_PROP_AUTOSIZE) < 0:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save heatmap on exit
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("heatmap_export.png", cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET))

    # ──────────────────────────────────────────────
    # SESSION SUMMARY
    # ──────────────────────────────────────────────
    session_stats.session_ended = True   # dashboard polls this to show the summary modal
    duration_s = int((datetime.now() - datetime.fromisoformat(session_stats["session_start"])).total_seconds())
    print()
    print("═" * 56)
    print("  🚦  TRAFFIC SESSION SUMMARY")
    print("═" * 56)
    print(f"  Started         : {session_stats['session_start']}")
    print(f"  Duration        : {duration_s // 60}m {duration_s % 60}s")
    print(f"  Total vehicles  : {len(session_stats['all_ids'])} unique IDs")
    print(f"  Peak traffic    : {session_stats['peak_count']} vehicles at {session_stats['peak_time']}")
    print(f"  Incidents       : {session_stats['total_incidents']}")
    print(f"  Speeders        : {session_stats['speeders_count']}")
    print(f"  Wrong-way IDs   : {len(session_stats['wrong_way_ids'])} ({list(session_stats['wrong_way_ids'])[:8]})")
    print(f"  Tailgate events : {session_stats['tailgate_events']}")
    print(f"  Log             : {LOG_FILE}")
    print(f"  Heatmap         : heatmap_export.png")
    print("═" * 56)
    print()
    # Flask is a daemon thread — keep the process alive so the dashboard can show the summary.
    try:
        input(f"Summary is on the dashboard (http://localhost:{FLASK_PORT}). Press Enter to exit...")
    except (EOFError, KeyboardInterrupt):
        pass

if __name__ == "__main__":
    main()
