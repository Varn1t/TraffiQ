# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

VIDEO_PATH          = "Traffic_Video.mp4"
MODEL_NAME          = "yolov8n.pt"
CONF_THRESHOLD      = 0.40          # confidence needed to start a new track
DETECT_IMGSZ        = 640           # 960 tracks distant vehicles ~2x better but ~40% slower on CPU
TRACK_CLASSES       = {"car", "bus", "truck", "motorcycle"}  # COCO names
INCIDENT_TIMEOUT    = 5.0           # seconds a vehicle must stay still → incident
INCIDENT_DIST_PX    = 15            # pixels of movement tolerance
LOG_FILE            = "traffic_log.csv"
FLASK_PORT          = 5050
AUTO_LANE_FRAMES    = 300           # warm-up frames used to auto-map lanes + calibrate speed
DASH_CYCLE_M        = 12.0          # dashed lane line + gap (US/Canada highways 3 m + 9 m; UK 9, DE 18)
SPEED_LIMIT_KMPH    = 50            # vehicles above this are flagged by speed camera
SPEEDER_LOG_FILE    = "speeders_log.csv"
AMBULANCE_BASE_MODEL = "yolov8s-worldv2.pt"          # open-vocabulary detector (auto-downloaded)
AMBULANCE_MODEL      = "yolov8s-world-ambulance.pt"  # cached with the "ambulance" prompt baked in
EMERGENCY_CLASSES   = {"bus", "truck"}  # fallback only (no ambulance model): large vehicle proxy
EMERGENCY_SPEED_KMH = 40            # fallback only: fast large vehicle = likely emergency vehicle
SPEED_SMOOTHING     = 0.3           # EMA weight of the newest speed reading (lower = smoother)
