import os
import time
import cv2
import numpy as np
from scipy.signal import find_peaks
from math import hypot
from collections import defaultdict, deque
from typing import TypedDict
from config import INCIDENT_TIMEOUT, INCIDENT_DIST_PX

try:
    from sklearn.ensemble import RandomForestRegressor as _RF
    _SKLEARN_OK = True
except ImportError:
    _RF = None
    _SKLEARN_OK = False
    print("[WARN] scikit-learn not found — ML prediction disabled. pip install scikit-learn")

# ──────────────────────────────────────────────
# INCIDENT DETECTOR
# ──────────────────────────────────────────────
class _IncidentEntry(TypedDict):
    pos: tuple[int, int]
    still_since: float
    lane: int

class IncidentDetector:
    """Flags vehicles that haven't moved for INCIDENT_TIMEOUT seconds."""
    def __init__(self):
        self.history: dict[int, _IncidentEntry] = {}  # track_id → entry

    def update(self, track_id: int, cx: int, cy: int, lane_id: int, timeout: float = None, now: float = None) -> bool:
        now = time.time() if now is None else now
        effective_timeout = timeout if timeout is not None else INCIDENT_TIMEOUT
        if track_id not in self.history:
            self.history[track_id] = {"pos": (cx, cy), "still_since": now, "lane": lane_id}
            return False
        prev = self.history[track_id]
        dist = hypot(cx - prev["pos"][0], cy - prev["pos"][1])
        if dist > INCIDENT_DIST_PX:
            # vehicle moved — reset timer
            self.history[track_id] = {"pos": (cx, cy), "still_since": now, "lane": lane_id}
            return False
        else:
            still_for = now - prev["still_since"]
            return still_for >= effective_timeout

    def cleanup(self, active_ids: set):
        self.history = {k: v for k, v in self.history.items() if k in active_ids}

# ──────────────────────────────────────────────
# LANE TREND TRACKER  (rule-based predictive optimisation)
# ──────────────────────────────────────────────
class LaneTrendTracker:
    """Rolling linear-regression slope per lane — purely maths, no ML.

    update() ingests one sample per frame.
    trend()  returns the slope (vehicles/sample):  +ve = rising, -ve = falling.
    label()  returns a display arrow: ↑  ↓  →
    """
    WINDOW = 20      # samples  (~10 s at typical processing speeds)
    THRESHOLD = 0.15 # slope magnitude to call a trend definite

    def __init__(self):
        self.history: dict = defaultdict(lambda: deque(maxlen=self.WINDOW))

    def update(self, lane_id: int, count: int):
        self.history[lane_id].append(count)

    def trend(self, lane_id: int) -> float:
        """Linear regression slope over the rolling window."""
        h = list(self.history[lane_id])
        n = len(h)
        if n < 3:
            return 0.0
        xs = list(range(n))
        mx, my = sum(xs) / n, sum(h) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, h))
        den = sum((x - mx) ** 2 for x in xs) or 1e-9
        return num / den

    def label(self, lane_id: int) -> str:
        """Unicode arrow — for the HTML dashboard."""
        s = self.trend(lane_id)
        if s >  self.THRESHOLD: return "\u2191"   # ↑
        if s < -self.THRESHOLD: return "\u2193"   # ↓
        return "\u2192"                           # →

    def label_ascii(self, lane_id: int) -> str:
        """ASCII arrow — for OpenCV putText which can't render Unicode."""
        s = self.trend(lane_id)
        if s >  self.THRESHOLD: return "^"
        if s < -self.THRESHOLD: return "v"
        return "-"

# ──────────────────────────────────────────────
# TRAFFIC PREDICTOR  (online Random Forest)
# ──────────────────────────────────────────────
class TrafficPredictor:
    """Per-lane Random-Forest regressor that predicts future vehicle counts.

    Every call to update() ingests the current count for a lane.
    Once MIN_SAMPLES are available a RF model is trained whose features are
    the last WINDOW observations and whose target is the count HORIZON steps
    ahead (≈ HORIZON * 3 seconds with 3-second snapshots).

    predict(lane_id) -> estimated future count (falls back to last known while warming up).
    """
    WINDOW        = 10   # past observations used as features
    HORIZON       = 5    # predict this many steps ahead  (~15 s)
    MIN_SAMPLES   = 20   # need this many before first fit
    RETRAIN_EVERY = 5    # retrain after N new samples

    def __init__(self, lane_ids: list):
        self.lane_ids = list(lane_ids)
        self.history: dict = defaultdict(lambda: deque(maxlen=200))
        self.models:  dict = {}                      # lane_id -> fitted RF
        self._since:  dict = defaultdict(int)        # samples since last retrain
        self._preds:  dict = {lid: 0.0 for lid in lane_ids}
        self._ready:  bool = False

    # ── public ──────────────────────────────────────────────
    def update(self, lane_id: int, count: int) -> None:
        self.history[lane_id].append(float(count))
        self._since[lane_id] += 1
        if (len(self.history[lane_id]) >= self.WINDOW + self.HORIZON
                and self._since[lane_id] >= self.RETRAIN_EVERY):
            self._retrain(lane_id)
            self._since[lane_id] = 0
        self._preds[lane_id] = self._infer(lane_id)

    def predict(self, lane_id: int) -> float:
        return max(0.0, self._preds.get(lane_id, 0.0))

    def is_ready(self) -> bool:
        return self._ready

    # ── private ─────────────────────────────────────────────
    def _retrain(self, lane_id: int) -> None:
        if not _SKLEARN_OK:
            return
        h = list(self.history[lane_id])
        X, y = [], []
        for i in range(len(h) - self.WINDOW - self.HORIZON + 1):
            X.append(h[i: i + self.WINDOW])
            y.append(h[i + self.WINDOW + self.HORIZON - 1])
        if len(X) < 10:
            return
        model = _RF(n_estimators=40, max_depth=5, random_state=42, n_jobs=1)
        model.fit(X, y)
        self.models[lane_id] = model
        self._ready = True

    def _infer(self, lane_id: int) -> float:
        h = list(self.history[lane_id])
        if lane_id not in self.models or len(h) < self.WINDOW:
            return float(h[-1]) if h else 0.0   # warm-up: last known
        return float(self.models[lane_id].predict([h[-self.WINDOW:]])[0])

# ──────────────────────────────────────────────
# FLOW RATE TRACKER
# ──────────────────────────────────────────────
class FlowRateTracker:
    """Counts unique vehicle IDs entering each lane per sliding 60-second window."""
    WINDOW = 60.0   # seconds

    def __init__(self):
        # lane_id → deque of (track_id, timestamp)
        self.log: dict = defaultdict(lambda: deque())

    def record(self, lane_id: int, track_id: int):
        now = time.time()
        self.log[lane_id].append((track_id, now))

    def rate(self, lane_id: int) -> float:
        """Vehicles per minute for this lane over the last 60 s."""
        now = time.time()
        cutoff = now - self.WINDOW
        buf = self.log[lane_id]
        # drop old entries
        while buf and buf[0][1] < cutoff:
            buf.popleft()
        unique_ids = len({tid for tid, _ in buf})
        return round(unique_ids / (self.WINDOW / 60), 1)   # per-minute rate

# ──────────────────────────────────────────────
# LOS GRADE  (Highway Capacity Manual simplified)
# ──────────────────────────────────────────────
def los_grade(vehicle_count: int) -> tuple:
    """Return (grade, colour_hex, description) for a lane vehicle count."""
    if vehicle_count <= 3:  return ("A", "#4ade80",  "Free flow")
    if vehicle_count <= 6:  return ("B", "#a3e635",  "Reasonable free flow")
    if vehicle_count <= 10: return ("C", "#facc15",  "Stable flow")
    if vehicle_count <= 15: return ("D", "#fb923c",  "Approaching unstable")
    if vehicle_count <= 22: return ("E", "#f87171",  "Unstable flow")
    return                         ("F", "#dc2626",  "Forced / breakdown")

def get_sklearn_ok():
    return _SKLEARN_OK

# ──────────────────────────────────────────────
# LINE-CROSSING COUNTER  (per lane, with direction)
# ──────────────────────────────────────────────
class LaneLineCounter:
    """Counts a vehicle once when it crosses its lane's count line.

    The line passes through the lane's centre, perpendicular to the lane's
    travel axis, so crossings = real throughput, and the side crossed to gives
    the compass direction (image up = N). The travel axis starts from a seed
    (auto-mapped lane direction, or the polygon's long axis) and is refined
    from how vehicles actually move (motion structure tensor), so a lane drawn
    at an odd angle still gets the right line.
    """
    PRIOR = 30.0   # the seed direction is worth this many observed moving vehicles

    def __init__(self, lanes: list, seed_dirs: list = None):
        self.polys = [np.asarray(p, float) for p in lanes]
        self.centers = [p.mean(axis=0) for p in self.polys]
        seed_dirs = seed_dirs or [None] * len(lanes)
        self.tensor, self._axis = {}, {}
        for i, (poly, d) in enumerate(zip(self.polys, seed_dirs), 1):
            d = np.asarray(d if d is not None else _long_axis(poly), float)
            d /= np.linalg.norm(d)
            self.tensor[i] = self.PRIOR * np.outer(d, d)
            self._axis[i] = d
        self.counts = {i: defaultdict(int) for i in range(1, len(lanes) + 1)}  # lane → {dir: n}
        self.last_cross: dict = {}   # lane → time of last crossing (signal gap-out detector)
        self._side: dict = {}        # (lane, track_id) → -1 / +1
        self._counted: set = set()   # (lane, track_id) already counted — stops jitter double-counts

    def axis(self, lane_id: int) -> np.ndarray:
        """Unit travel axis. Sign kept continuous with the previous value so sides never flip."""
        _, vecs = np.linalg.eigh(self.tensor[lane_id])
        a = vecs[:, -1]
        if a @ self._axis[lane_id] < 0:
            a = -a
        self._axis[lane_id] = a
        return a

    def segment(self, lane_id: int) -> tuple:
        """Count-line endpoints (clipped to the lane polygon) for drawing."""
        c, a = self.centers[lane_id - 1], self._axis[lane_id]
        q = np.array([-a[1], a[0]])
        ts = []
        poly = self.polys[lane_id - 1]
        for p0, p1 in zip(poly, np.roll(poly, -1, axis=0)):
            e = p1 - p0
            den = q[0] * -e[1] + q[1] * e[0]            # solve c + t·q = p0 + s·e
            if abs(den) < 1e-9:
                continue
            r = p0 - c
            t = (r[0] * -e[1] + r[1] * e[0]) / den
            s = (q[0] * r[1] - q[1] * r[0]) / den
            if 0 <= s <= 1:
                ts.append(t)
        lo = max([t for t in ts if t <= 0], default=-20.0)
        hi = min([t for t in ts if t >= 0], default=20.0)
        return tuple(int(v) for v in c + lo * q), tuple(int(v) for v in c + hi * q)

    def update(self, lane_id: int, track_id: int, cx: float, cy: float, t: float,
               dx: float = 0.0, dy: float = 0.0):
        """Returns the direction ("N"/"S"/"E"/"W") on a new crossing, else None.
        dx, dy = the vehicle's recent displacement (refines the lane's travel axis)."""
        m = hypot(dx, dy)
        if m > 5:
            self.tensor[lane_id] += np.outer((dx / m, dy / m), (dx / m, dy / m))
        a = self.axis(lane_id)
        side = 1 if (np.array([cx, cy]) - self.centers[lane_id - 1]) @ a >= 0 else -1
        key = (lane_id, track_id)
        prev, self._side[key] = self._side.get(key), side
        if prev is None or prev == side or key in self._counted:
            return None
        self._counted.add(key)
        d = heading(*(a * side))
        self.counts[lane_id][d] += 1
        self.last_cross[lane_id] = t
        return d

    def prune(self, active_ids: set):
        self._side = {k: v for k, v in self._side.items() if k[1] in active_ids}
        self._counted = {k for k in self._counted if k[1] in active_ids}


def _long_axis(poly: np.ndarray) -> np.ndarray:
    """Direction of the polygon's longer side pair (initial travel-axis guess)."""
    e = np.roll(poly, -1, axis=0) - poly
    return e[np.argmax(np.hypot(e[:, 0], e[:, 1]))]


def point_in_lane(poly: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

# ──────────────────────────────────────────────
# AUTO SPEED CALIBRATION  (no manual pixel-to-metre)
# ──────────────────────────────────────────────
class AutoCalibrator:
    """Learns the camera geometry from the cars themselves, continuously.

    Flat road + pinhole camera (no roll): a ground point at image row y lies
    at depth Z = f·h / (y - y_h), where y_h = horizon row, h = camera height
    (m), f = focal length (px). A car of real width W there appears
    w = W·(y - y_h)/h px wide, so over many cars:
        y_bottom = y_h + h · (w / W)          ← straight line → y_h and h
    The car's roof far edge (height CAR_H, ~ROOF_REACH m behind the near end)
    appears at y_top - y_h = f(h - CAR_H)/(Z + ROOF_REACH), which gives f.
    Then any ground pixel maps to metres:
        Z = f·h/(y - y_h),   X = (x - cx)·h/(y - y_h)
    Priors are an average passenger car; only cars are sampled. Cars moving
    sideways in the image show their length instead of their width.
    """
    CAR_W, CAR_L, CAR_H, ROOF_REACH = 1.8, 4.5, 1.5, 3.5   # metres
    MIN_SAMPLES = 150

    def __init__(self, frame_shape):
        self.img_h, self.img_w = frame_shape[:2]
        self.cx = self.img_w / 2
        self.samples = deque(maxlen=4000)   # (y_bottom, w / W_eff, y_top, w, is_depth_mover)
        self.horizon = self.cam_h = self.focal = None
        self.focal_estimated = False
        self.focal_source = None   # "dashes" (lane markings) > "cars" (roof heights) > "guess"

    @property
    def ready(self) -> bool:
        return self.focal is not None

    def add(self, x_bottom, y_bottom, y_top, w, dx, dy):
        """One observation of a moving car (bbox bottom-centre, top row, width, displacement)."""
        if abs(dy) >= 2 * abs(dx):
            w_eff, depth = self.CAR_W, True     # toward/away: we see its width
        elif abs(dx) >= 2 * abs(dy):
            w_eff, depth = self.CAR_L, False    # sideways: we see its length
        else:
            return                              # oblique: width is a mix — skip
        if abs(x_bottom - self.cx) > 0.35 * self.img_w or w < 8:
            return                              # far off-axis cars show their side too
        self.samples.append((y_bottom, w / w_eff, y_top, w, depth))

    def fit(self) -> bool:
        if len(self.samples) < self.MIN_SAMPLES:
            return False
        S = np.array(self.samples, float)
        y_b, u = S[:, 0], S[:, 1]
        keep = np.ones(len(S), bool)
        for _ in range(4):                      # robust line fit: drop > 3·MAD outliers, refit
            h, y_h = np.polyfit(u[keep], y_b[keep], 1)
            res = np.abs(y_b - (y_h + h * u))
            mad = np.median(res[keep]) + 1e-6
            keep = res < 3 * 1.4826 * mad
        if not (1.0 < h < 300) or y_h >= np.percentile(y_b, 5):
            return False                        # implausible geometry — keep the previous fit
        depth = keep & (S[:, 4] > 0)
        y_t, w = S[depth, 2], S[depth, 3]
        f = None
        if h > self.CAR_H + 0.3:
            with np.errstate(divide="ignore", invalid="ignore"):
                den = (h - self.CAR_H) / (y_t - y_h) - self.CAR_W / w
                f_i = self.ROOF_REACH / den
            f_i = f_i[(y_t > y_h + 2) & (den > 0)]
            f_i = f_i[(f_i > 0.3 * self.img_w) & (f_i < 5 * self.img_w)]
            if len(f_i) >= 20:
                f = float(np.median(f_i))
        self.horizon, self.cam_h = float(y_h), float(h)
        if self.focal_source == "dashes":   # dashes fix depth = f·h/(y-y_h): keep f·h, not f
            self.focal = self._fh_dash / self.cam_h
        else:   # lane-dash focal length beats the car-height estimate
            self.focal_estimated = f is not None
            self.focal_source = "cars" if f is not None else "guess"
            self.focal = f if f is not None else float(self.img_w)   # fallback ≈ 53° field of view
        return True

    def fit_focal_from_dashes(self, road_gray, dividers, cycle_m: float) -> bool:
        """Longitudinal scale from dashed lane markings (more reliable than car heights).

        Dashes repeat every cycle_m metres of road. Along a divider, depth is
        Z = f·u with u = h/(y - y_h), so the dash period measured in u is
        cycle_m / f. road_gray = traffic-free background (median of frames),
        dividers = [(far_point, near_point)] lines between same-direction lanes.
        """
        if self.horizon is None or not dividers:
            return False
        tophat = cv2.morphologyEx(road_gray, cv2.MORPH_TOPHAT,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))).astype(np.float32)
        y_min = self.horizon + 0.08 * self.img_h
        spans = [(max(yf, y_min), min(yn, self.img_h - 1)) for (_, yf), (_, yn) in dividers]
        y_ref = np.median([a for a, b in spans if b - a > 60] or [0])
        if not y_ref:
            return False
        du = self.cam_h / (y_ref - self.horizon) ** 2 / 4   # 1/4 of a pixel row's u-step at the far end
        acs = []
        for ((xf, yf), (xn, yn)), (y0, y1) in zip(dividers, spans):
            if y1 - y0 < 60:
                continue
            ys = np.arange(int(y0), int(y1))
            xs = xf + (ys - yf) * (xn - xf) / (yn - yf)
            half = np.maximum(3, 0.6 * (ys - self.horizon) / self.cam_h).astype(int)   # ±0.6 m sideways
            prof = np.array([tophat[y, max(0, int(x) - hw):max(1, int(x) + hw + 1)].max()
                             if 0 <= x < self.img_w else 0.0 for y, x, hw in zip(ys, xs, half)])
            u = self.cam_h / (ys - self.horizon)             # decreasing with y
            grid = np.arange(u.min(), u.max(), du)
            if len(grid) < 40:
                continue
            p = np.interp(grid, u[::-1], prof[::-1])
            k = max(3, len(p) // 4)
            p = p - np.convolve(p, np.ones(k) / k, mode="same")  # remove slow brightness trends
            ac = np.correlate(p, p, "full")[len(p) - 1:]
            if ac[0] > 0:
                acs.append(ac / ac[0])
        if not acs:
            return False
        n = min(len(a) for a in acs)
        ac = np.mean([a[:n] for a in acs], axis=0)
        peaks, props = find_peaks(ac[: n // 2], prominence=0.15, height=0.15)
        if not len(peaks) or n // peaks[0] < 3:              # need >= 3 dash cycles in view
            return False
        k = peaks[0]                                         # parabolic sub-sample peak position
        lag = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-12)
        f = cycle_m / (lag * du)
        if not (0.3 * self.img_w < f < 12 * self.img_w):
            return False
        self.focal, self.focal_estimated, self.focal_source = float(f), True, "dashes"
        self._fh_dash = self.focal * self.cam_h
        return True

    def to_ground(self, x, y):
        """Image point on the road → (X lateral, Z depth) in metres, or None near/above horizon."""
        if not self.ready or y - self.horizon < 0.08 * self.img_h:   # too near the horizon: 1 px = metres
            return None
        k = self.cam_h / (y - self.horizon)
        return (x - self.cx) * k, self.focal * k

    def summary(self) -> str:
        if not self.ready:
            return f"CAL: learning {len(self.samples)}/{self.MIN_SAMPLES}"
        return f"CAL: cam {self.cam_h:.1f}m  f {self.focal:.0f}px ({self.focal_source})"


def heading(dx: float, dy: float) -> str:
    """Compass heading of an image-space displacement (image up = N = away from camera).
    N/S unless clearly sideways (> 60° from vertical): perspective lanes run diagonally
    in the image but are still toward/away-from-camera traffic."""
    if abs(dy) >= 0.5 * hypot(dx, dy):
        return "S" if dy > 0 else "N"
    return "E" if dx > 0 else "W"

# ──────────────────────────────────────────────
# SIGNAL CONTROLLER  (actuated, demand-responsive)
# ──────────────────────────────────────────────
class SignalController:
    """One green lane at a time, like a fully-actuated traffic signal:

    * MIN_GREEN always served (safety), then the green "gaps out" once no
      vehicle has crossed the lane's count line for GAP seconds while another
      lane is waiting, or "maxes out" at a demand-scaled max green.
    * YELLOW clearance between every change.
    * Next lane = highest priority (demand + trend + time waited); any lane
      waiting MAX_WAIT seconds is forced next (starvation guard).
    * Ambulance pre-emption: a waiting lane with an ambulance ends the current
      green after EMERGENCY_MIN seconds and goes next; an ambulance in the
      green lane holds green (up to MAX_WAIT).
    * plan() predicts each red lane's wait using the same priority rule the
      controller will actually apply, rather than assuming round-robin order.
    """
    MIN_GREEN, YELLOW, GAP, HEADWAY = 8.0, 3.0, 4.0, 2.0   # seconds; HEADWAY = discharge per vehicle
    MAX_WAIT, WAIT_SCALE, EMERGENCY_MIN = 120.0, 5.0, 5.0

    def __init__(self, n_lanes: int, now: float = 0.0):
        self.n = n_lanes
        self.green = 1
        self.next_lane = None
        self.phase = "green"            # "green" | "yellow" (yellow = self.green clearing)
        self.phase_start = now
        self.max_green = None           # set from demand on the first step
        self.last_green = {1: now}
        self.last_cross: dict = {}
        self.reason = "start"

    def _max_green(self, lane, demand, trend):
        return min(90.0, max(15.0, demand.get(lane, 0) * 3 + trend.get(lane, 0) * 4))

    def _est_green(self, lane, demand, trend):
        return min(self._max_green(lane, demand, trend),
                   max(self.MIN_GREEN, demand.get(lane, 0) * self.HEADWAY))

    def _score(self, lane, now, demand, trend):
        waited = now - self.last_green.get(lane, now - self.MAX_WAIT)
        if waited >= self.MAX_WAIT:
            return float("inf")
        return demand.get(lane, 0) + trend.get(lane, 0) * 2 + waited / self.WAIT_SCALE

    def _order(self, now, demand, trend, emergency_lane):
        """Waiting lanes in the order they will get green (ambulance lane first)."""
        waiting = [l for l in range(1, self.n + 1) if l != self.green]
        return sorted(waiting, reverse=True,
                      key=lambda l: (l == emergency_lane, self._score(l, now, demand, trend)))

    def _others_waiting(self, demand):
        return any(demand.get(l, 0) > 0 for l in range(1, self.n + 1) if l != self.green)

    def step(self, now, demand, trend, last_cross, emergency_lane=None) -> dict:
        self.last_cross = last_cross
        if self.max_green is None:
            self.max_green = self._max_green(self.green, demand, trend)
        elapsed = now - self.phase_start
        if self.phase == "yellow":
            if elapsed >= self.YELLOW:
                self.green, self.phase, self.phase_start = self.next_lane, "green", now
                self.last_green[self.green] = now
                self.max_green = self._max_green(self.green, demand, trend)
        elif self.n > 1:
            end = None
            if emergency_lane and emergency_lane != self.green:
                if elapsed >= self.EMERGENCY_MIN:
                    end = f"AMBULANCE pre-empt -> Lane {emergency_lane}"
            elif emergency_lane == self.green and elapsed < self.MAX_WAIT:
                pass                                        # hold green for the ambulance
            elif elapsed >= self.max_green:
                end = "max-out"
            elif elapsed >= self.MIN_GREEN:
                idle = now - max(last_cross.get(self.green, self.phase_start), self.phase_start)
                if idle >= self.GAP and self._others_waiting(demand):
                    end = "gap-out"
            if end:
                self.next_lane = self._order(now, demand, trend, emergency_lane)[0]
                self.phase, self.phase_start, self.reason = "yellow", now, end
        return self.plan(now, demand, trend, emergency_lane)

    def _green_remaining(self, now, elapsed, demand, trend, emergency_lane):
        """Expected seconds left on the current green, mirroring step()'s end rules."""
        max_green = self.max_green or self._max_green(self.green, demand, trend)
        to_max = max(0.0, max_green - elapsed)
        if self.n < 2:
            return to_max
        if emergency_lane and emergency_lane != self.green:
            return max(0.0, self.EMERGENCY_MIN - elapsed)
        if emergency_lane == self.green or not self._others_waiting(demand):
            return to_max                        # nothing will gap it out
        idle = now - max(self.last_cross.get(self.green, self.phase_start), self.phase_start)
        if idle >= self.GAP * 0.5:               # stream has thinned — gap-out due once min green is served
            return min(to_max, max(self.MIN_GREEN - elapsed, self.GAP - idle, 0.0))
        est = self._est_green(self.green, demand, trend)
        # still flowing past the queue estimate → continuous stream, expect max-out
        return to_max if elapsed >= est else min(to_max, est - elapsed)

    def plan(self, now, demand, trend, emergency_lane=None) -> dict:
        """Current phase + predicted wait (s) until green for every other lane."""
        elapsed = now - self.phase_start
        if self.phase == "yellow":
            remaining = max(0.0, self.YELLOW - elapsed)
            t = remaining
            order = [self.next_lane] + [l for l in self._order(now, demand, trend, emergency_lane)
                                        if l != self.next_lane] + [self.green]
        else:
            remaining = self._green_remaining(now, elapsed, demand, trend, emergency_lane)
            t = remaining + self.YELLOW
            order = self._order(now, demand, trend, emergency_lane)
        waits = {}
        for l in order:
            waits[l] = int(round(t))
            t += self._est_green(l, demand, trend) + self.YELLOW
        return {"green": self.green, "phase": self.phase, "remaining": int(round(remaining)),
                "waits": waits, "next": order[0] if order else self.green, "reason": self.reason}

# ──────────────────────────────────────────────
# AMBULANCE DETECTOR  (open-vocabulary YOLO-World)
# ──────────────────────────────────────────────
class AmbulanceDetector:
    """COCO (yolov8n) has no ambulance class, so every EVERY_N frames an
    open-vocabulary YOLO-World model prompted with "ambulance" runs on the
    frame; its boxes are matched to tracked vehicles and a track is confirmed
    after HITS matches (filters one-off false positives).

    The prompt is baked into a cached .pt on first run, so CLIP (auto-installed
    by ultralytics, ~340 MB) is only needed once.
    """
    EVERY_N, HITS, CONF = 10, 2, 0.25   # CONF tuned: real ambulances scored 0.29-0.51, normal traffic <= 0.21

    def __init__(self, base_weights: str, cache_path: str):
        self.model = None
        self.hits: dict = defaultdict(int)
        self.confirmed: set = set()
        try:
            from ultralytics import YOLOWorld
            if os.path.exists(cache_path):
                self.model = YOLOWorld(cache_path)
            else:
                print("[AMB] Building ambulance model (first run downloads weights + CLIP)...")
                self.model = YOLOWorld(base_weights)
                self.model.set_classes(["ambulance"])
                self.model.save(cache_path)
        except Exception as e:
            print(f"[WARN] Ambulance model unavailable ({e}) — using fast bus/truck heuristic instead.")

    def update(self, frame, frame_id: int, tracked) -> set:
        """Returns the set of confirmed ambulance track IDs."""
        if self.model is None or frame_id % self.EVERY_N or len(tracked) == 0:
            return self.confirmed
        r = self.model(frame, conf=self.CONF, verbose=False)[0]
        for bx1, by1, bx2, by2 in r.boxes.xyxy.cpu().numpy():
            ex, ey = (bx1 + bx2) / 2, (by1 + by2) / 2
            for (x1, y1, x2, y2), tid in zip(tracked.xyxy, tracked.tracker_id):
                if x1 <= ex <= x2 and y1 <= ey <= y2:
                    self.hits[int(tid)] += 1
                    if self.hits[int(tid)] >= self.HITS:
                        self.confirmed.add(int(tid))
                    break
        return self.confirmed

    def prune(self, active_ids: set):
        self.hits = defaultdict(int, {k: v for k, v in self.hits.items() if k in active_ids})
        self.confirmed &= active_ids
