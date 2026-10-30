"""
Lane setup: automatic lane mapping from vehicle trajectories, a Y/N
confirmation screen, and a manual fallback (click 4 corners per lane).

How auto-mapping works
----------------------
On a straight road seen in perspective, every lane is a wedge whose edges
meet at the road's vanishing point (VP). So:
  1. Track vehicles for a few seconds; fit a line to each trajectory
     (bbox bottom-centre = where the tyres touch the road).
  2. RANSAC the VP the trajectories converge on (up to 3 roads, e.g. an
     intersection; cross traffic that doesn't fit is left out).
  3. Project every trajectory point along its ray from the VP onto one
     reference line across the road -> a 1-D "lateral position" s.
  4. Split by travel direction, histogram s, and each peak is a lane
     centre; the valleys between peaks are the lane boundaries.
  5. Each lane polygon = the wedge between its two boundary rays, clipped
     to where vehicles were actually seen.
Side-on roads (parallel trajectories, VP at infinity) fall out of the same
maths with parallel rays.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

MIN_TRACK_PTS = 8        # points a trajectory needs to be used
MIN_TRACK_DISP = 40      # px it must move
MIN_LANE_TRACKS = 3      # tracks needed to call a peak a lane
VP_ANGLE_DEG = 4.0       # trajectory must point at the VP within this angle

LANE_COLORS = [(255, 128, 0), (0, 200, 255), (255, 0, 200), (0, 255, 128),
               (128, 128, 255), (255, 255, 0), (0, 128, 255), (200, 0, 255)]


# ──────────────────────────────────────────────
# AUTO MAPPING  (pure: trajectories in, lanes out)
# ──────────────────────────────────────────────
def _line(c, d):
    """Homogeneous line through point c with direction d (unit normal form)."""
    l = np.cross([c[0], c[1], 1.0], [c[0] + d[0], c[1] + d[1], 1.0])
    return l / (np.hypot(l[0], l[1]) + 1e-12)


def _dir_to(c, vp):
    """Unit image direction from point c toward homogeneous point vp (works at infinity)."""
    v = np.array([vp[0] - vp[2] * c[0], vp[1] - vp[2] * c[1]])
    return v / (np.linalg.norm(v) + 1e-12)


def _ransac_vp(segs, iters=400, rng=np.random.default_rng(0)):
    """Vanishing point most trajectories point at. Returns (vp homogeneous, inlier mask)."""
    lines = np.array([_line(s["c"], s["d"]) for s in segs])
    weights = np.array([s["len"] for s in segs])
    C = np.array([s["c"] for s in segs])
    D = np.array([s["d"] for s in segs])
    p = weights / weights.sum()
    sin_tol = np.sin(np.radians(VP_ANGLE_DEG))

    def inliers(vp):
        v = np.stack([vp[0] - vp[2] * C[:, 0], vp[1] - vp[2] * C[:, 1]], axis=1)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return np.abs(D[:, 0] * v[:, 1] - D[:, 1] * v[:, 0]) < sin_tol   # |sin(angle)| to the VP

    best, best_score = None, -1.0
    for _ in range(iters):
        i, j = rng.choice(len(segs), 2, replace=False, p=p)
        vp = np.cross(lines[i], lines[j])
        if not np.any(vp):
            continue
        score = weights[inliers(vp)].sum()
        if score > best_score:
            best, best_score = vp, score
    mask = inliers(best)
    # refine: least-squares point closest to all inlier lines (SVD handles VP at infinity)
    _, _, vt = np.linalg.svd(lines[mask] * weights[mask, None])
    vp = vt[-1]
    if vp[2] < 0:          # homogeneous sign is arbitrary; w > 0 keeps "toward the VP" pointing at it
        vp = -vp
    return vp, inliers(vp)


def _ray_point(vp, b, u, c, a):
    """Point on the ray from vp through b whose along-road coordinate (p-c)·u equals a."""
    if abs(vp[2]) > 1e-9:
        v = vp[:2] / vp[2]
        den = (b - v) @ u
        if abs(den) > 1e-9:
            return v + (a - (v - c) @ u) / den * (b - v)
    return b + (a - (b - c) @ u) * u      # parallel rays


def _road_lanes(road, vp):
    """Lanes of one road (trajectories sharing a VP)."""
    pts = np.vstack([s["P"] for s in road])                  # x, y, w
    wts = np.concatenate([np.full(len(s["P"]), 1.0 / len(s["P"])) for s in road])
    c = np.average(pts[:, :2], axis=0, weights=wts)          # reference point: data centroid
    u = _dir_to(c, vp)                                       # along road, toward VP
    n = np.array([-u[1], u[0]])                              # across road
    finite = abs(vp[2]) > 1e-9 and np.linalg.norm(vp[:2] / vp[2] - c) < 1e5
    v = vp[:2] / vp[2] if finite else None

    lanes = []
    for sign in (1, -1):                                     # away from camera / toward camera
        group = [s for s in road if np.sign(s["d"] @ _dir_to(s["c"], vp)) == sign]
        if len(group) < MIN_LANE_TRACKS:
            continue
        s_all, w_all, a_all, wt_all, tid_all = [], [], [], [], []
        for k, s in enumerate(group):
            P = s["P"][:, :2]
            if finite:   # slide each point along its VP ray onto the reference line
                t = ((c - v) @ u) / ((P - v) @ u)
                q = v + t[:, None] * (P - v)
            else:
                t = np.ones(len(P))
                q = P - ((P - c) @ u)[:, None] * u
            keep = (t > 0) & (t < 4)                         # skip points far beyond / behind reference
            s_all.append(((q - c) @ n)[keep]); w_all.append((s["P"][:, 2] * t)[keep])
            a_all.append(((P - c) @ u)[keep]); wt_all.append(np.full(keep.sum(), 1.0 / len(P)))
            tid_all.append(np.full(keep.sum(), k))
        s_all, w_all, a_all, wt_all, tid_all = map(np.concatenate, (s_all, w_all, a_all, wt_all, tid_all))
        if len(s_all) < MIN_TRACK_PTS:
            continue

        car_w = float(np.median(w_all))                      # vehicle width at the reference line
        bin_px = max(1.0, car_w / 12)
        edges = np.arange(s_all.min() - car_w, s_all.max() + car_w + bin_px, bin_px)
        hist, _ = np.histogram(s_all, bins=edges, weights=wt_all)
        smooth = gaussian_filter1d(hist, sigma=max(1.0, 0.18 * car_w / bin_px))
        peaks, _ = find_peaks(smooth, distance=max(1, int(1.2 * car_w / bin_px)),
                              prominence=0.08 * smooth.max())
        if not len(peaks):
            continue
        centers = edges[peaks] + bin_px / 2
        # boundaries: valleys between neighbouring peaks, half a lane beyond the outer ones
        bounds = [edges[peaks[i] + np.argmin(smooth[peaks[i]:peaks[i + 1] + 1])] + bin_px / 2
                  for i in range(len(peaks) - 1)]
        half = np.median(np.diff(centers)) / 2 if len(centers) > 1 else car_w
        bounds = [centers[0] - half] + bounds + [centers[-1] + half]

        for lo, hi in zip(bounds[:-1], bounds[1:]):
            in_lane = (s_all >= lo) & (s_all < hi)
            if len(np.unique(tid_all[in_lane])) < MIN_LANE_TRACKS:
                continue
            a_near, a_far = np.percentile(a_all[in_lane], [1, 97])
            bl, br = c + lo * n, c + hi * n
            poly = np.array([_ray_point(vp, bl, u, c, a_far), _ray_point(vp, br, u, c, a_far),
                             _ray_point(vp, br, u, c, a_near), _ray_point(vp, bl, u, c, a_near)])
            mid = poly.mean(axis=0)
            lanes.append((poly, sign * _dir_to(mid, vp)))
    return lanes


def map_lanes(tracks: dict, max_roads: int = 3) -> tuple:
    """tracks: {track_id: [(x_bottom, y_bottom, bbox_width), ...]}.
    Returns (polygons [np.int32 (4,2)], travel directions [unit (dx, dy)]), left→right."""
    segs = []
    for pts in tracks.values():
        if len(pts) < MIN_TRACK_PTS:
            continue
        P = np.asarray(pts, float)
        disp = P[-1, :2] - P[0, :2]
        if np.hypot(*disp) < MIN_TRACK_DISP:
            continue
        vx, vy, _, _ = cv2.fitLine(P[:, :2].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        d = np.array([vx, vy]) * (1 if np.array([vx, vy]) @ disp >= 0 else -1)
        segs.append({"P": P, "d": d, "c": P[:, :2].mean(axis=0), "len": float(np.hypot(*disp))})

    lanes, remaining = [], segs
    for _ in range(max_roads):
        if len(remaining) < 2 * MIN_LANE_TRACKS:
            break
        vp, inl = _ransac_vp(remaining)
        if inl.sum() < max(2 * MIN_LANE_TRACKS, 0.15 * len(segs)):
            break
        lanes += _road_lanes([s for s, i in zip(remaining, inl) if i], vp)
        remaining = [s for s, i in zip(remaining, inl) if not i]

    lanes.sort(key=lambda l: l[0][2:, 0].mean())            # left → right by near edge
    return [np.round(p).astype(np.int32) for p, _ in lanes], [tuple(d) for _, d in lanes]


def lane_dividers(polys) -> list:
    """Edges shared by neighbouring auto-mapped lanes (where dashed markings are).
    Polygons are [L_far, R_far, R_near, L_near]; returns [(far_pt, near_pt)]."""
    out = []
    for a, b in zip(polys, polys[1:]):
        p, q = a[1].astype(float), a[2].astype(float)              # a's right edge
        d = (q - p) / (np.linalg.norm(q - p) + 1e-9)
        off = [abs(d[0] * (v[1] - p[1]) - d[1] * (v[0] - p[0])) for v in (b[0], b[3])]
        if max(off) <= 3:                                          # b's left edge on the same line
            ends = sorted([tuple(p), tuple(q), tuple(b[0].astype(float)), tuple(b[3].astype(float))],
                          key=lambda v: v[1])
            out.append((ends[0], ends[-1]))                        # far-most → near-most
    return out


# ──────────────────────────────────────────────
# DRAWING HELPERS
# ──────────────────────────────────────────────
def lane_anchor(poly, frame_shape) -> tuple:
    """Text position near the lane's lowest (nearest) edge, kept on screen."""
    low = poly[np.argsort(poly[:, 1])[-2:]]
    x = int(np.clip(low[:, 0].min() + 6, 5, frame_shape[1] - 260))
    y = int(np.clip(low[:, 1].mean() - 12, 40, frame_shape[0] - 10))
    return x, y


def draw_lane_overlay(img, lanes, dirs=None):
    over = img.copy()
    for i, poly in enumerate(lanes):
        cv2.fillPoly(over, [poly], LANE_COLORS[i % len(LANE_COLORS)])
    out = cv2.addWeighted(over, 0.28, img, 0.72, 0)
    for i, poly in enumerate(lanes):
        col = LANE_COLORS[i % len(LANE_COLORS)]
        cv2.polylines(out, [poly], True, col, 2)
        mid = poly.mean(axis=0).astype(int)
        cv2.putText(out, f"Lane {i + 1}", lane_anchor(poly, img.shape),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if dirs and dirs[i] is not None:
            tip = (mid + 45 * np.asarray(dirs[i])).astype(int)
            cv2.arrowedLine(out, tuple(mid), tuple(tip), (255, 255, 255), 3, tipLength=0.35)
    return out


def _banner(img, lines):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 16 + 34 * len(lines)), (0, 0, 0), -1)
    for k, (text, col) in enumerate(lines):
        cv2.putText(out, text, (16, 38 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
    return out


# ──────────────────────────────────────────────
# SETUP UI
# ──────────────────────────────────────────────
WINDOW = "TraffiQ Setup"


def show_progress(frame, done, total):
    pct = int(100 * done / max(total, 1))
    cv2.imshow(WINDOW, _banner(frame, [(f"Auto-mapping lanes from traffic... {pct}%", (0, 255, 255))]))
    cv2.waitKey(1)


def confirm_lanes(frame, lanes, dirs) -> str:
    """Show the auto-mapped lanes. Returns 'y', 'n' or 'q'."""
    print(f"Auto-mapped {len(lanes)} lane(s). Are the lanes correctly mapped? [Y]es / [N]o (draw manually) / [Q]uit")
    img = _banner(draw_lane_overlay(frame, lanes, dirs), [
        (f"Auto-mapped {len(lanes)} lane(s). Are the lanes correctly mapped?", (255, 255, 255)),
        ("[Y] Yes, continue     [N] No, draw manually     [Q] Quit", (0, 255, 255))])
    while True:
        cv2.imshow(WINDOW, img)
        k = cv2.waitKey(50) & 0xFF
        if k in (ord("y"), ord("Y"), 13):
            return "y"
        if k in (ord("n"), ord("N")):
            return "n"
        if k in (ord("q"), ord("Q"), 27):
            return "q"


def draw_lanes_manually(frame):
    """Click 4 corners per lane (any order around the lane). U = undo, ENTER = done, Q = quit.
    Returns a list of polygons, or None if the user quit."""
    lanes, pts = [], []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            if len(pts) == 4:
                lanes.append(np.array(pts, np.int32))
                print(f"Lane {len(lanes)} set: {pts}")
                pts.clear()

    cv2.setMouseCallback(WINDOW, on_mouse)
    print("Click the 4 corners of each lane. [U] undo  [ENTER] done  [Q] quit")
    while True:
        img = draw_lane_overlay(frame, lanes)
        for p in pts:
            cv2.circle(img, p, 6, (0, 255, 255), -1)
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 255, 255), 2)
        img = _banner(img, [(f"Click 4 corners per lane ({len(lanes)} drawn)", (255, 255, 255)),
                            ("[U] Undo   [ENTER] Done   [Q] Quit", (0, 255, 255))])
        cv2.imshow(WINDOW, img)
        k = cv2.waitKey(30) & 0xFF
        if k == 13:
            break
        if k in (ord("u"), ord("U")):
            if pts:
                pts.clear()
            elif lanes:
                lanes.pop()
        if k in (ord("q"), ord("Q"), 27):
            return None
    cv2.setMouseCallback(WINDOW, lambda *a: None)
    return lanes
