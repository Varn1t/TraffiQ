"""Smoke checks for the pure-logic parts. Run: python test_traffiq_checks.py"""
from analytics import IncidentDetector, LaneLineCounter, SignalController, heading
from web_app import app
from state import shared_state

# Incident detector runs on the supplied (video) clock, not wall time
d = IncidentDetector()
assert not d.update(1, 100, 100, 1, timeout=5, now=0.0)
assert not d.update(1, 102, 101, 1, timeout=5, now=4.9)
assert d.update(1, 101, 100, 1, timeout=5, now=5.0)
assert not d.update(1, 200, 100, 1, timeout=5, now=6.0)   # moved → timer resets

# Control endpoint validates and clamps input
c = app.test_client()
with c.session_transaction() as s:
    s["authenticated"] = True
assert c.post("/api/control/update", json={"mode": "bogus"}).status_code == 400
assert c.post("/api/control/update", json={"speed_limit": "fast"}).status_code == 400
assert c.post("/api/control/update", json={"conf_threshold": 5, "speed_limit": "60.5"}).status_code == 200
assert shared_state["conf_threshold"] == 0.95 and shared_state["speed_limit"] == 60
assert app.test_client().post("/api/control/update", json={}).status_code == 401
assert c.get("/dashboard").status_code == 200 and b"Crossed" in c.get("/dashboard").data   # template still renders

import numpy as np
def rect(x1, y1, x2, y2):
    return np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], np.int32)

# Line counter: tall lane → horizontal mid-line at y=100; counts once per track, with direction
lc = LaneLineCounter([rect(0, 0, 50, 200), rect(100, 0, 400, 50)])
assert lc.update(1, 7, 25, 90, 1.0) is None          # first sighting, above line
assert lc.update(1, 7, 25, 110, 1.1) == "S"          # crossed downward
assert lc.update(1, 7, 25, 95, 1.2) is None          # jitter back — not recounted
assert lc.update(2, 8, 300, 25, 2.0) is None and lc.update(2, 8, 200, 25, 2.1) == "W"   # wide lane → vertical line
assert dict(lc.counts[1]) == {"S": 1} and lc.last_cross == {1: 1.1, 2: 2.1}
assert set(lc.segment(1)) == {(0, 100), (50, 100)}         # count line clipped to the lane
assert heading(5, -1) == "E" and heading(0, -5) == "N"
# wide box but traffic moves vertically → axis is learned from motion, line becomes horizontal
lc = LaneLineCounter([rect(0, 500, 700, 1000)])
assert abs(lc.axis(1)[0]) > 0.99                      # box-shape guess: travel along x
for i in range(100):
    lc.update(1, 100 + i, 50, 520, 0.0, dx=3, dy=60)  # vehicles moving down
assert abs(lc.axis(1)[1]) > 0.99
assert lc.update(1, 1, 300, 700, 1.0, 2, 40) is None and lc.update(1, 1, 310, 800, 1.1, 2, 40) == "S"
# seeded direction (auto-mapped lane): slanted wedge counts along the seed straight away
lc = LaneLineCounter([np.array([(400, 500), (500, 500), (700, 1000), (500, 1000)], np.int32)], [(0.37, 0.93)])
assert lc.update(1, 1, 510, 700, 0.0) is None and lc.update(1, 1, 540, 790, 0.1) == "S"
assert lc.update(1, 2, 200, 740, 1.2, 1, 40) is None and lc.update(1, 2, 400, 740, 1.3, 1, 40) is None  # sideways drift ≠ crossing

# Signal controller
D = {1: 5, 2: 5, 3: 5}; T = {}
sc = SignalController(3)
assert sc.step(5.0, D, T, {})["green"] == 1           # min green holds
p = sc.step(8.0, D, T, {})                            # no crossings for >= GAP after min green → gap-out
assert p["phase"] == "yellow" and sc.reason == "gap-out"
p = sc.step(11.0, D, T, {})                           # yellow done → next lane green
assert p["phase"] == "green" and p["green"] in (2, 3)
g = p["green"]; other = ({2, 3} - {g}).pop()
assert list(p["waits"]) == [other, 1]                 # lane 1 just served → waits longest
assert p["waits"][other] < p["waits"][1]
# vehicles keep crossing → green extends until max-out
t = 11.0
while sc.phase == "green" and t < 200:
    t += 1; sc.step(t, D, T, {g: t})
assert sc.reason == "max-out" and t - 11.0 >= 15
# continuous stream past the queue estimate → remaining predicts max-out, not 0
sc = SignalController(2)
for t in range(0, 20):
    p = sc.step(float(t), {1: 8, 2: 8}, T, {1: float(t)})   # max green 24 s, queue estimate 16 s
assert p["phase"] == "green" and p["remaining"] == round(sc.max_green - 19)
assert p["waits"][2] == p["remaining"] + sc.YELLOW
# ambulance in a waiting lane pre-empts after EMERGENCY_MIN
sc = SignalController(3)
sc.step(1.0, D, T, {1: 1.0}, emergency_lane=3)
assert sc.phase == "green"
sc.step(5.0, D, T, {1: 5.0}, emergency_lane=3)
assert sc.phase == "yellow" and sc.next_lane == 3 and sc.reason.startswith("AMBULANCE")
# ambulance in the green lane holds green even with no crossings
sc = SignalController(2)
for t in range(1, 60):
    sc.step(float(t), D, T, {}, emergency_lane=1)
assert sc.phase == "green" and sc.green == 1
# single lane never switches
sc = SignalController(1)
assert sc.step(500.0, {1: 3}, T, {})["green"] == 1 and sc.phase == "green"

# ── Auto speed calibration recovers a known synthetic camera ──
import numpy as np
from analytics import AutoCalibrator, point_in_lane
from lane_setup import map_lanes
F, CAM_H, HOR = 1500.0, 9.0, 400.0
cal = AutoCalibrator((1080, 1920))
rng = np.random.default_rng(1)
for Z in np.linspace(12, 80, 300):
    X = rng.choice([-5.4, -1.8, 1.8, 5.4])
    Zf = Z * rng.normal(1, 0.01)                        # 1 % size noise
    cal.add(960 + F * X / Z, HOR + F * CAM_H / Z, HOR + F * (CAM_H - cal.CAR_H) / (Zf + cal.ROOF_REACH),
            F * cal.CAR_W / Zf, 1.0, 40.0)
assert cal.fit() and cal.focal_estimated
assert abs(cal.cam_h - CAM_H) < 0.5 and abs(cal.horizon - HOR) < 10 and abs(cal.focal - F) / F < 0.08
(_, z1), (_, z2) = cal.to_ground(960, HOR + F * CAM_H / 20), cal.to_ground(960, HOR + F * CAM_H / 30)
assert abs((z2 - z1) - 10) < 1.0                        # 10 m of road measures ~10 m

# ── Lane auto-mapping finds lanes that converge on a vanishing point ──
VP = np.array([960.0, 300.0])
bottoms = {300: 1, 620: 1, 940: 1, 1260: -1, 1580: -1}   # lane bottom x → +1 toward camera, -1 away
tracks, tid = {}, 0
for bx, sgn in bottoms.items():
    for _ in range(10):
        ys = np.linspace(520, 1000, 25)[::sgn]
        off = rng.normal(0, 25)
        pts = []
        for y in ys:
            t = (y - VP[1]) / (1060 - VP[1])
            x = VP[0] + t * (bx + off - VP[0]) + rng.normal(0, 3)
            pts.append((x, y + rng.normal(0, 3), 260 * t))
        tracks[tid] = pts; tid += 1
tracks[tid] = [(100 + 30 * k, 450, 60) for k in range(20)]  # one cross-traffic track: ignored
polys, dirs = map_lanes(tracks)
assert len(polys) == 5, len(polys)
for poly, (bx, sgn), d in zip(polys, bottoms.items(), dirs):
    t = (850 - VP[1]) / (1060 - VP[1])
    xs = {b: VP[0] + t * (b - VP[0]) for b in bottoms}
    assert point_in_lane(poly, xs[bx], 850)                            # own lane centre inside
    assert not any(point_in_lane(poly, x, 850) for b, x in xs.items() if b != bx)
    assert np.sign(d[1]) == sgn                                         # travel direction

# ── Dashed lane markings fix the focal length (depth scale) ──
import cv2
road = np.full((1080, 1920), 70, np.uint8)
for k in range(40):                                     # 3 m dash every 12 m, drawn by the true camera
    z0, z1 = 15 + 12 * k, 18 + 12 * k
    ya, yb = HOR + F * CAM_H / z1, HOR + F * CAM_H / z0
    if yb >= 1080 or ya < HOR + 5:
        continue
    half = max(1, int(0.08 * (yb - HOR) / CAM_H))
    cv2.rectangle(road, (960 - half, int(ya)), (960 + half, int(min(yb, 1079))), 230, -1)
cal.focal, cal.focal_source = 999.0, "cars"             # pretend the car-height estimate was bad
assert cal.fit_focal_from_dashes(road, [((960.0, 450.0), (960.0, 1079.0))], 12.0)
assert cal.focal_source == "dashes" and abs(cal.focal - F) / F < 0.05, cal.focal
cal.fit()                                               # a later refit keeps the dash scale
assert cal.focal_source == "dashes" and abs(cal.focal - F) / F < 0.08

print("all checks passed")
