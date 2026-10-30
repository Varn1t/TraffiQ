# 🚦 TraffiQ — AI-Powered Traffic Analysis System

A professional-grade real-time traffic analysis system built with **YOLOv8**, **ByteTrack**, and **Flask**. Detects and tracks vehicles across user-defined lanes, computes per-lane statistics, flags incidents, measures speeds, and displays everything in a live web dashboard with a purple-black gradient UI.

Featuring a premium, two-way interactive **Control Panel** allowing live parameter configuration without restarting the system.

---

## ✨ Features

| Feature                        | Description                                                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| ⚙️ Interactive Control Panel | Collapsible sidebar for live web-based parameter configuration                                                   |
| 🎯 Multi-Object Tracking       | ByteTrack algorithm with unique vehicle IDs                                                                      |
| 🛣️ Auto Lane Mapping         | Lanes mapped from vehicle trajectories (vanishing-point wedges), Y/N confirm, manual 4-click fallback            |
| 📐 Auto Speed Calibration      | Camera height/horizon from car sizes, depth scale from lane dashes — no manual pixel-to-metre                   |
| ⚡ Lane-Level Analytics        | Per-lane counts, Level of Service (LOS) A–F grading, flow rate, queue length                                    |
| 📷 Speed Camera                | Auto-calibrated ground-plane speed estimation + dynamic speeding cameras + CSV logging                           |
| ⚠️ Incident Detection        | Dynamic threshold incident tracking for stopped vehicles (>2s to 30s)                                            |
| 🚑 Ambulance Detection         | Open-vocabulary YOLO-World ("ambulance" prompt) confirms ambulances on tracked vehicles and pre-empts the signal |
| 🚦 Actuated Signal Control     | Min green, gap-out on line-crossing detectors, demand-scaled max green, yellow clearance, starvation guard       |
| ⏱️ Accurate Wait Times       | Red-lane waits predicted with the same priority rule the controller applies                                      |
| ➖ Line-Crossing Counts        | Per-lane count line (travel axis learned from motion) with N/S/E/W direction totals                              |
| 🧭 Vehicle Heading             | Per-vehicle compass heading; EMA-smoothed speeds                                                                 |
| 📊 Live Chart History          | 2-minute rolling bar/line chart (Chart.js)                                                                       |
| 🔥 Heatmap Overlay             | Accumulative density heatmap via OpenCV                                                                          |
| 📈 Trend Prediction            | Rolling linear regression per lane                                                                               |
| 💾 CSV Data Logging            | Every detection and violation exported for offline analysis                                                      |
| 🔐 Auth-Protected Dashboard    | Login-gated live dashboard with session management                                                               |

---

## ⚙️ Interactive Control Panel & Sidebar

TraffiQ transitions from a passive monitor to an active control center via its dynamic two-way **Control Panel**:

* **Collapsible Sidebar:** Open and close the control console via the hovering gear (`⚙️`) button or `Escape` key. Styled with blur backdrops and neon-purple glow themes.
* **Visualization Mode Toggles:** Switch between active processing modes (`Lanes`, `Heatmap`, `Speed`, `Timer`) dynamically. The underlying OpenCV pipeline renders the corresponding overlay instantly.
* **Dynamic Speed Trigger Limit:** Adjust speeding camera threshold (20 - 120 km/h) on the fly via slider control. Bounding boxes highlight speeding vehicles instantly.
* **Incident Timeout Configuration:** Slide the stopped vehicle detection threshold (2s - 30s) dynamically.
* **YOLO Confidence Slider:** Slide detection threshold (0.10 - 0.95) to filter out background noise or increase object detection density instantly.
* **Intelligent Cooldown Lockouts:** Features a robust 5-second interaction cooldown protection. It displays an amber-glass warning banner reading `Please wait 5 seconds before you change` when inputs are spammed, and automatically reverts sliders/modes to verified backend states on failure.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Varn1t/Traffic-Intelligence.git
cd "Traffic Intelligence"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your credentials

Copy the example env file and fill in your own values:

```bash
cp .env.example .env
```

Edit `.env`:

```
DASH_USER=your_username
DASH_PASS=your_password
```

### 4. Add your video file

Place your traffic video in the project folder and update `VIDEO_PATH` in `config.py`:

```python
VIDEO_PATH = "Traffic_Video.mp4"
```

### 5. Run the System

```bash
python main.py
```

1. **Auto lane mapping (~15 s):** TraffiQ watches the first `AUTO_LANE_FRAMES` frames, tracks vehicles, and maps each lane from where traffic actually drives (lanes are wedges meeting at the road's vanishing point, so they follow perspective).
2. **"Are the lanes correctly mapped?"** Press **Y** to continue, or **N** to draw them yourself (click the 4 corners of each lane, **U** to undo, **ENTER** when done).
3. **Speed calibration is automatic:** camera height and horizon are learned from the sizes of passing cars, and the depth scale from dashed lane markings (`DASH_CYCLE_M`, 12 m for US/Canada highways). It keeps refining while running; the HUD shows `CAL: cam …m f …px (dashes|cars|guess)`.
4. The YOLOv8 model runs in a background thread, starts local camera playback, and spins up the live dashboard.
5. Open **http://localhost:5050** in your browser, log in, and control the system dynamically.

---

## 🛠️ Tech Stack

- **Python** — core logic
- **YOLOv8** (Ultralytics) — vehicle detection
- **YOLO-World** (Ultralytics) — zero-shot ambulance detection
- **ByteTrack** (Supervision) — multi-object tracking
- **OpenCV** — video processing & heatmap overlays
- **Flask** — web server & REST API
- **Chart.js** — live dashboard charts
- **scikit-learn** — Random Forest traffic prediction
- **Vanilla CSS** — premium glassmorphism styling and animations

---

## 📁 Project Structure

```
traffiq/
├── main.py             # Main entry point & video processing loop
├── config.py           # Configuration variables and thresholds
├── state.py            # Thread-safe shared state & stats tracking
├── loggers.py          # Logging classes (CSV, SpeedCamera)
├── analytics.py        # Core logic: ML predictor, incident detection, flow rates
├── templates.py        # UI: Dashboard and landing page HTML
├── web_app.py          # Flask application logic
├── requirements.txt    # Python dependencies
├── .env.example        # Credential template (copy → .env)
├── .gitignore
└── README.md
```

> **Note:** Your `.env`, video files (`*.mp4`), model weights (`*.pt`), and CSV logs are excluded from the repo via `.gitignore`.

---

## 📄 License

All rights reserved by Varn1t

---

*Built by Varnit*
