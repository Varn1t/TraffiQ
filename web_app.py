import os
import secrets
import logging
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, jsonify, render_template_string, session, request, redirect, url_for
from config import FLASK_PORT
from state import shared_state, state_lock, history_buf, session_stats
from datetime import datetime
from templates import DASHBOARD_HTML, LANDING_HTML

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(24)  # empty value → random

DASH_USER = os.environ.get("DASH_USER", "admin")
DASH_PASS = os.environ.get("DASH_PASS", "changeme")
if DASH_PASS == "changeme":
    print("[WARN] Using default dashboard password — set DASH_USER/DASH_PASS in .env "
          "(the dashboard listens on all network interfaces).")

@app.route("/")
def home():
    return render_template_string(LANDING_HTML)

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    if secrets.compare_digest(username, DASH_USER) & secrets.compare_digest(password, DASH_PASS):
        session["authenticated"] = True
        return redirect(url_for("dashboard"))
    return redirect(url_for("home") + "?error=1")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if not session.get("authenticated"):
        return redirect(url_for("home") + "?error=1")
    with state_lock:
        s = dict(shared_state)
    return render_template_string(DASHBOARD_HTML, stats=s)

@app.route("/api/stats")
def api_stats():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    with state_lock:
        return jsonify(shared_state)

@app.route("/api/history")
def api_history():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(list(history_buf))

@app.route("/api/summary")
def api_summary():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    start_str = session_stats.session_start
    try:
        start_dt = datetime.fromisoformat(start_str)
        elapsed = int((datetime.now() - start_dt).total_seconds())
        hours, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        duration = f"{hours:02d}:{mins:02d}:{secs:02d}"
    except Exception:
        duration = "—"
    return jsonify({
        "duration":        duration,
        "total_vehicles":  len(session_stats.all_ids),
        "peak_count":      session_stats.peak_count,
        "peak_time":       session_stats.peak_time,
        "total_incidents": session_stats.total_incidents,
        "speeders":        session_stats.speeders_count,
        "wrong_way":       len(session_stats.wrong_way_ids),
        "tailgate_events": session_stats.tailgate_events,
        "session_start":   start_str,
        "session_ended":   session_stats.session_ended,
    })

@app.route("/api/control/update", methods=["POST"])
def update_controls():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    updates = {}
    try:
        if "mode" in data:
            if data["mode"] not in ("lanes", "heatmap", "speed", "timer"):
                raise ValueError("mode")
            updates["mode"] = data["mode"]
        if "speed_limit" in data:
            updates["speed_limit"] = min(200, max(5, int(float(data["speed_limit"]))))
        if "incident_timeout" in data:
            updates["incident_timeout"] = min(120.0, max(1.0, float(data["incident_timeout"])))
        if "conf_threshold" in data:
            updates["conf_threshold"] = min(0.95, max(0.05, float(data["conf_threshold"])))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid settings"}), 400
    with state_lock:
        shared_state.update(updates)
    return jsonify({"status": "success", "settings": updates})

def run_flask():
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)  # silence Flask request logs
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)
