# detector.py
import os
import time
import csv
import queue
import logging
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Dict, Tuple, Any, Set
from datetime import datetime, timezone
from collections import defaultdict, deque
from dotenv import load_dotenv

import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient
import requests

# Flask imports for backend
from flask import Flask, Response, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

load_dotenv()

# --------------------- CONFIG & SETUP ---------------------

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")
CSV_FILE = os.getenv("DETECTIONS_CSV", "detections_log.csv")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))

# Gmail Configuration
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT")

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Alert Configuration
ALERT_BUFFER_SECONDS = int(os.getenv("ALERT_BUFFER_SECONDS", "60"))
ALERT_BATCH_INTERVAL = int(os.getenv("ALERT_BATCH_INTERVAL", "300"))
ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "true").lower() == "true"
ENABLE_TELEGRAM_ALERTS = os.getenv("ENABLE_TELEGRAM_ALERTS", "true").lower() == "true"

# Required PPE items
REQUIRED_PPE = {
    "Coverall", "Gloves", "Goggles", "Hairnet", "Mask", "Shoe Cover"
}

# Negative indicators
NEGATIVE_CLASSES = {"No Hairnet"}

if not ROBOFLOW_API_KEY or not MODEL_ID:
    raise RuntimeError("Missing ROBOFLOW_API_KEY or MODEL_ID in environment variables!")

client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOFLOW_API_KEY)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ppe-detector")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# --------------------- FLASK BACKEND SETUP ---------------------

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_frame = None
        self.stats = {
            'total_detections': 0,
            'violations': 0,
            'alerts_sent': 0,
            'start_time': time.time(),
            'frame_count': 0,
            'fps': 0
        }
        self.ppe_status = {item: {'detected': False, 'confidence': 0} for item in REQUIRED_PPE}
        self.alerts = deque(maxlen=50)
        self.logs = deque(maxlen=100)
        self.last_frame_time = time.time()
        
    def update_frame(self, frame):
        with self.lock:
            self.current_frame = frame.copy() if frame is not None else None
            self.stats['frame_count'] += 1
            now = time.time()
            elapsed = now - self.last_frame_time
            if elapsed > 0:
                self.stats['fps'] = round(1 / elapsed, 1)
            self.last_frame_time = now
    
    def update_detections(self, detections_list, missing_items):
        with self.lock:
            for item in self.ppe_status:
                self.ppe_status[item] = {'detected': False, 'confidence': 0}
            
            for det in detections_list:
                class_name = det['class']
                if class_name in self.ppe_status:
                    self.ppe_status[class_name] = {
                        'detected': True,
                        'confidence': round(det['confidence'] * 100, 2)
                    }
                    self.stats['total_detections'] += 1
            
            if missing_items:
                self.stats['violations'] += 1
    
    def add_alert(self, alert_type, title, message):
        with self.lock:
            alert = {
                'type': alert_type,
                'title': title,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.appendleft(alert)
            socketio.emit('new_alert', alert)
    
    def add_log(self, message, level='info'):
        with self.lock:
            log = {
                'message': message,
                'level': level,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            self.logs.appendleft(log)
            socketio.emit('new_log', log)
    
    def increment_alerts_sent(self):
        with self.lock:
            self.stats['alerts_sent'] += 1
    
    def get_current_state(self):
        with self.lock:
            return {
                'ppe_status': self.ppe_status.copy(),
                'stats': self.stats.copy(),
                'alerts': list(self.alerts)[:10],
                'logs': list(self.logs)[:50]
            }

shared_state = SharedState()

# Flask routes
@app.route('/api/status')
def get_status():
    return jsonify(shared_state.get_current_state())

def generate_frames():
    while True:
        with shared_state.lock:
            frame = shared_state.current_frame
        
        if frame is None:
            time.sleep(0.1)
            continue
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/api/video-feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    emit('initial_state', shared_state.get_current_state())

# --------------------- TELEGRAM ALERT SYSTEM ---------------------

def send_telegram_message(message: str, parse_mode: str = "HTML") -> bool:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": parse_mode}
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

def send_telegram_photo(photo_bytes: bytes, caption: str = "") -> bool:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("violation.jpg", photo_bytes, "image/jpeg")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
        response = requests.post(url, files=files, data=data, timeout=15)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram photo: {e}")
        return False

def send_telegram_alert(violations: Dict):
    if not ENABLE_TELEGRAM_ALERTS:
        return
    
    try:
        all_missing = defaultdict(int)
        for data in violations.values():
            for item in data['missing']:
                all_missing[item] += 1
        
        message = f"""🚨 <b>PPE Safety Violation Alert</b> 🚨

📊 <b>Summary</b>
• Total Violations: {len(violations)}
• Time Period: Last {ALERT_BATCH_INTERVAL // 60} minutes

⚠️ <b>Missing PPE Items:</b>
"""
        for item, count in sorted(all_missing.items(), key=lambda x: x[1], reverse=True):
            message += f"  ❌ {item}: {count} occurrence(s)\n"
        
        send_telegram_message(message)
        
        for data in violations.values():
            if data.get('frame') is not None:
                _, buffer = cv2.imencode('.jpg', data['frame'])
                send_telegram_photo(buffer.tobytes(), "⚠️ Violation detected")
                break
        
        logger.info(f"📱 Telegram alert sent")
        shared_state.add_alert('success', 'Telegram Alert Sent', f'Sent to chat {TELEGRAM_CHAT_ID}')
        shared_state.increment_alerts_sent()
        
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")

# --------------------- EMAIL ALERT SYSTEM ---------------------

class PPEAlertManager:
    def __init__(self):
        self.violation_buffer = defaultdict(list)
        self.last_alert_time = None
        self.lock = threading.Lock()
        
    def record_violation(self, missing_items: Set[str], frame_img=None):
        if not missing_items:
            return
        with self.lock:
            current_time = time.time()
            self.violation_buffer[current_time] = {'missing': list(missing_items), 'frame': frame_img}
            
    def should_send_alert(self) -> bool:
        with self.lock:
            if not self.violation_buffer:
                return False
            current_time = time.time()
            oldest_violation = min(self.violation_buffer.keys())
            if current_time - oldest_violation < ALERT_BUFFER_SECONDS:
                return False
            if self.last_alert_time and (current_time - self.last_alert_time < ALERT_BATCH_INTERVAL):
                return False
            return True
    
    def get_and_clear_violations(self) -> Dict:
        with self.lock:
            violations = dict(self.violation_buffer)
            self.violation_buffer.clear()
            self.last_alert_time = time.time()
            return violations

def send_gmail_alert(violations: Dict):
    if not ENABLE_EMAIL_ALERTS:
        return
    
    try:
        all_missing = defaultdict(int)
        for data in violations.values():
            for item in data['missing']:
                all_missing[item] += 1
        
        msg = MIMEMultipart('related')
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = ALERT_RECIPIENT
        msg['Subject'] = f"⚠️ PPE Safety Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        body = f"""<html><body>
        <h1>⚠️ PPE Safety Violation Alert</h1>
        <p><strong>Total Violations:</strong> {len(violations)}</p>
        <h2>Missing PPE Items</h2>
        <ul>"""
        
        for item, count in sorted(all_missing.items(), key=lambda x: x[1], reverse=True):
            body += f"<li>{item}: {count} occurrences</li>"
        
        body += "</ul></body></html>"
        msg.attach(MIMEText(body, 'html'))
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"✉️ Email alert sent")
        shared_state.add_alert('success', 'Email Alert Sent', f'Sent to {ALERT_RECIPIENT}')
        shared_state.increment_alerts_sent()
        
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")

# --------------------- UTILS ---------------------

def now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()

def log_detection(row: Dict[str, Any]) -> None:
    header = ["timestamp", "frame_id", "class", "confidence", "x", "y", "w", "h"]
    exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def bbox_to_xywh(b: Dict[str, float]) -> Tuple[int, int, int, int]:
    if "x" in b and "y" in b and "w" in b and "h" in b:
        return int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
    if "xmin" in b and "ymin" in b and "xmax" in b and "ymax" in b:
        x, y = int(b["xmin"]), int(b["ymin"])
        return x, y, int(b["xmax"] - b["xmin"]), int(b["ymax"] - b["ymin"])
    return int(b.get("left", 0)), int(b.get("top", 0)), int(b.get("width", 0)), int(b.get("height", 0))

def check_ppe_compliance(detected_classes: Set[str]) -> Set[str]:
    missing = REQUIRED_PPE - detected_classes
    if "No Hairnet" in detected_classes:
        missing.add("Hairnet")
    return missing

# --------------------- PROCESSING CORE ---------------------

def process_result(result: Dict[str, Any], frame_id: int, frame_img, alert_manager: PPEAlertManager):
    preds = result.get("predictions", []) if isinstance(result, dict) else []
    dets = []
    detected_classes = set()

    for p in preds:
        cname = p.get("class") or ""
        conf = float(p.get("confidence", 0))
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x, y, w, h = bbox_to_xywh(p.get("bbox", {}))
        dets.append({"class": cname, "confidence": conf, "x": x, "y": y, "w": w, "h": h, "raw": p})
        detected_classes.add(cname)
        
        log_detection({
            "timestamp": now_ts(),
            "frame_id": frame_id,
            "class": cname,
            "confidence": f"{conf:.4f}",
            "x": x, "y": y, "w": w, "h": h
        })

    missing_items = check_ppe_compliance(detected_classes)
    if missing_items:
        alert_manager.record_violation(missing_items, frame_img)
        logger.warning(f"⚠️ Frame {frame_id}: Missing PPE - {', '.join(missing_items)}")
        shared_state.add_log(f"Missing PPE: {', '.join(missing_items)}", 'warn')

    shared_state.update_detections(dets, missing_items)

    if not dets:
        return frame_img, dets, missing_items

    xys = []
    labels = []
    scores = []
    class_names = list(set(d["class"] for d in dets))
    class_to_id = {c: i for i, c in enumerate(class_names)}
    class_ids = []

    for d in dets:
        x1, y1 = d["x"], d["y"]
        x2, y2 = x1 + d["w"], y1 + d["h"]
        xys.append([x1, y1, x2, y2])
        class_ids.append(class_to_id[d["class"]])
        labels.append(d["class"])
        scores.append(d["confidence"])

    detections = sv.Detections(
        xyxy=np.array(xys),
        class_id=np.array(class_ids),
        confidence=np.array(scores)
    )

    annotated = frame_img.copy()
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections,
                                         labels=[f"{c} ({s:.2f})" for c, s in zip(labels, scores)])
    
    if missing_items:
        warning_text = f"MISSING: {', '.join(missing_items)}"
        cv2.putText(annotated, warning_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return annotated, dets, missing_items

# --------------------- THREADED PIPELINE ---------------------

def threaded_detector(local_camera_index=0, max_fps=10):
    cap = cv2.VideoCapture(local_camera_index)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    frame_q = queue.Queue(maxsize=5)
    result_q = queue.Queue(maxsize=5)
    stop_event = threading.Event()
    frame_id = 0
    
    alert_manager = PPEAlertManager()

    def cam_worker():
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                logger.warning("Camera read failed")
                continue
            if frame_q.full():
                frame_q.get_nowait()
            frame_q.put(frame)

    def infer_worker():
        while not stop_event.is_set():
            try:
                frame = frame_q.get(timeout=1)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                result = client.infer(frame, model_id=MODEL_ID)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                continue

            elapsed = time.time() - t0
            sleep_time = max(0, (1.0 / max_fps) - elapsed)
            time.sleep(sleep_time)

            if result_q.full():
                result_q.get_nowait()
            result_q.put((frame, result))
    
    def alert_worker():
        while not stop_event.is_set():
            time.sleep(10)
            if alert_manager.should_send_alert():
                violations = alert_manager.get_and_clear_violations()
                
                if ENABLE_EMAIL_ALERTS:
                    threading.Thread(target=send_gmail_alert, args=(violations,), daemon=True).start()
                
                if ENABLE_TELEGRAM_ALERTS:
                    threading.Thread(target=send_telegram_alert, args=(violations,), daemon=True).start()

    threading.Thread(target=cam_worker, daemon=True).start()
    threading.Thread(target=infer_worker, daemon=True).start()
    threading.Thread(target=alert_worker, daemon=True).start()

    logger.info("Started webcam. Press q to exit.")
    shared_state.add_log("PPE detection started", "info")
    
    logger.info(f"🌐 Dashboard: http://localhost:5000")
    logger.info(f"📹 Video feed: http://localhost:5000/api/video-feed")

    try:
        while True:
            try:
                frame, result = result_q.get(timeout=1)
            except queue.Empty:
                continue

            frame_id += 1
            annotated, _, _ = process_result(result, frame_id, frame, alert_manager)
            
            shared_state.update_frame(annotated)
            
            cv2.imshow("PPE Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Stopped detector.")

# --------------------- ENTRY POINT ---------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PPE Detection System with Web Dashboard")
    print("="*60)
    print("Starting backend server on http://localhost:5000")
    print("Open frontend.html in your browser after server starts")
    print("="*60 + "\n")
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(
        target=lambda: socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    
    # Give Flask time to start
    time.sleep(2)
    shared_state.add_log("Backend server started", "info")
    
    # Start detector
    threaded_detector(local_camera_index=0, max_fps=10)