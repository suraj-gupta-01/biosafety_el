# detector.py
import os
import time
import csv
import queue
import logging
import threading
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Dict, Tuple, Any, Set
from datetime import datetime, timezone
from collections import defaultdict
from dotenv import load_dotenv
from io import BytesIO

import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient

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
ALERT_BATCH_INTERVAL = int(os.getenv("ALERT_BATCH_INTERVAL", "300"))  # 5 minutes
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

# Validation
if ENABLE_EMAIL_ALERTS and (not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD or not ALERT_RECIPIENT):
    logger.warning("Email alerts enabled but Gmail credentials not configured. Email alerts will be disabled.")
    ENABLE_EMAIL_ALERTS = False

if ENABLE_TELEGRAM_ALERTS and (not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID):
    logger.warning("Telegram alerts enabled but credentials not configured. Telegram alerts will be disabled.")
    ENABLE_TELEGRAM_ALERTS = False


# --------------------- TELEGRAM ALERT SYSTEM ---------------------

def send_telegram_message(message: str, parse_mode: str = "HTML") -> bool:
    """Send a text message via Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": parse_mode
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def send_telegram_photo(photo_bytes: bytes, caption: str = "") -> bool:
    """Send a photo via Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("violation.jpg", photo_bytes, "image/jpeg")}
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML"
        }
        response = requests.post(url, files=files, data=data, timeout=15)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram photo: {e}")
        return False


def send_telegram_alert(violations: Dict):
    """Send Telegram alert for PPE violations"""
    if not ENABLE_TELEGRAM_ALERTS:
        return
    
    try:
        # Aggregate missing items
        all_missing = defaultdict(int)
        for data in violations.values():
            for item in data['missing']:
                all_missing[item] += 1
        
        # Create message
        message = f"""
🚨 <b>PPE Safety Violation Alert</b> 🚨

📊 <b>Summary</b>
• Total Violations: {len(violations)}
• Time Period: Last {ALERT_BATCH_INTERVAL // 60} minutes
• Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ <b>Missing PPE Items:</b>
"""
        
        for item, count in sorted(all_missing.items(), key=lambda x: x[1], reverse=True):
            message += f"  ❌ {item}: {count} occurrence(s)\n"
        
        message += "\n📋 <b>Required PPE Status:</b>\n"
        
        for item in sorted(REQUIRED_PPE):
            if item in all_missing:
                message += f"  ❌ {item}\n"
            else:
                message += f"  ✅ {item}\n"
        
        message += "\n⚡ <b>Action Required:</b> Ensure all personnel wear complete PPE equipment!"
        
        # Send text message
        send_telegram_message(message)
        
        # Send photo if available
        for data in violations.values():
            if data.get('frame') is not None:
                _, buffer = cv2.imencode('.jpg', data['frame'])
                photo_caption = f"⚠️ Violation captured at {datetime.now().strftime('%H:%M:%S')}"
                send_telegram_photo(buffer.tobytes(), photo_caption)
                break  # Only send one photo
        
        logger.info(f"📱 Telegram alert sent to chat {TELEGRAM_CHAT_ID}")
        
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")


# --------------------- EMAIL ALERT SYSTEM ---------------------

class PPEAlertManager:
    def __init__(self):
        self.violation_buffer = defaultdict(list)
        self.last_alert_time = None
        self.lock = threading.Lock()
        
    def record_violation(self, missing_items: Set[str], frame_img=None):
        """Record a PPE violation"""
        if not missing_items:
            return
            
        with self.lock:
            current_time = time.time()
            self.violation_buffer[current_time] = {
                'missing': list(missing_items),
                'frame': frame_img
            }
            
    def should_send_alert(self) -> bool:
        """Check if enough time has passed and violations exist"""
        with self.lock:
            if not self.violation_buffer:
                return False
                
            current_time = time.time()
            
            # Check if buffer time has passed since first violation
            oldest_violation = min(self.violation_buffer.keys())
            if current_time - oldest_violation < ALERT_BUFFER_SECONDS:
                return False
            
            # Check if batch interval has passed since last alert
            if self.last_alert_time and (current_time - self.last_alert_time < ALERT_BATCH_INTERVAL):
                return False
                
            return True
    
    def get_and_clear_violations(self) -> Dict:
        """Get accumulated violations and clear buffer"""
        with self.lock:
            violations = dict(self.violation_buffer)
            self.violation_buffer.clear()
            self.last_alert_time = time.time()
            return violations


def send_gmail_alert(violations: Dict):
    """Send email alert for PPE violations"""
    if not ENABLE_EMAIL_ALERTS:
        return
    
    try:
        # Aggregate missing items
        all_missing = defaultdict(int)
        for data in violations.values():
            for item in data['missing']:
                all_missing[item] += 1
        
        # Create email
        msg = MIMEMultipart('related')
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = ALERT_RECIPIENT
        msg['Subject'] = f"⚠️ PPE Safety Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Email body
        body = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .header {{ background-color: #ff4444; color: white; padding: 20px; }}
        .content {{ padding: 20px; }}
        .violation {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ff9800; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚠️ PPE Safety Violation Alert</h1>
    </div>
    <div class="content">
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total Violations:</strong> {len(violations)}</p>
            <p><strong>Time Period:</strong> Last {ALERT_BATCH_INTERVAL // 60} minutes</p>
        </div>
        
        <h2>Missing PPE Items</h2>
        <table>
            <tr>
                <th>PPE Item</th>
                <th>Occurrences</th>
            </tr>
"""
        
        for item, count in sorted(all_missing.items(), key=lambda x: x[1], reverse=True):
            body += f"""
            <tr>
                <td>{item}</td>
                <td>{count}</td>
            </tr>
"""
        
        body += """
        </table>
        
        <div class="violation">
            <p><strong>Action Required:</strong> Please ensure all personnel are wearing complete PPE equipment.</p>
        </div>
        
        <h3>Required PPE:</h3>
        <ul>
"""
        
        for item in sorted(REQUIRED_PPE):
            status = "❌ MISSING" if item in all_missing else "✓"
            body += f"            <li>{status} {item}</li>\n"
        
        body += """
        </ul>
        
        <p style="margin-top: 20px; font-size: 12px; color: #666;">
            This is an automated alert from the PPE Detection System.<br>
            Alert generated at: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC') + """
        </p>
    </div>
</body>
</html>
"""
        
        msg.attach(MIMEText(body, 'html'))
        
        # Attach a sample frame if available
        for data in violations.values():
            if data.get('frame') is not None:
                _, buffer = cv2.imencode('.jpg', data['frame'])
                image = MIMEImage(buffer.tobytes())
                image.add_header('Content-ID', '<violation_image>')
                msg.attach(image)
                break
        
        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"✉️ Email alert sent to {ALERT_RECIPIENT}")
        
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
    """Check which required PPE items are missing"""
    missing = REQUIRED_PPE - detected_classes
    
    # Check for negative indicators
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

    # Check PPE compliance
    missing_items = check_ppe_compliance(detected_classes)
    if missing_items:
        alert_manager.record_violation(missing_items, frame_img)
        logger.warning(f"⚠️ Frame {frame_id}: Missing PPE - {', '.join(missing_items)}")

    if not dets:
        return frame_img, dets, missing_items

    # Build Supervision detections
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
    
    # Add missing PPE warning on frame
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

    # CAMERA THREAD
    def cam_worker():
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                logger.warning("Camera read failed")
                continue
            if frame_q.full():
                frame_q.get_nowait()
            frame_q.put(frame)

    # INFERENCE THREAD
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
    
    # ALERT THREAD
    def alert_worker():
        while not stop_event.is_set():
            time.sleep(10)  # Check every 10 seconds
            if alert_manager.should_send_alert():
                violations = alert_manager.get_and_clear_violations()
                
                # Send both email and telegram alerts
                if ENABLE_EMAIL_ALERTS:
                    threading.Thread(target=send_gmail_alert, args=(violations,), daemon=True).start()
                
                if ENABLE_TELEGRAM_ALERTS:
                    threading.Thread(target=send_telegram_alert, args=(violations,), daemon=True).start()

    threading.Thread(target=cam_worker, daemon=True).start()
    threading.Thread(target=infer_worker, daemon=True).start()
    threading.Thread(target=alert_worker, daemon=True).start()

    logger.info("Started webcam. Press q to exit.")
    
    if ENABLE_EMAIL_ALERTS:
        logger.info(f"📧 Email alerts enabled → {ALERT_RECIPIENT}")
    if ENABLE_TELEGRAM_ALERTS:
        logger.info(f"📱 Telegram alerts enabled → Chat ID {TELEGRAM_CHAT_ID}")
    
    logger.info(f"⏱️ Alert buffer: {ALERT_BUFFER_SECONDS}s, Batch interval: {ALERT_BATCH_INTERVAL}s")

    try:
        while True:
            try:
                frame, result = result_q.get(timeout=1)
            except queue.Empty:
                continue

            frame_id += 1
            annotated, _, _ = process_result(result, frame_id, frame, alert_manager)
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
    threaded_detector(local_camera_index=0, max_fps=10)