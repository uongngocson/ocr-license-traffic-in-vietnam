import os
import sys
import json
import time
import queue
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np
from video_pipeline.engine import RealtimeTrafficANPR, estimate_sharpness

STREAM_HOST = "127.0.0.1"
STREAM_PORT = int(os.environ.get("STREAM_PORT", 5006))

# Khởi tạo duy nhất một RealtimeTrafficANPR instance trong RAM
print("=== [Stream Engine] Khởi tạo mô hình RealtimeTrafficANPR duy nhất trên CPU ===")
global_anpr = RealtimeTrafficANPR(conf_thresh=0.35, target_det_width=512, detect_interval=2)
print("=== [Stream Engine] Đã nạp xong mô hình cho toàn bộ luồng Camera ===")

# Quản lý SSE client connections
event_listeners = []
event_lock = threading.Lock()

def broadcast_event(event_data):
    msg = f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode("utf-8")
    with event_lock:
        to_remove = []
        for q in event_listeners:
            try:
                q.put_nowait(msg)
            except Exception:
                to_remove.append(q)
        for q in to_remove:
            if q in event_listeners:
                event_listeners.remove(q)

class StreamRequestHandler(BaseHTTPRequestHandler):
    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "ANPR Stream Engine"}).encode("utf-8"))
            return

        if path == "/events":
            # Server-Sent Events (SSE)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            client_queue = queue.Queue(maxsize=100)
            with event_lock:
                event_listeners.append(client_queue)

            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        msg = client_queue.get(timeout=10.0)
                        self.wfile.write(msg)
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
            except Exception:
                pass
            finally:
                with event_lock:
                    if client_queue in event_listeners:
                        event_listeners.remove(client_queue)
            return

        if path == "/stream":
            source = params.get("source", ["traffic_1"])[0]
            video_map = {
                "traffic_1": "traffic_videos/vietnam_traffic_1.mp4",
                "traffic_2": "traffic_videos/vietnam_traffic_2.mp4",
                "traffic_3": "traffic_videos/vietnam_traffic_3.mp4",
            }
            video_path = video_map.get(source, "traffic_videos/vietnam_traffic_1.mp4")
            if not os.path.exists(video_path):
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            cap = cv2.VideoCapture(video_path)
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            target_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_delay = 1.0 / target_fps

            scale = global_anpr.target_det_width / orig_w
            det_w = global_anpr.target_det_width
            det_h = int(orig_h * scale)

            last_detections = []
            frame_idx = 0
            reported_tracks = set()

            try:
                while True:
                    t_start = time.perf_counter()
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        reported_tracks.clear()
                        continue

                    # 1. Detection Scheduling
                    if frame_idx % global_anpr.detect_interval == 0:
                        det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
                        det_results = global_anpr.alpr.detector.predict(det_frame)
                        detections = []
                        for d in det_results:
                            b = d.bounding_box
                            detections.append([
                                float(b.x1) / scale,
                                float(b.y1) / scale,
                                float(b.x2) / scale,
                                float(b.y2) / scale,
                                float(d.confidence)
                            ])
                        last_detections = detections
                    else:
                        detections = last_detections

                    # 2. Update Tracker
                    tracks = global_anpr.tracker.update(detections, frame_idx)

                    # 3. Best Frame & Adaptive OCR
                    for trk in tracks:
                        x1, y1, x2, y2 = [int(v) for v in trk.bbox]
                        pad_x = int((x2 - x1) * 0.05)
                        pad_y = int((y2 - y1) * 0.05)
                        cx1 = max(0, x1 - pad_x)
                        cy1 = max(0, y1 - pad_y)
                        cx2 = min(orig_w, x2 + pad_x)
                        cy2 = min(orig_h, y2 + pad_y)

                        if cx2 > cx1 and cy2 > cy1:
                            crop = frame[cy1:cy2, cx1:cx2]
                            sharp = estimate_sharpness(crop)
                            trk.evaluate_candidate(crop, sharp, trk.conf, frame_idx)

                            if trk.hit_streak >= 2 and not trk.ocr_done and trk.best_crop is not None:
                                ocr_res = global_anpr.alpr.ocr.predict(trk.best_crop)
                                if ocr_res and ocr_res.text:
                                    text = ocr_res.text.strip()
                                    conf = ocr_res.confidence
                                    conf_val = float(sum(conf)/len(conf)) if isinstance(conf, list) else float(conf)
                                    trk.ocr_result = text
                                    trk.ocr_confidence = conf_val
                                    trk.ocr_done = True

                                    if trk.track_id not in reported_tracks:
                                        reported_tracks.add(trk.track_id)
                                        import base64
                                        suc, crop_buf = cv2.imencode(".jpg", trk.best_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf).decode("utf-8") if suc else ""
                                        
                                        bw = float(x2 - x1)
                                        bh = float(y2 - y1)
                                        ar = bw / bh if bh > 0 else 1.0
                                        p_type = "Biển Dài (1 Dòng)" if ar >= 2.1 else "Biển Vuông (2 Dòng)"

                                        event_payload = {
                                            "track_id": trk.track_id,
                                            "plate": text,
                                            "confidence": conf_val,
                                            "plate_type": p_type,
                                            "crop_image": crop_b64,
                                            "region": ocr_res.region or "Vietnam",
                                            "source": source,
                                            "timestamp": time.strftime("%H:%M:%S")
                                        }
                                        broadcast_event(event_payload)

                    # 4. Vẽ HUD
                    annotated = frame.copy()
                    for trk in tracks:
                        if trk.time_since_update > 0:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in trk.bbox]
                        color = (34, 197, 94) if trk.ocr_result else (99, 102, 241)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                        label = f"ID #{trk.track_id}"
                        if trk.ocr_result:
                            label += f" | {trk.ocr_result} ({trk.ocr_confidence:.0%})"

                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
                        cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # HUD Banner
                    calc_fps = 1.0 / (time.perf_counter() - t_start) if (time.perf_counter() - t_start) > 0 else 30.0
                    hud_text = f"CCTV LIVE [CPU] | {source.upper()} | Live FPS: {calc_fps:.1f} | Active: {len(tracks)} | Identified: {len(reported_tracks)}"
                    cv2.rectangle(annotated, (0, 0), (orig_w, 48), (15, 23, 42), -1)
                    cv2.putText(annotated, hud_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (241, 245, 249), 2)
                    cv2.line(annotated, (0, 48), (orig_w, 48), (14, 165, 233), 2)

                    disp_w = min(1280, orig_w)
                    disp_h = int(orig_h * (disp_w / orig_w))
                    disp_frame = cv2.resize(annotated, (disp_w, disp_h))

                    ret_enc, jpeg = cv2.imencode(".jpg", disp_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret_enc:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()

                    frame_idx += 1
                    elapsed = time.perf_counter() - t_start
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)

            except Exception:
                # Ngắt kết nối bình thường khi client đóng tab/refresh
                pass
            finally:
                cap.release()
            return

        self.send_response(404)
        self.end_headers()

class RobustThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        try:
            super().handle_error(request, client_address)
        except Exception:
            pass

def run_stream_server():
    server_address = (STREAM_HOST, STREAM_PORT)
    httpd = RobustThreadingHTTPServer(server_address, StreamRequestHandler)
    print(f"=== [Stream Engine] ThreadingHTTPServer đang lắng nghe tại http://{STREAM_HOST}:{STREAM_PORT} ===")
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()

if __name__ == "__main__":
    run_stream_server()
