import os
import sys
import json
import base64
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import cv2
from fast_alpr import ALPR

HOST = "127.0.0.1"
PORT = int(os.environ.get("ALPR_WORKER_PORT", 5005))

print("=== [Production Engine] Khởi tạo mô hình FastALPR AI Worker (CPU) ===")
alpr_instance = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    detector_providers=["CPUExecutionProvider"],
    ocr_model="cct-xs-v2-global-model",
    ocr_device="cpu",
    ocr_providers=["CPUExecutionProvider"],
)
print("=== [Production Engine] FastALPR ONNX Runtime CPU đã nạp vào RAM ===")

class ALPRRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "service": "FastALPR Enterprise Worker",
                "device": "cpu",
                "hardware": "CPU (CPUExecutionProvider)",
                "detector": "yolo-v9-t-384-license-plate-end2end",
                "ocr": "cct-xs-v2-global-model"
            })
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        if self.path != "/predict":
            self._send_json(404, {"error": "Not Found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json(400, {"success": False, "error": "Dữ liệu yêu cầu rỗng"})
                return

            raw_bytes = self.rfile.read(content_length)
            
            # Giải mã ảnh từ raw bytes (hỗ trợ cả binary image lẫn JSON base64)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                try:
                    payload = json.loads(raw_bytes.decode("utf-8"))
                    if "image_base64" in payload:
                        b64_data = payload["image_base64"]
                        if "," in b64_data:
                            b64_data = b64_data.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    pass

            if frame is None:
                self._send_json(400, {"success": False, "error": "Định dạng tệp ảnh không hợp lệ hoặc bị hỏng"})
                return

            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            results = alpr_instance.predict(frame)
            drawn = alpr_instance.draw_predictions(frame.copy())
            inference_ms = (time.perf_counter() - t0) * 1000

            # Encode ảnh toàn cảnh kết quả sang Base64
            success_enc, buffer = cv2.imencode(".jpg", drawn.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            annotated_base64 = ""
            if success_enc:
                annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

            plates_data = []
            for r in results:
                ocr_text = r.ocr.text if r.ocr else ""
                
                det_conf = float(r.detection.confidence)
                ocr_conf_val = 0.0
                char_confs = []
                if r.ocr and r.ocr.confidence is not None:
                    if isinstance(r.ocr.confidence, (list, tuple)):
                        char_confs = [float(c) for c in r.ocr.confidence]
                        ocr_conf_val = float(sum(char_confs) / len(char_confs)) if char_confs else 0.0
                    else:
                        ocr_conf_val = float(r.ocr.confidence)

                bbox = r.detection.bounding_box
                bw = float(bbox.width)
                bh = float(bbox.height)
                aspect_ratio = bw / bh if bh > 0 else 1.0
                plate_type = "Biển Dài (1 Dòng)" if aspect_ratio >= 2.1 else "Biển Vuông (2 Dòng)"

                # Cắt ảnh crop của biển số với padding nhẹ
                pad_x = int(bw * 0.05)
                pad_y = int(bh * 0.05)
                x1 = int(max(0, bbox.x1 - pad_x))
                y1 = int(max(0, bbox.y1 - pad_y))
                x2 = int(min(w, bbox.x2 + pad_x))
                y2 = int(min(h, bbox.y2 + pad_y))

                crop_base64 = ""
                if x2 > x1 and y2 > y1:
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size > 0:
                        suc_crop, crop_buf = cv2.imencode(".jpg", crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if suc_crop:
                            crop_base64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf).decode("utf-8")

                region = r.ocr.region if (r.ocr and r.ocr.region) else "Vietnam"
                reg_conf = float(r.ocr.region_confidence) if (r.ocr and r.ocr.region_confidence) else 0.0

                plates_data.append({
                    "text": ocr_text,
                    "detect_confidence": det_conf,
                    "ocr_confidence": ocr_conf_val,
                    "char_confidences": char_confs,
                    "region": region,
                    "region_confidence": reg_conf,
                    "plate_type": plate_type,
                    "crop_image": crop_base64,
                    "bounding_box": {
                        "x1": float(bbox.x1),
                        "y1": float(bbox.y1),
                        "x2": float(bbox.x2),
                        "y2": float(bbox.y2),
                        "width": bw,
                        "height": bh
                    }
                })

            response_data = {
                "success": True,
                "image_width": w,
                "image_height": h,
                "inference_time_ms": round(inference_ms, 2),
                "total_plates": len(plates_data),
                "plates": plates_data,
                "annotated_image": annotated_base64
            }
            self._send_json(200, response_data)

        except Exception as ex:
            self._send_json(500, {"success": False, "error": str(ex)})

    def log_message(self, format, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

def run_server():
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, ALPRRequestHandler)
    print(f"=== [Production Engine] Worker đang lắng nghe tại http://{HOST}:{PORT} ===")
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n=== Dừng Worker ===")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
