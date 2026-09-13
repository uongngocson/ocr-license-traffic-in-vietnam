import os
import cv2
from fast_alpr import ALPR

def format_confidence(conf):
    if conf is None:
        return "N/A"
    if isinstance(conf, (list, tuple)):
        avg_conf = sum(conf) / len(conf) if conf else 0.0
        return f"{avg_conf:.2%} (theo từng ký tự: {[round(c, 2) for c in conf]})"
    return f"{conf:.2%}"

def main():
    print("=== Khởi tạo FastALPR trên CPU ===")
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        detector_providers=["CPUExecutionProvider"],
        ocr_model="cct-xs-v2-global-model",
        ocr_device="cpu",
        ocr_providers=["CPUExecutionProvider"],
    )

    image_path = "assets/test_image.png"
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
        return

    print(f"=== Đang nhận diện biển số từ ảnh: {image_path} ===")
    results = alpr.predict(image_path)

    print("\n=== KẾT QUẢ RAW ===")
    print(results)

    print("\n=== CHI TIẾT NHẬN DIỆN ===")
    if not results:
        print("Không tìm thấy biển số nào.")
    else:
        for idx, res in enumerate(results, 1):
            ocr_text = res.ocr.text if res.ocr else "N/A"
            ocr_conf = format_confidence(res.ocr.confidence) if res.ocr else "N/A"
            det_conf = f"{res.detection.confidence:.2%}"
            bbox = res.detection.bounding_box
            print(f"\nBiển số #{idx}:")
            print(f"  - Biển số nhận diện: {ocr_text}")
            print(f"  - Độ tin cậy OCR:     {ocr_conf}")
            print(f"  - Độ tin cậy Detect:  {det_conf}")
            print(f"  - Tọa độ BoundingBox: x1={bbox.x1:.1f}, y1={bbox.y1:.1f}, x2={bbox.x2:.1f}, y2={bbox.y2:.1f}")

    # Vẽ và lưu ảnh kết quả
    print("\n=== Đang vẽ bounding box và lưu ảnh kết quả ===")
    frame = cv2.imread(image_path)
    drawn = alpr.draw_predictions(frame)
    output_path = "output_result.png"
    cv2.imwrite(output_path, drawn.image)
    print(f"Ảnh kết quả đã được lưu tại: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
