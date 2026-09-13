import os
import time
import urllib.request
import cv2
from fast_alpr import ALPR

# Danh sách hình ảnh test đa dạng từ các nguồn thực tế trên internet
TEST_IMAGES = [
    {
        "category": "European Plate (Châu Âu)",
        "name": "eu_plate_1.jpg",
        "url": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/eu/eu1.jpg",
        "description": "Biển số châu Âu xe Mercedes trắng"
    },
    {
        "category": "European Plate (Châu Âu)",
        "name": "eu_plate_2.jpg",
        "url": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/eu/eu2.jpg",
        "description": "Biển số châu Âu góc nghiêng"
    },
    {
        "category": "European Plate (Châu Âu)",
        "name": "eu_plate_3.jpg",
        "url": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/eu/eu3.jpg",
        "description": "Biển số châu Âu trên xe Audi"
    },
    {
        "category": "US Plate (Bắc Mỹ)",
        "name": "us_plate_1.jpg",
        "url": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/us/usimages/1000676.jpg",
        "description": "Biển số Mỹ dạng vuông chữ nhật nhỏ"
    },
    {
        "category": "US Plate (Bắc Mỹ)",
        "name": "us_plate_2.jpg",
        "url": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/us/usimages/1001227.jpg",
        "description": "Biển số Mỹ góc nhìn phía sau"
    },
    {
        "category": "South America Plate (Brazil/Mercosur)",
        "name": "br_plate_1.jpg",
        "url": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/br/AYO9034.jpg",
        "description": "Biển số Brazil AYO9034"
    },
    {
        "category": "South America Plate (Brazil/Mercosur)",
        "name": "br_plate_2.jpg",
        "url": "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/br/AZJ6991.jpg",
        "description": "Biển số Brazil AZJ6991"
    },
    {
        "category": "Vietnamese Plate (Việt Nam)",
        "name": "vn_plate_1.jpg",
        "url": "https://raw.githubusercontent.com/mrzaizai2k/VIETNAMESE_LICENSE_PLATE/master/data/image/1.jpg",
        "description": "Biển số xe Việt Nam thực tế #1"
    },
    {
        "category": "Vietnamese Plate (Việt Nam)",
        "name": "vn_plate_2.jpg",
        "url": "https://raw.githubusercontent.com/mrzaizai2k/VIETNAMESE_LICENSE_PLATE/master/data/image/10.jpg",
        "description": "Biển số xe Việt Nam thực tế #2"
    },
    {
        "category": "Vietnamese Plate (Việt Nam)",
        "name": "vn_plate_3.jpg",
        "url": "https://raw.githubusercontent.com/mrzaizai2k/VIETNAMESE_LICENSE_PLATE/master/data/image/12.jpg",
        "description": "Biển số xe Việt Nam thực tế #3"
    },
    {
        "category": "Dashcam Real-world (Camera hành trình)",
        "name": "dashcam_1.jpg",
        "url": "https://raw.githubusercontent.com/RobertLucian/license-plate-dataset/master/dataset/valid/images/dayride_type1_001.mp4%23t%3D1055.jpg",
        "description": "Khung hình camera hành trình xe đang di chuyển ngoài đường"
    },
    {
        "category": "Dashcam Real-world (Camera hành trình)",
        "name": "dashcam_2.jpg",
        "url": "https://raw.githubusercontent.com/RobertLucian/license-plate-dataset/master/dataset/valid/images/dayride_type1_001.mp4%23t%3D1141.jpg",
        "description": "Khung hình camera hành trình khoảng cách xa"
    }
]

DATASET_DIR = "test_dataset"
RESULTS_DIR = "test_results"

def download_image(url: str, save_path: str) -> bool:
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            with open(save_path, "wb") as f:
                f.write(content)
        return True
    except Exception as e:
        print(f"  [!] Lỗi khi tải {url}: {e}")
        return False

def format_conf(conf):
    if conf is None:
        return "N/A"
    if isinstance(conf, (list, tuple)):
        return f"{sum(conf)/len(conf):.1%}" if conf else "0.0%"
    return f"{conf:.1%}"

def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("======================================================================")
    print("      BẮT ĐẦU CHẠY BỘ TEST SUITE TOÀN DIỆN CHO FAST-ALPR (CPU)")
    print("======================================================================\n")

    # 1. Tải ảnh
    print("--- BƯỚC 1: Tải các ảnh test từ Internet ---")
    valid_test_items = []
    for item in TEST_IMAGES:
        file_path = os.path.join(DATASET_DIR, item["name"])
        print(f"-> Đang tải {item['name']} ({item['category']})...", end=" ")
        success = download_image(item["url"], file_path)
        if success:
            print("OK")
            item["file_path"] = file_path
            valid_test_items.append(item)
        else:
            print("FAILED")

    print(f"\nĐã chuẩn bị thành công {len(valid_test_items)}/{len(TEST_IMAGES)} ảnh test.\n")

    # 2. Khởi tạo FastALPR CPU
    print("--- BƯỚC 2: Khởi tạo mô hình FastALPR trên CPU ---")
    start_init = time.perf_counter()
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        detector_providers=["CPUExecutionProvider"],
        ocr_model="cct-xs-v2-global-model",
        ocr_device="cpu",
        ocr_providers=["CPUExecutionProvider"],
    )
    init_time = (time.perf_counter() - start_init) * 1000
    print(f"Khởi tạo xong trong {init_time:.1f} ms.\n")

    # 3. Chạy từng test case
    print("--- BƯỚC 3: Chạy nhận diện trên từng ảnh ---")
    test_summary = []

    for idx, item in enumerate(valid_test_items, 1):
        file_path = item["file_path"]
        print(f"\n[Testcase {idx}/{len(valid_test_items)}] {item['name']} - {item['description']}")
        
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"  [!] Không thể đọc ảnh bằng OpenCV: {file_path}")
            continue

        h, w = frame.shape[:2]
        
        # Đo thời gian inference
        t0 = time.perf_counter()
        results = alpr.predict(frame)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Vẽ bounding box và lưu ảnh kết quả
        drawn = alpr.draw_predictions(frame)
        result_img_path = os.path.join(RESULTS_DIR, f"result_{item['name']}")
        cv2.imwrite(result_img_path, drawn.image)

        num_plates = len(results)
        plate_texts = []
        for r in results:
            text = r.ocr.text if r.ocr else "<Không OCR>"
            det_c = format_conf(r.detection.confidence)
            ocr_c = format_conf(r.ocr.confidence) if r.ocr else "N/A"
            region = r.ocr.region if (r.ocr and r.ocr.region) else "Unknown"
            plate_texts.append(f"{text} (Det: {det_c}, OCR: {ocr_c}, Vùng: {region})")

        print(f"  - Kích thước ảnh: {w}x{h} px")
        print(f"  - Thời gian xử lý CPU: {latency_ms:.1f} ms")
        print(f"  - Số biển số phát hiện: {num_plates}")
        for p in plate_texts:
            print(f"    * {p}")
        print(f"  - Đã lưu ảnh kết quả tại: {result_img_path}")

        test_summary.append({
            "name": item["name"],
            "category": item["category"],
            "num_plates": num_plates,
            "plates": [r.ocr.text for r in results if r.ocr],
            "latency_ms": latency_ms,
            "result_img": result_img_path
        })

    # 4. Thống kê tổng hợp
    print("\n======================================================================")
    print("                    BẢNG TỔNG KẾT KẾT QUẢ TEST")
    print("======================================================================")
    print(f"{'STT':<4} | {'Tên ảnh':<18} | {'Nhóm':<22} | {'Biển số':<15} | {'Thời gian (CPU)':<16} | {'Trạng thái'}")
    print("-" * 90)
    
    total_latency = 0.0
    success_count = 0
    for idx, s in enumerate(test_summary, 1):
        plates_str = ", ".join(s["plates"]) if s["plates"] else "(None)"
        status = "THÀNH CÔNG" if s["num_plates"] > 0 else "KHÔNG PHÁT HIỆN"
        if s["num_plates"] > 0:
            success_count += 1
        total_latency += s["latency_ms"]
        print(f"{idx:<4} | {s['name']:<18} | {s['category'][:20]:<22} | {plates_str:<15} | {s['latency_ms']:>6.1f} ms       | {status}")

    avg_latency = total_latency / len(test_summary) if test_summary else 0.0
    print("-" * 90)
    print(f"Tổng số ảnh test: {len(test_summary)}")
    print(f"Phát hiện thành công: {success_count}/{len(test_summary)} ({success_count/len(test_summary):.1%})")
    print(f"Thời gian xử lý trung bình trên CPU: {avg_latency:.1f} ms/ảnh (~{1000/avg_latency:.1f} FPS)")
    print(f"Tất cả ảnh kết quả có vẽ biển số và bounding box đã được lưu tại: {os.path.abspath(RESULTS_DIR)}")
    print("======================================================================\n")

if __name__ == "__main__":
    main()
