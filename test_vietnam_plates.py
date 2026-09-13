import os
import time
import urllib.request
import csv
import cv2
from concurrent.futures import ThreadPoolExecutor
from fast_alpr import ALPR

VN_IMAGE_SOURCES = [
    # Nguồn 1: mrzaizai2k/VIETNAMESE_LICENSE_PLATE (20 ảnh xe thực tế tại VN)
    {
        "source": "mrzaizai2k",
        "base_url": "https://raw.githubusercontent.com/mrzaizai2k/VIETNAMESE_LICENSE_PLATE/master/data/image/",
        "files": [
            "1.1.PNG", "1.jpg", "10.jpg", "11.jpg", "12.jpg", "13.jpg", "14.jpg",
            "15.jpg", "16.jpg", "17.jpg", "19.jpg", "2.1.png", "2.jpg", "20.jpg",
            "21.jpg", "22.jpg", "3.jpg", "8.1.jpg", "9.2.jpg", "9.jpg"
        ],
        "type": "Xe hơi & Xe máy đường phố"
    },
    # Nguồn 2: NamYCM/NhanDienBienSoXe/img (27 ảnh xe hơi + xe máy)
    {
        "source": "NamYCM_img",
        "base_url": "https://raw.githubusercontent.com/NamYCM/NhanDienBienSoXe/master/img/",
        "files": [
            "0.jpg", "1.png", "2.png", "3.png", "4.jpg", "5.jpg", "choi.jpg", "m8.jpg",
            "testxm1.jpg", "testxm2.jpg", "xh1.jpg", "xh2.jpg", "xh3.jpg", "xh4.jpg",
            "xh5.jpg", "xh5meo.jpg", "xm1.jpg", "xm2.jpg", "xm3.jpg", "xm4.jpg",
            "xm5.jpg", "xm6.jpg", "xm6demo.jpg", "xm7.jpg", "xm7nho.jpg", "xm8.jpg", "xm9nho.jpg"
        ],
        "type": "Ô tô & Xe máy đa góc độ"
    },
    # Nguồn 3: pthang23/License_Plate_Recognition (7 ảnh)
    {
        "source": "pthang23",
        "base_url": "https://raw.githubusercontent.com/pthang23/License_Plate_Recognition/master/demo/raw/",
        "files": [
            "test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg", "test7.jpg"
        ],
        "type": "Ảnh test thực tế ô tô VN"
    },
    # Nguồn 4: quangnhat185/Plate_detect_and_recognize (3 ảnh)
    {
        "source": "quangnhat185",
        "base_url": "https://raw.githubusercontent.com/quangnhat185/Plate_detect_and_recognize/master/Plate_examples/",
        "files": [
            "vietnam_car_rectangle_plate.jpg",
            "vietnam_car_square_plate.jpg",
            "vietnam_motor_plate.jpg"
        ],
        "type": "Biển chuẩn: dài, vuông, xe máy"
    },
    # Nguồn 5: NamYCM/NhanDienBienSoXe/Bike_back (20 ảnh barrier bãi giữ xe)
    {
        "source": "NamYCM_Bike_back",
        "base_url": "https://raw.githubusercontent.com/NamYCM/NhanDienBienSoXe/master/Bike_back/",
        "files": [
            f"{i:04d}.jpg" for i in range(402, 422)
        ],
        "type": "Camera bãi giữ xe máy"
    }
]

DATASET_DIR = "vn_test_dataset"
RESULTS_DIR = "vn_test_results"
CSV_REPORT = "vn_test_report.csv"

def download_one(task):
    url, save_path = task
    if os.path.exists(save_path) and os.path.getsize(save_path) > 500:
        return True, save_path
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) < 200:
                return False, save_path
            with open(save_path, "wb") as f:
                f.write(data)
        return True, save_path
    except Exception:
        return False, save_path

def format_conf(conf):
    if conf is None:
        return "N/A"
    if isinstance(conf, (list, tuple)):
        return f"{sum(conf)/len(conf):.1%}" if conf else "0.0%"
    return f"{conf:.1%}"

def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("==========================================================================")
    print("      BỘ TEST TOÀN BỘ 100% BIỂN SỐ VIỆT NAM VỚI FAST-ALPR (CPU)")
    print("==========================================================================\n")

    # 1. Thu thập danh sách ảnh
    tasks = []
    metadata = {}
    for src in VN_IMAGE_SOURCES:
        for fname in src["files"]:
            local_name = f"{src['source']}_{fname}"
            save_path = os.path.join(DATASET_DIR, local_name)
            url = src["base_url"] + fname
            tasks.append((url, save_path))
            metadata[save_path] = {
                "source": src["source"],
                "orig_name": fname,
                "type": src["type"]
            }

    print(f"--- BƯỚC 1: Tải {len(tasks)} ảnh biển số Việt Nam (đa luồng) ---")
    downloaded_files = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(download_one, tasks)
        for ok, path in results:
            if ok:
                downloaded_files.append(path)

    print(f"Đã tải thành công: {len(downloaded_files)}/{len(tasks)} ảnh biển số Việt Nam.\n")

    # 2. Khởi tạo FastALPR trên CPU
    print("--- BƯỚC 2: Khởi tạo mô hình FastALPR (CPUExecutionProvider) ---")
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

    # 3. Chạy kiểm thử trên từng ảnh
    print("--- BƯỚC 3: Bắt đầu nhận diện biển số trên CPU ---")
    summary_rows = []
    total_latency = 0.0
    detected_images_count = 0
    total_plates_detected = 0

    csv_records = [
        ["STT", "Tên file", "Nguồn", "Phân loại", "Kích thước", "Số biển phát hiện", "Danh sách biển số", "Độ tin cậy Detect", "Độ tin cậy OCR", "Vùng", "Thời gian CPU (ms)", "Trạng thái"]
    ]

    for idx, img_path in enumerate(downloaded_files, 1):
        info = metadata[img_path]
        fname = os.path.basename(img_path)
        
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        results = alpr.predict(frame)
        latency = (time.perf_counter() - t0) * 1000
        total_latency += latency

        # Vẽ kết quả và lưu ảnh
        drawn = alpr.draw_predictions(frame)
        res_img_name = f"res_{fname}.jpg"
        res_img_path = os.path.join(RESULTS_DIR, res_img_name)
        cv2.imwrite(res_img_path, drawn.image)

        num_plates = len(results)
        if num_plates > 0:
            detected_images_count += 1
            total_plates_detected += num_plates

        plates_txt_list = []
        det_confs = []
        ocr_confs = []
        regions = []

        for r in results:
            t = r.ocr.text if r.ocr else "<No OCR>"
            plates_txt_list.append(t)
            det_confs.append(format_conf(r.detection.confidence))
            ocr_confs.append(format_conf(r.ocr.confidence) if r.ocr else "N/A")
            regions.append(r.ocr.region if (r.ocr and r.ocr.region) else "N/A")

        plates_display = ", ".join(plates_txt_list) if plates_txt_list else "(Không có)"
        det_display = ", ".join(det_confs) if det_confs else "N/A"
        ocr_display = ", ".join(ocr_confs) if ocr_confs else "N/A"
        region_display = ", ".join(set(regions)) if regions else "N/A"
        status = "THÀNH CÔNG" if num_plates > 0 else "CHƯA NHẬN DIỆN"

        print(f"[{idx:2d}/{len(downloaded_files)}] {fname:<30} | {num_plates} biển: {plates_display:<18} | {latency:5.1f}ms | {status}")

        csv_records.append([
            idx, fname, info["source"], info["type"], f"{w}x{h}", num_plates,
            plates_display, det_display, ocr_display, region_display, f"{latency:.1f}", status
        ])

    # Ghi file CSV báo cáo
    with open(CSV_REPORT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(csv_records)

    # 4. Thống kê chung
    avg_latency = total_latency / len(downloaded_files) if downloaded_files else 0.0
    det_rate = (detected_images_count / len(downloaded_files) * 100) if downloaded_files else 0.0

    print("\n==========================================================================")
    print("                         BẢNG TỔNG KẾT TOÀN DIỆN")
    print("==========================================================================")
    print(f"1. Tổng số ảnh biển số VN đã tải & test: {len(downloaded_files)} ảnh")
    print(f"2. Số ảnh phát hiện có biển số:           {detected_images_count}/{len(downloaded_files)} ({det_rate:.1f}%)")
    print(f"3. Tổng số biển số xe được phát hiện:     {total_plates_detected} biển số")
    print(f"4. Thời gian xử lý trung bình trên CPU:    {avg_latency:.1f} ms/ảnh (~{1000/avg_latency:.1f} FPS)")
    print(f"5. File CSV báo cáo chi tiết:              {os.path.abspath(CSV_REPORT)}")
    print(f"6. Thư mục lưu toàn bộ ảnh kết quả vẽ box: {os.path.abspath(RESULTS_DIR)}")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()
