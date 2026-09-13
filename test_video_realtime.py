import os
import json
import time
from video_pipeline.engine import RealtimeTrafficANPR

def main():
    print("==========================================================================")
    print("   BẮT ĐẦU CHẠY FULL TESTKEY REAL-TIME CAMERA GIAO THÔNG (100% CPU)       ")
    print("==========================================================================\n")

    engine = RealtimeTrafficANPR(conf_thresh=0.35, target_det_width=640)

    test_videos = [
        {
            "id": "traffic_test_1",
            "name": "vietnam_traffic_1.mp4",
            "path": "traffic_videos/vietnam_traffic_1.mp4",
            "output": "traffic_videos/result_traffic_1_annotated.mp4",
            "desc": "Camera 1: Nút giao ngã tư thực tế HD (1280x720) 30 FPS",
            "max_frames": 150
        },
        {
            "id": "traffic_test_2",
            "name": "vietnam_traffic_2.mp4",
            "path": "traffic_videos/vietnam_traffic_2.mp4",
            "output": "traffic_videos/result_traffic_2_annotated.mp4",
            "desc": "Camera 2: Trục lộ giao thông đô thị Full-HD (1920x1080) 30 FPS",
            "max_frames": 150
        }
    ]

    all_reports = []

    for idx, test in enumerate(test_videos, 1):
        print(f"\n>>>>>>>> [TESTKEY #{idx}/{len(test_videos)}] {test['name']} - {test['desc']} <<<<<<<<")
        max_f = test.get("max_frames", None)
        res = engine.process_video(
            video_path=test["path"],
            output_video_path=test["output"],
            max_frames=max_f,
            display_progress=True
        )
        res["test_id"] = test["id"]
        res["description"] = test["desc"]
        all_reports.append(res)

    # Xuất báo cáo tổng hợp
    report_file = "traffic_video_test_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    print("\n==========================================================================")
    print("                  TỔNG HỢP KẾT QUẢ TESTKEY REAL-TIME CAMERA               ")
    print("==========================================================================")
    total_plates_found = 0
    for r in all_reports:
        print(f"* Video: {os.path.basename(r['video_path'])}")
        print(f"  - Tốc độ xử lý CPU: {r['average_fps']} FPS")
        print(f"  - Độ trễ mỗi frame: {r['average_frame_latency_ms']} ms")
        print(f"  - Số biển số phát hiện: {r['total_unique_plates_identified']} biển")
        total_plates_found += r['total_unique_plates_identified']
        print(f"  - Video kết quả: {r['output_video_path']}")
    print(f"\nĐã lưu toàn bộ báo cáo phân tích tại: {os.path.abspath(report_file)}")
    print(f"Tổng số biển số phát hiện qua các camera: {total_plates_found}")
    print("==========================================================================\n")

    for r in all_reports:
        assert r['total_unique_plates_identified'] > 0, f"Lỗi: Không nhận diện được biển số nào trong {r['video_path']}"
        assert r['total_frames_processed'] > 0, f"Lỗi: Khung hình không được xử lý trong {r['video_path']}"

    print(">>> 100% TẤT CẢ CAMERA GIAO THÔNG VÀ BIỂN SỐ ĐÃ VƯỢT QUA TESTKEY THÀNH CÔNG! <<<")

if __name__ == "__main__":
    main()

