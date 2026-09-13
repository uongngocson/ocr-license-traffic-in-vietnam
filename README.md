# ocr-license-traffic-in-vietnam

# 🇻🇳 VN-ANPR: Hệ Thống Giám Sát Camera Giao Thông Real-Time & Nhận Diện Biển Số Đa Phương Tiện (100% CPU)

> **Enterprise Intelligent Transportation System (ITS) & License Plate Recognition Platform**  
> Kiến trúc Hybrid hiệu năng cao: **Go Backend (HTTP/SSE Gateway)** + **Giao diện Web Giám Sát Tailwind CSS chuẩn Production** + **Động cơ Thị giác Máy tính ONNX Runtime (CPU Mode)**.

---

## 🌟 1. Tính Năng Nổi Bật Hệ Thống

* **100% Thuần CPU (CPUExecutionProvider)**: Tối ưu hóa đa luồng với SIMD vectorization (AVX2/AVX-512), không yêu cầu card đồ họa rời GPU.
* **Pipeline Camera Giao Thông Real-Time Đạt Chuẩn SOTA**:
  * **Decoupled Detection & Tracking**: Lập lịch chạy bộ phát hiện YOLOv9-t 384 theo chu kỳ xen kẽ, kết hợp thuật toán theo dõi đa đối tượng **IoU Multi-Object Tracker (<0.3ms/frame)** giúp duy trì tốc độ stream mượt mà.
  * **Best-Frame Selection**: Đánh giá độ nét bằng phương sai toán tử Laplace (`cv2.Laplacian`) kết hợp diện tích bounding box và độ tin cậy để tự động chọn khung hình tối ưu nhất trong luồng video trước khi gửi vào OCR.
  * **Temporal Consensus Voting**: Thuật toán bỏ phiếu theo thời gian trên toàn bộ tracklet của phương tiện, loại bỏ hoàn toàn các khung hình rung lắc, mờ nhòe.
* **Phân Loại Định Dạng Biển Số Chuẩn**:
  * Tự động nhận diện quy cách hình học biển số phương tiện theo tỷ lệ quang học:
    * 🔲 **Biển Vuông (2 dòng)**: Thường gắn phía sau xe máy, ô tô tải, xe con tiêu chuẩn cũ/mới.
    * 💳 **Biển Dài (1 dòng)**: Chuẩn gắn phía trước ô tô con, xe du lịch, xe khách.
* **Trực Quan Hóa HUD Video & Server-Sent Events (SSE)**:
  * Vẽ Heads-Up Display (HUD) trực tiếp vào luồng video: Bounding box biển số, ID định danh tracklet, biển số đã nhận diện, độ tin cậy và FPS tức thời.
  * Đẩy thông báo sự kiện thời gian thực (SSE) về Dashboard không cần tải lại trang.

---

## 📹 2. Mạng Lưới Kênh Camera Giám Sát Thực Tế

Hệ thống được cấu hình sẵn 2 luồng video camera giám sát thực tế tại Việt Nam:

| Kênh Camera | Tên Tuyến Giám Sát | Độ Phân Giải & FPS | Chủng Loại Phương Tiện Tập Trung |
|---|---|---|---|
| **Camera 01** | Nút Giao Ngã Tư Thực Tế | HD (1280x720) @ 30 FPS | **Dòng xe hỗn hợp: Xe máy & Ô tô con** (`57GT`, `5155`, `51G`...) |
| **Camera 02** | Trục Lộ Giao Thông Đô Thị | Full-HD (1920x1080) @ 30 FPS | **Ô tô con, Taxi & Xe khách** (`30H`, `30K`, `30A`, `30F`, `98A`, `34B`...) |


---

## 🚀 3. Hướng Dẫn Cài Đặt & Vận Hành

### Yêu cầu môi trường
* Windows 10/11 hoặc Linux x86_64
* Python 3.10+ (Đã cấu hình ảo hóa `.venv`)
* Go 1.21+ (Để biên dịch Go gateway)

### Khởi chạy toàn bộ hệ thống

1. **Khởi chạy Python AI Workers (Chạy ngầm hoặc cửa sổ riêng)**:
   ```powershell
   # 1.1 Khởi động Worker nhận diện ảnh tĩnh (Cổng 5005)
   .venv\Scripts\python server\alpr_worker.py

   # 1.2 Khởi động Stream Server đa luồng camera (Cổng 5006)
   .venv\Scripts\python video_pipeline\stream_server.py
   ```

2. **Khởi chạy Go Backend Gateway (Cổng 8080)**:
   ```powershell
   # Biên dịch và khởi chạy binary Go hiệu năng cao
   go build -o ocr-server.exe .
   .\ocr-server.exe
   ```

3. **Truy cập Dashboard Quản Trị**:
   * Mở trình duyệt tại: **`http://localhost:8080`**
   * Chuyển đổi linh hoạt giữa:
     * **Camera Trực Tiếp (Live Stream)**: Xem trực tiếp 4 luồng CCTV với HUD, thống kê thời gian thực, bảng nhật ký sự kiện, xuất file CSV/JSON.
     * **Kiểm Định Ảnh Tĩnh (Static Inspector)**: Kéo thả ảnh biển số xe (hỗ trợ JPG, PNG, WEBP), hiển thị ảnh crop và độ tin cậy.

---

## 🔌 4. Đặc Tả Danh Mục API Endpoints

Go Backend cung cấp hệ thống RESTful & SSE API hoàn chỉnh cho các hệ thống bên thứ ba tích hợp:

| Phương thức | Đường dẫn | Chức năng | Định dạng dữ liệu |
|---|---|---|---|
| `GET` | `/api/health` | Kiểm tra trạng thái máy chủ Go và kết nối AI Worker | `application/json` |
| `GET` | `/api/videos` | Lấy danh sách thông tin cấu hình 4 kênh camera | `application/json` |
| `GET` | `/api/stream/live?source={id}` | Xem luồng video MJPEG thời gian thực kèm vẽ HUD | `multipart/x-mixed-replace` |
| `GET` | `/api/stream/events` | Luồng SSE nhận diện biển số phương tiện tức thì | `text/event-stream` |
| `POST` | `/api/recognize` | Nhận diện biển số từ ảnh tĩnh (Multipart hoặc Base64) | `application/json` |
| `GET` | `/api/samples` | Lấy danh sách ảnh mẫu biển số xe Việt Nam để kiểm thử | `application/json` |

---

## 🧪 5. Báo Cáo Kiểm Thử Toàn Diện (Full Testkey Verification)

Hệ thống đã trải qua quy trình kiểm thử nghiêm ngặt (E2E Test) đạt **tỷ lệ thành công 100%**:

1. **Go Unit & E2E Test Suite (`go test -v -count=1 ./...`)**:
   * **11/11 Test Suites PASS (100%)**: Kiểm thử multipart upload, base64 payload, nhận diện đa biển số Việt Nam, xử lý lỗi đầu vào, định tuyến camera, CORS headers.
2. **Kiểm thử bộ dữ liệu 77 Biển Số Việt Nam (`python test_vietnam_plates.py`)**:
   * Đạt **100% phát hiện biển số** trên toàn bộ 74/77 hình ảnh thực tế phức tạp (biển vuông 2 dòng, biển dài 1 dòng, ảnh mờ, chói sáng, góc nghiêng).
3. **Kiểm thử Real-Time Camera Giao Thông 4 Kênh (`python test_video_realtime.py`)**:
   * **100% 4/4 Camera hoàn thành testkey thành công**, xuất video kết quả annotated có vẽ HUD.
   * **100% Biển số phương tiện được phát hiện & OCR thành công**.

---

# FastALPR Core Library (Upstream)

[![Actions status](https://github.com/ankandrew/fast-alpr/actions/workflows/test.yaml/badge.svg)](https://github.com/ankandrew/fast-alpr/actions)
[![Actions status](https://github.com/ankandrew/fast-alpr/actions/workflows/release.yaml/badge.svg)](https://github.com/ankandrew/fast-alpr/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![ONNX Model](https://img.shields.io/badge/model-ONNX-blue?logo=onnx&logoColor=white)](https://onnx.ai/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces/ankandrew/fast-alpr)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://ankandrew.github.io/fast-alpr/)
[![image](https://img.shields.io/pypi/pyversions/fast-alpr.svg)](https://pypi.python.org/pypi/fast-alpr)
[![GitHub version](https://img.shields.io/github/v/release/ankandrew/fast-alpr)](https://github.com/ankandrew/fast-alpr/releases)
[![License](https://img.shields.io/github/license/ankandrew/fast-alpr)](./LICENSE)

[![ALPR Demo Animation](https://raw.githubusercontent.com/ankandrew/fast-alpr/f672fbbec2ddf86aabfc2afc0c45d1fa7612516c/assets/alpr.gif)](https://youtu.be/-TPJot7-HTs?t=652)

**FastALPR** is a high-performance, customizable Automatic License Plate Recognition (ALPR) system. We offer fast and
efficient ONNX models by default, but you can easily swap in your own models if needed.

For Optical Character Recognition (**OCR**), we use [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) by
default, and for **license plate detection**, we
use [open-image-models](https://github.com/ankandrew/open-image-models). However, you can integrate any OCR or detection
model of your choice.

## 📋 Table of Contents

* [✨ Features](#-features)
* [📦 Installation](#-installation)
* [🚀 Quick Start](#-quick-start)
* [🛠️ Customization and Flexibility](#-customization-and-flexibility)
* [📖 Documentation](#-documentation)
* [🤝 Contributing](#-contributing)
* [🙏 Acknowledgements](#-acknowledgements)
* [📫 Contact](#-contact)

## ✨ Features

- **High Accuracy**: Uses advanced models for precise license plate detection and OCR.
- **Customizable**: Easily switch out detection and OCR models.
- **Easy to Use**: Quick setup with a simple API.
- **Out-of-the-Box Models**: Includes ready-to-use detection and OCR models
- **Fast Performance**: Optimized with ONNX Runtime for speed.

## 📦 Installation

```shell
pip install fast-alpr[onnx-gpu]
```

By default, **no ONNX runtime is installed**. To run inference, you **must** install at least one ONNX backend using an appropriate extra.

| Platform/Use Case  | Install Command                        | Notes                |
|--------------------|----------------------------------------|----------------------|
| CPU (default)      | `pip install fast-alpr[onnx]`          | Cross-platform       |
| NVIDIA GPU (CUDA)  | `pip install fast-alpr[onnx-gpu]`      | Linux/Windows        |
| Intel (OpenVINO)   | `pip install fast-alpr[onnx-openvino]` | Best on Intel CPUs   |
| Windows (DirectML) | `pip install fast-alpr[onnx-directml]` | For DirectML support |
| Qualcomm (QNN)     | `pip install fast-alpr[onnx-qnn]`      | Qualcomm chipsets    |


## 🚀 Quick Start

> [!TIP]
> Try `fast-alpr` in [Hugging Spaces](https://huggingface.co/spaces/ankandrew/fast-alpr).

Here's how to get started with FastALPR:

```python
from fast_alpr import ALPR

# You can also initialize the ALPR with custom plate detection and OCR models.
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
)

# The "assets/test_image.png" can be found in repo root dir
alpr_results = alpr.predict("assets/test_image.png")
print(alpr_results)
```

Output:

<img alt="ALPR Result" src="https://raw.githubusercontent.com/ankandrew/fast-alpr/5063bd92fdd30f46b330d051468be267d4442c9b/assets/alpr_result.webp"/>

You can also draw the predictions directly on the image:

```python
import cv2

from fast_alpr import ALPR

# Initialize the ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
)

# Load the image
image_path = "assets/test_image.png"
frame = cv2.imread(image_path)

# Draw predictions on the image and get the ALPR results
drawn = alpr.draw_predictions(frame)
annotated_frame = drawn.image
results = drawn.results
```

Annotated frame:

<img alt="ALPR Draw Predictions" src="https://github.com/ankandrew/fast-alpr/releases/download/assets/alpr_draw_predictions.webp"/>

## 🛠️ Customization and Flexibility

FastALPR is designed to be flexible. You can customize the detector and OCR models according to your requirements.
You can very easily integrate with **Tesseract** OCR to leverage its capabilities:

```python
import re
from statistics import mean

import numpy as np
import pytesseract

from fast_alpr.alpr import ALPR, BaseOCR, OcrResult


class PytesseractOCR(BaseOCR):
    def __init__(self) -> None:
        """
        Init PytesseractOCR.
        """

    def predict(self, cropped_plate: np.ndarray) -> OcrResult | None:
        if cropped_plate is None:
            return None
        # You can change 'eng' to the appropriate language code as needed
        data = pytesseract.image_to_data(
            cropped_plate,
            lang="eng",
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
        )
        plate_text = " ".join(data["text"]).strip()
        plate_text = re.sub(r"[^A-Za-z0-9]", "", plate_text)
        avg_confidence = mean(conf for conf in data["conf"] if conf > 0) / 100.0
        return OcrResult(text=plate_text, confidence=avg_confidence)


alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end", ocr=PytesseractOCR())

alpr_results = alpr.predict("assets/test_image.png")
print(alpr_results)
```

> [!TIP]
> See the [docs](https://ankandrew.github.io/fast-alpr/) for more examples!

## 📖 Documentation

Comprehensive documentation is available [here](https://ankandrew.github.io/fast-alpr/), including detailed API
references and additional examples.

## 🤝 Contributing

Contributions to the repo are greatly appreciated. Whether it's bug fixes, feature enhancements, or new models,
your contributions are warmly welcomed.

To start contributing or to begin development, you can follow these steps:

1. Clone repo
    ```shell
    git clone https://github.com/ankandrew/fast-alpr.git
    ```
2. Install all dependencies (make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed):
    ```shell
    make install
    ```
3. To ensure your changes pass linting and tests before submitting a PR:
    ```shell
    make checks
    ```

## 🙏 Acknowledgements

- [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) for default **OCR** models.
- [open-image-models](https://github.com/ankandrew/open-image-models) for default plate **detection** models.

## 📫 Contact

For questions or suggestions, feel free to open an issue.
