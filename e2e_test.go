package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// setupE2ETestServer chuẩn bị server kết nối với Python AI Worker thực tế
func setupE2ETestServer(t *testing.T) (*AppServer, func()) {
	workerURL := "http://127.0.0.1:5005"
	pythonExe := filepath.Join(".venv", "Scripts", "python.exe")
	workerScript := filepath.Join("server", "alpr_worker.py")

	workerClient := NewWorkerClient(workerURL)
	if err := workerClient.EnsureWorkerRunning(pythonExe, workerScript); err != nil {
		t.Skipf("Bỏ qua E2E test vì không thể khởi chạy AI Worker: %v", err)
	}

	app := &AppServer{
		Worker:     workerClient,
		StaticDir:  "static",
		DatasetDir: "vn_test_dataset",
	}

	cleanup := func() {
		// Cleanup nếu cần
	}
	return app, cleanup
}

// 1. E2E Test với ảnh chuẩn FastALPR: assets/test_image.png
func TestE2E_RealModel_AssetImage(t *testing.T) {
	app, cleanup := setupE2ETestServer(t)
	defer cleanup()

	imgPath := filepath.Join("assets", "test_image.png")
	imgBytes, err := os.ReadFile(imgPath)
	if err != nil {
		t.Fatalf("Không thể đọc ảnh assets/test_image.png: %v", err)
	}

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("image", "test_image.png")
	if err != nil {
		t.Fatalf("Tạo form file thất bại: %v", err)
	}
	_, _ = part.Write(imgBytes)
	_ = writer.Close()

	req := httptest.NewRequest(http.MethodPost, "/api/recognize", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	rec := httptest.NewRecorder()

	app.HandleRecognize(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("Kỳ vọng mã 200, nhận: %d - %s", rec.Code, rec.Body.String())
	}

	var resp RecognizeResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Giải mã JSON thất bại: %v", err)
	}

	if !resp.Success {
		t.Fatalf("Kỳ vọng resp.Success=true")
	}
	if resp.TotalPlates < 1 {
		t.Fatalf("Kỳ vọng phát hiện ít nhất 1 biển số, nhận: %d", resp.TotalPlates)
	}

	foundExpected := false
	for _, p := range resp.Plates {
		if strings.Contains(p.Text, "5AU5341") {
			foundExpected = true
			if p.OcrConfidence < 0.95 {
				t.Errorf("Độ tin cậy OCR quá thấp: %.2f", p.OcrConfidence)
			}
			break
		}
	}
	if !foundExpected {
		t.Errorf("Không tìm thấy biển số '5AU5341', nhận được: %v", resp.Plates)
	}

	if !strings.HasPrefix(resp.AnnotatedImage, "data:image/jpeg;base64,") {
		t.Errorf("AnnotatedImage không đúng định dạng Base64 data URI")
	}
	if resp.InferenceTimeMs <= 0 {
		t.Errorf("Thời gian suy luận không hợp lệ: %.2f", resp.InferenceTimeMs)
	}
}

// 2. E2E Test với ảnh biển số Việt Nam: vn_test_dataset/mrzaizai2k_1.jpg
func TestE2E_RealModel_VietnamImage(t *testing.T) {
	app, cleanup := setupE2ETestServer(t)
	defer cleanup()

	imgPath := filepath.Join("vn_test_dataset", "mrzaizai2k_1.jpg")
	imgBytes, err := os.ReadFile(imgPath)
	if err != nil {
		t.Fatalf("Không thể đọc ảnh: %v", err)
	}

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, _ := writer.CreateFormFile("image", "mrzaizai2k_1.jpg")
	_, _ = part.Write(imgBytes)
	_ = writer.Close()

	req := httptest.NewRequest(http.MethodPost, "/api/recognize", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	rec := httptest.NewRecorder()

	app.HandleRecognize(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("Kỳ vọng mã 200, nhận: %d", rec.Code)
	}

	var resp RecognizeResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &resp)

	if resp.TotalPlates < 1 {
		t.Fatalf("Kỳ vọng phát hiện biển số Việt Nam, nhận: 0")
	}

	plate := resp.Plates[0]
	if !strings.Contains(plate.Text, "61T32222") {
		t.Errorf("Kỳ vọng biển số chứa '61T32222', nhận: %s", plate.Text)
	}
	if plate.Region != "Vietnam" {
		t.Errorf("Kỳ vọng Region='Vietnam', nhận: %s", plate.Region)
	}
	if plate.OcrConfidence < 0.95 {
		t.Errorf("Kỳ vọng OCR Confidence >= 0.95, nhận: %.2f", plate.OcrConfidence)
	}
}

// 3. E2E Test với JSON Base64 request
func TestE2E_RealModel_Base64JSON(t *testing.T) {
	app, cleanup := setupE2ETestServer(t)
	defer cleanup()

	imgPath := filepath.Join("assets", "test_image.png")
	imgBytes, _ := os.ReadFile(imgPath)
	b64Str := "data:image/png;base64," + base64.StdEncoding.EncodeToString(imgBytes)

	jsonPayload := map[string]string{
		"image_base64": b64Str,
	}
	jsonBytes, _ := json.Marshal(jsonPayload)

	req := httptest.NewRequest(http.MethodPost, "/api/recognize", bytes.NewReader(jsonBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	app.HandleRecognize(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("Kỳ vọng mã 200 qua JSON Base64, nhận: %d", rec.Code)
	}

	var resp RecognizeResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &resp)

	if !resp.Success || resp.TotalPlates == 0 {
		t.Fatalf("Kỳ vọng nhận diện thành công qua JSON Base64")
	}
}

// 4. E2E Test ảnh đa biển số Việt Nam: vn_test_dataset/mrzaizai2k_10.jpg
func TestE2E_RealModel_MultiPlateVietnam(t *testing.T) {
	app, cleanup := setupE2ETestServer(t)
	defer cleanup()

	imgPath := filepath.Join("vn_test_dataset", "mrzaizai2k_10.jpg")
	imgBytes, err := os.ReadFile(imgPath)
	if err != nil {
		t.Fatalf("Không thể đọc ảnh: %v", err)
	}

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, _ := writer.CreateFormFile("image", "mrzaizai2k_10.jpg")
	_, _ = part.Write(imgBytes)
	_ = writer.Close()

	req := httptest.NewRequest(http.MethodPost, "/api/recognize", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	rec := httptest.NewRecorder()

	app.HandleRecognize(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("Kỳ vọng mã 200, nhận: %d", rec.Code)
	}

	var resp RecognizeResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &resp)

	if resp.TotalPlates < 3 {
		t.Errorf("Kỳ vọng phát hiện ít nhất 3 biển số trong ảnh đa phương tiện, nhận: %d", resp.TotalPlates)
	}
}
