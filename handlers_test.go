package main

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// MockWorkerServer tạo một httptest server giả lập AI Worker cho unit test
func createMockWorkerServer(t *testing.T) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"status":"ok","service":"MockWorker","device":"cpu"}`))
		case "/predict":
			// Giả lập kết quả nhận diện
			resp := WorkerPredictResponse{
				Success:         true,
				ImageWidth:      800,
				ImageHeight:     600,
				InferenceTimeMs: 32.5,
				TotalPlates:     1,
				Plates: []PlateResult{
					{
						Text:             "61T32222",
						DetectConfidence: 0.906,
						OcrConfidence:    0.999,
						Region:           "Vietnam",
						RegionConfidence: 0.999,
						BoundingBox: BoundingBox{
							X1: 100, Y1: 200, X2: 300, Y2: 260, Width: 200, Height: 60,
						},
					},
				},
				AnnotatedImage: "data:image/jpeg;base64,/9j/mockimage",
			}
			_ = json.NewEncoder(w).Encode(resp)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

func setupTestServer(t *testing.T, mockWorkerURL string) *AppServer {
	staticDir := "static"
	datasetDir := "vn_test_dataset"
	return &AppServer{
		Worker:     NewWorkerClient(mockWorkerURL),
		StaticDir:  staticDir,
		DatasetDir: datasetDir,
	}
}

// 1. Test GET /api/health
func TestHandleHealth(t *testing.T) {
	mockWorker := createMockWorkerServer(t)
	defer mockWorker.Close()

	app := setupTestServer(t, mockWorker.URL)

	req := httptest.NewRequest(http.MethodGet, "/api/health", nil)
	rec := httptest.NewRecorder()

	app.HandleHealth(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("kỳ vọng mã 200, nhận được: %d", rec.Code)
	}

	var resp HealthResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("giải mã JSON thất bại: %v", err)
	}

	if resp.Status != "ok" {
		t.Errorf("kỳ vọng status 'ok', nhận được: %s", resp.Status)
	}
	if resp.WorkerStatus != "running" {
		t.Errorf("kỳ vọng worker_status 'running', nhận được: %s", resp.WorkerStatus)
	}
	if !strings.Contains(resp.Device, "CPU") {
		t.Errorf("kỳ vọng thiết bị chứa CPU, nhận được: %s", resp.Device)
	}
}

// 2. Test GET /api/samples
func TestHandleSamples(t *testing.T) {
	mockWorker := createMockWorkerServer(t)
	defer mockWorker.Close()

	app := setupTestServer(t, mockWorker.URL)

	req := httptest.NewRequest(http.MethodGet, "/api/samples", nil)
	rec := httptest.NewRecorder()

	app.HandleSamples(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("kỳ vọng mã 200, nhận được: %d", rec.Code)
	}

	var samples []SampleItem
	if err := json.Unmarshal(rec.Body.Bytes(), &samples); err != nil {
		t.Fatalf("giải mã samples JSON thất bại: %v", err)
	}

	if len(samples) == 0 {
		t.Errorf("danh sách samples không được rỗng")
	}

	hasVN := false
	for _, s := range samples {
		if strings.Contains(s.Category, "Việt Nam") {
			hasVN = true
			break
		}
	}
	if !hasVN {
		t.Errorf("danh sách samples cần chứa ít nhất một mẫu Việt Nam")
	}
}

// 3. Test GET /api/sample-image/
func TestHandleSampleImage(t *testing.T) {
	mockWorker := createMockWorkerServer(t)
	defer mockWorker.Close()

	app := setupTestServer(t, mockWorker.URL)

	// Test ảnh có thật
	req := httptest.NewRequest(http.MethodGet, "/api/sample-image/test_image.png", nil)
	rec := httptest.NewRecorder()
	app.HandleSampleImage(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("kỳ vọng mã 200 cho test_image.png, nhận được: %d", rec.Code)
	}

	// Test ảnh không tồn tại
	req404 := httptest.NewRequest(http.MethodGet, "/api/sample-image/not_exist.jpg", nil)
	rec404 := httptest.NewRecorder()
	app.HandleSampleImage(rec404, req404)

	if rec404.Code != http.StatusNotFound {
		t.Errorf("kỳ vọng mã 404 cho ảnh không tồn tại, nhận được: %d", rec404.Code)
	}
}

// 4. Test GET / (index.html)
func TestHandleIndex(t *testing.T) {
	mockWorker := createMockWorkerServer(t)
	defer mockWorker.Close()

	app := setupTestServer(t, mockWorker.URL)

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	app.HandleIndex(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("kỳ vọng mã 200 khi truy cập /, nhận được: %d", rec.Code)
	}

	body := rec.Body.String()
	if !strings.Contains(body, "VN-ANPR Enterprise") {
		t.Errorf("kỳ vọng index.html chứa 'VN-ANPR Enterprise'")
	}
}

// 5. Test Validation POST /api/recognize
func TestHandleRecognize_Validation(t *testing.T) {
	mockWorker := createMockWorkerServer(t)
	defer mockWorker.Close()

	app := setupTestServer(t, mockWorker.URL)

	// Method không hợp lệ: GET
	reqGet := httptest.NewRequest(http.MethodGet, "/api/recognize", nil)
	recGet := httptest.NewRecorder()
	app.HandleRecognize(recGet, reqGet)
	if recGet.Code != http.StatusMethodNotAllowed {
		t.Errorf("kỳ vọng mã 405 khi gọi GET /api/recognize, nhận: %d", recGet.Code)
	}

	// Body rỗng
	reqEmpty := httptest.NewRequest(http.MethodPost, "/api/recognize", bytes.NewReader([]byte{}))
	recEmpty := httptest.NewRecorder()
	app.HandleRecognize(recEmpty, reqEmpty)
	if recEmpty.Code != http.StatusBadRequest {
		t.Errorf("kỳ vọng mã 400 khi body rỗng, nhận: %d", recEmpty.Code)
	}

	// Tệp quá nhỏ (< 50 bytes)
	reqTiny := httptest.NewRequest(http.MethodPost, "/api/recognize", bytes.NewReader([]byte("short string")))
	recTiny := httptest.NewRecorder()
	app.HandleRecognize(recTiny, reqTiny)
	if recTiny.Code != http.StatusBadRequest {
		t.Errorf("kỳ vọng mã 400 khi dữ liệu quá nhỏ, nhận: %d", recTiny.Code)
	}
}

// 6. Test Thành Công Multipart Upload với Mock Worker
func TestHandleRecognize_MultipartSuccess(t *testing.T) {
	mockWorker := createMockWorkerServer(t)
	defer mockWorker.Close()

	app := setupTestServer(t, mockWorker.URL)

	// Đọc ảnh mẫu
	testImgPath := filepath.Join("assets", "test_image.png")
	imgData, err := os.ReadFile(testImgPath)
	if err != nil {
		t.Fatalf("không thể đọc ảnh mẫu: %v", err)
	}

	// Tạo multipart request
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("image", "test.png")
	if err != nil {
		t.Fatalf("tạo multipart form file thất bại: %v", err)
	}
	_, _ = part.Write(imgData)
	_ = writer.Close()

	req := httptest.NewRequest(http.MethodPost, "/api/recognize", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	rec := httptest.NewRecorder()

	app.HandleRecognize(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("kỳ vọng mã 200, nhận: %d, body: %s", rec.Code, rec.Body.String())
	}

	var resp RecognizeResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("giải mã JSON thất bại: %v", err)
	}

	if !resp.Success {
		t.Errorf("kỳ vọng success=true")
	}
	if resp.TotalPlates != 1 {
		t.Errorf("kỳ vọng 1 biển số, nhận: %d", resp.TotalPlates)
	}
	if len(resp.Plates) > 0 && resp.Plates[0].Text != "61T32222" {
		t.Errorf("kỳ vọng biển số '61T32222', nhận: %s", resp.Plates[0].Text)
	}
}

// 7. Test GET /api/videos
func TestHandleVideos(t *testing.T) {
	mockWorker := createMockWorkerServer(t)
	defer mockWorker.Close()

	app := setupTestServer(t, mockWorker.URL)

	req := httptest.NewRequest(http.MethodGet, "/api/videos", nil)
	rec := httptest.NewRecorder()
	app.HandleVideos(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("kỳ vọng mã 200, nhận: %d", rec.Code)
	}

	var videos []map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &videos); err != nil {
		t.Fatalf("giải mã JSON videos thất bại: %v", err)
	}

	if len(videos) < 2 {
		t.Errorf("kỳ vọng ít nhất 2 kênh camera, nhận: %d", len(videos))
	}
}

