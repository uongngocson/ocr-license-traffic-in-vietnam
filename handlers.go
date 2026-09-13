package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// AppServer chứa các phụ thuộc để xử lý HTTP request
type AppServer struct {
	Worker     *WorkerClient
	StreamURL  string
	StaticDir  string
	DatasetDir string
}

// RegisterRoutes đăng ký toàn bộ endpoints
func (s *AppServer) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/", s.HandleIndex)
	mux.HandleFunc("/api/health", s.HandleHealth)
	mux.HandleFunc("/api/recognize", s.HandleRecognize)
	mux.HandleFunc("/api/samples", s.HandleSamples)
	mux.HandleFunc("/api/sample-image/", s.HandleSampleImage)
	mux.HandleFunc("/api/videos", s.HandleVideos)
	mux.HandleFunc("/api/stream/live", s.HandleStreamLive)
	mux.HandleFunc("/api/stream/events", s.HandleStreamEvents)
}

func enableCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
}

func sendJSON(w http.ResponseWriter, statusCode int, data interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(data)
}

// HandleIndex phục vụ giao diện Single-Page HTML
func (s *AppServer) HandleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		relPath := strings.TrimPrefix(r.URL.Path, "/")
		targetFile := filepath.Join(s.StaticDir, relPath)
		if _, err := os.Stat(targetFile); err == nil {
			http.ServeFile(w, r, targetFile)
			return
		}
		http.NotFound(w, r)
		return
	}

	indexPath := filepath.Join(s.StaticDir, "index.html")
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		http.Error(w, "Chưa tìm thấy index.html", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	http.ServeFile(w, r, indexPath)
}

// HandleHealth kiểm tra sức khỏe của Go Backend và Python Worker
func (s *AppServer) HandleHealth(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	workerOK, _ := s.Worker.CheckHealth()
	workerStatus := "running"
	if !workerOK {
		workerStatus = "unreachable"
	}

	resp := HealthResponse{
		Status:       "ok",
		Backend:      "Go FastALPR API Server",
		GoVersion:    runtime.Version(),
		WorkerStatus: workerStatus,
		WorkerURL:    s.Worker.BaseURL,
		Device:       "CPU (CPUExecutionProvider)",
	}

	sendJSON(w, http.StatusOK, resp)
}

// HandleRecognize tiếp nhận ảnh tải lên và gọi AI Worker
func (s *AppServer) HandleRecognize(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	if r.Method != http.MethodPost {
		sendJSON(w, http.StatusMethodNotAllowed, RecognizeResponse{
			Success: false,
			Error:   "Chỉ hỗ trợ phương thức POST",
		})
		return
	}

	startTime := time.Now()
	var imageBytes []byte

	contentType := r.Header.Get("Content-Type")
	if strings.HasPrefix(contentType, "multipart/form-data") {
		if err := r.ParseMultipartForm(20 << 20); err != nil {
			sendJSON(w, http.StatusBadRequest, RecognizeResponse{
				Success: false,
				Error:   fmt.Sprintf("Lỗi phân tích form-data: %v", err),
			})
			return
		}

		file, _, err := r.FormFile("image")
		if err != nil {
			file, _, err = r.FormFile("file")
		}
		if err != nil {
			sendJSON(w, http.StatusBadRequest, RecognizeResponse{
				Success: false,
				Error:   "Không tìm thấy trường 'image' hoặc 'file' trong multipart body",
			})
			return
		}
		defer file.Close()

		var buf bytes.Buffer
		if _, err := io.Copy(&buf, file); err != nil {
			sendJSON(w, http.StatusInternalServerError, RecognizeResponse{
				Success: false,
				Error:   "Đọc tệp ảnh thất bại",
			})
			return
		}
		imageBytes = buf.Bytes()
	} else if strings.HasPrefix(contentType, "application/json") {
		var jsonReq JsonRecognizeRequest
		if err := json.NewDecoder(r.Body).Decode(&jsonReq); err != nil {
			sendJSON(w, http.StatusBadRequest, RecognizeResponse{
				Success: false,
				Error:   "Định dạng JSON không hợp lệ",
			})
			return
		}

		b64Data := jsonReq.ImageBase64
		if idx := strings.Index(b64Data, ","); idx != -1 {
			b64Data = b64Data[idx+1:]
		}

		decoded, err := base64.StdEncoding.DecodeString(b64Data)
		if err != nil {
			sendJSON(w, http.StatusBadRequest, RecognizeResponse{
				Success: false,
				Error:   "Dữ liệu Base64 ảnh không hợp lệ",
			})
			return
		}
		imageBytes = decoded
	} else {
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil || len(bodyBytes) == 0 {
			sendJSON(w, http.StatusBadRequest, RecognizeResponse{
				Success: false,
				Error:   "Không nhận được dữ liệu ảnh",
			})
			return
		}
		imageBytes = bodyBytes
	}

	if len(imageBytes) < 50 {
		sendJSON(w, http.StatusBadRequest, RecognizeResponse{
			Success: false,
			Error:   "Dữ liệu ảnh quá nhỏ hoặc rỗng",
		})
		return
	}

	workerResp, err := s.Worker.Predict(imageBytes)
	if err != nil {
		sendJSON(w, http.StatusInternalServerError, RecognizeResponse{
			Success: false,
			Error:   fmt.Sprintf("Lỗi suy luận AI Worker: %v", err),
		})
		return
	}

	backendDurationMs := float64(time.Since(startTime).Microseconds()) / 1000.0

	finalResp := RecognizeResponse{
		Success:         workerResp.Success,
		Error:           workerResp.Error,
		ImageWidth:      workerResp.ImageWidth,
		ImageHeight:     workerResp.ImageHeight,
		InferenceTimeMs: workerResp.InferenceTimeMs,
		BackendTimeMs:   backendDurationMs,
		TotalPlates:     workerResp.TotalPlates,
		Plates:          workerResp.Plates,
		AnnotatedImage:  workerResp.AnnotatedImage,
	}

	sendJSON(w, http.StatusOK, finalResp)
}

// HandleSamples trả về danh sách ảnh mẫu đại diện cho người dùng test nhanh
func (s *AppServer) HandleSamples(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	samples := []SampleItem{
		{
			ID:          "vn-car-1",
			Title:       "Ô tô Việt Nam (61T-322.22)",
			Category:    "Việt Nam",
			Description: "Biển ô tô Việt Nam một dòng chụp góc thẳng",
			Filename:    "mrzaizai2k_1.jpg",
			URL:         "/api/sample-image/mrzaizai2k_1.jpg",
		},
		{
			ID:          "vn-multi-3",
			Title:       "Đa biển số (3 xe cùng lúc)",
			Category:    "Việt Nam",
			Description: "Khung hình nhiều phương tiện giao thông trên đường",
			Filename:    "mrzaizai2k_10.jpg",
			URL:         "/api/sample-image/mrzaizai2k_10.jpg",
		},
		{
			ID:          "vn-bike-gate",
			Title:       "Xe máy Bãi giữ xe",
			Category:    "Việt Nam",
			Description: "Ảnh chụp từ camera bãi xe barrier (65L1-007.80)",
			Filename:    "NamYCM_Bike_back_0402.jpg",
			URL:         "/api/sample-image/NamYCM_Bike_back_0402.jpg",
		},
		{
			ID:          "vn-car-square",
			Title:       "Ô tô Biển Vuông (88A-118.86)",
			Category:    "Việt Nam",
			Description: "Biển vuông 2 dòng xe ô tô",
			Filename:    "quangnhat185_vietnam_car_square_plate.jpg",
			URL:         "/api/sample-image/quangnhat185_vietnam_car_square_plate.jpg",
		},
		{
			ID:          "eu-car-1",
			Title:       "Biển Quốc Tế (5AU5341)",
			Category:    "Châu Âu",
			Description: "Ảnh mẫu tiêu chuẩn FastALPR",
			Filename:    "test_image.png",
			URL:         "/api/sample-image/test_image.png",
		},
	}

	sendJSON(w, http.StatusOK, samples)
}

// HandleSampleImage phục vụ tải dữ liệu file ảnh mẫu
func (s *AppServer) HandleSampleImage(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	fname := strings.TrimPrefix(r.URL.Path, "/api/sample-image/")
	fname = filepath.Base(fname)

	candidate1 := filepath.Join(s.DatasetDir, fname)
	candidate2 := filepath.Join("assets", fname)

	var targetPath string
	if _, err := os.Stat(candidate1); err == nil {
		targetPath = candidate1
	} else if _, err := os.Stat(candidate2); err == nil {
		targetPath = candidate2
	} else {
		http.NotFound(w, r)
		return
	}

	switch strings.ToLower(filepath.Ext(targetPath)) {
	case ".jpg", ".jpeg":
		w.Header().Set("Content-Type", "image/jpeg")
	case ".png":
		w.Header().Set("Content-Type", "image/png")
	case ".webp":
		w.Header().Set("Content-Type", "image/webp")
	}

	http.ServeFile(w, r, targetPath)
}

// HandleVideos trả về danh sách các kênh camera giao thông có sẵn
func (s *AppServer) HandleVideos(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	videos := []map[string]string{
		{
			"id":          "traffic_1",
			"name":        "Camera 01 - Nút Giao Ngã Tư (HD 720p)",
			"resolution":  "1280 x 720 (HD 720p)",
			"fps":         "30.0 FPS",
			"description": "Nút giao ngã tư thực tế tại Việt Nam (Camera giám sát giao thông đô thị)",
		},
		{
			"id":          "traffic_2",
			"name":        "Camera 02 - Tuyến Giao Thông Đô Thị (Full-HD)",
			"resolution":  "1920 x 1080 (Full-HD)",
			"fps":         "30.0 FPS",
			"description": "Trục lộ giao thông đô thị mật độ cao (Camera giám sát làn đường)",
		},
	}

	sendJSON(w, http.StatusOK, videos)
}

// HandleStreamLive chuyển tiếp luồng MJPEG Real-Time từ Python Stream Engine
func (s *AppServer) HandleStreamLive(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	source := r.URL.Query().Get("source")
	if source == "" {
		source = "traffic_1"
	}

	streamURL := s.StreamURL
	if streamURL == "" {
		streamURL = "http://127.0.0.1:5006"
	}

	targetURL := fmt.Sprintf("%s/stream?source=%s", streamURL, source)
	req, err := http.NewRequestWithContext(r.Context(), "GET", targetURL, nil)
	if err != nil {
		http.Error(w, "Yêu cầu stream không hợp lệ", http.StatusBadRequest)
		return
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		http.Error(w, "Không thể kết nối tới Động cơ Stream: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	for k, v := range resp.Header {
		w.Header()[k] = v
	}
	w.WriteHeader(resp.StatusCode)

	flusher, ok := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		select {
		case <-r.Context().Done():
			return
		default:
			n, err := resp.Body.Read(buf)
			if n > 0 {
				if _, wErr := w.Write(buf[:n]); wErr != nil {
					return
				}
				if ok {
					flusher.Flush()
				}
			}
			if err != nil {
				return
			}
		}
	}
}

// HandleStreamEvents chuyển tiếp sự kiện Server-Sent Events (SSE) Real-Time
func (s *AppServer) HandleStreamEvents(w http.ResponseWriter, r *http.Request) {
	enableCORS(w)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	streamURL := s.StreamURL
	if streamURL == "" {
		streamURL = "http://127.0.0.1:5006"
	}

	targetURL := fmt.Sprintf("%s/events", streamURL)
	req, err := http.NewRequestWithContext(r.Context(), "GET", targetURL, nil)
	if err != nil {
		http.Error(w, "Yêu cầu events không hợp lệ", http.StatusBadRequest)
		return
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		http.Error(w, "Không thể kết nối event stream: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	flusher, ok := w.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		select {
		case <-r.Context().Done():
			return
		default:
			n, err := resp.Body.Read(buf)
			if n > 0 {
				if _, wErr := w.Write(buf[:n]); wErr != nil {
					return
				}
				if ok {
					flusher.Flush()
				}
			}
			if err != nil {
				return
			}
		}
	}
}
