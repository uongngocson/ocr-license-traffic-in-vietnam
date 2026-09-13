package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"time"
)

// WorkerClient quản lý giao tiếp HTTP với tiến trình Python AI Worker
type WorkerClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewWorkerClient khởi tạo một client mới với timeout cấu hình
func NewWorkerClient(baseURL string) *WorkerClient {
	return &WorkerClient{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// CheckHealth kiểm tra xem Python AI Worker có đang phản hồi không
func (w *WorkerClient) CheckHealth() (bool, error) {
	req, err := http.NewRequest("GET", w.BaseURL+"/health", nil)
	if err != nil {
		return false, err
	}

	client := &http.Client{Timeout: 1 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK, nil
}

// EnsureWorkerRunning kiểm tra và tự động khởi động Python Worker nếu chưa chạy
func (w *WorkerClient) EnsureWorkerRunning(pythonExe, scriptPath string) error {
	running, _ := w.CheckHealth()
	if running {
		return nil
	}

	fmt.Printf("[Go Backend] AI Worker chưa chạy. Đang tự động khởi động: %s %s...\n", pythonExe, scriptPath)

	if _, err := os.Stat(pythonExe); os.IsNotExist(err) {
		return fmt.Errorf("không tìm thấy Python tại đường dẫn: %s", pythonExe)
	}
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return fmt.Errorf("không tìm thấy worker script tại đường dẫn: %s", scriptPath)
	}

	cmd := exec.Command(pythonExe, scriptPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("lỗi khởi động worker: %w", err)
	}

	// Chờ tối đa 15 giây để worker nạp xong model vào RAM
	fmt.Print("[Go Backend] Đang chờ AI Worker sẵn sàng")
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		time.Sleep(500 * time.Millisecond)
		fmt.Print(".")
		if ok, _ := w.CheckHealth(); ok {
			fmt.Println("\n[Go Backend] AI Worker đã kết nối thành công!")
			return nil
		}
	}

	return fmt.Errorf("quá thời gian chờ AI Worker khởi động (15s)")
}

// Predict gửi dữ liệu byte của ảnh sang Worker để suy luận
func (w *WorkerClient) Predict(imageBytes []byte) (*WorkerPredictResponse, error) {
	req, err := http.NewRequest("POST", w.BaseURL+"/predict", bytes.NewReader(imageBytes))
	if err != nil {
		return nil, fmt.Errorf("tạo request thất bại: %w", err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")

	resp, err := w.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gửi request tới AI worker thất bại: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("đọc phản hồi từ worker thất bại: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Error string `json:"error"`
		}
		_ = json.Unmarshal(respBody, &errResp)
		if errResp.Error != "" {
			return nil, fmt.Errorf("lỗi từ AI worker (%d): %s", resp.StatusCode, errResp.Error)
		}
		return nil, fmt.Errorf("AI worker trả về mã lỗi HTTP %d", resp.StatusCode)
	}

	var result WorkerPredictResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("giải mã JSON từ worker thất bại: %w", err)
	}

	return &result, nil
}
