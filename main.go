package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

func ensureStreamServerRunning(streamURL, pythonExe, scriptPath string) {
	client := &http.Client{Timeout: 1 * time.Second}
	resp, err := client.Get(streamURL + "/health")
	if err == nil && resp.StatusCode == http.StatusOK {
		resp.Body.Close()
		return
	}

	fmt.Printf("[Go Backend] Stream Server chưa chạy. Đang tự động khởi động: %s %s...\n", pythonExe, scriptPath)
	cmd := exec.Command(pythonExe, scriptPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		log.Printf("[CẢNH BÁO] Không thể khởi động Stream Server: %v\n", err)
		return
	}

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		time.Sleep(500 * time.Millisecond)
		r, e := client.Get(streamURL + "/health")
		if e == nil && r.StatusCode == http.StatusOK {
			r.Body.Close()
			fmt.Println("[Go Backend] Stream Server đã sẵn sàng!")
			return
		}
	}
}

func main() {
	port := flag.String("port", "8080", "Cổng HTTP Server của Go Backend")
	workerURL := flag.String("worker", "http://127.0.0.1:5005", "URL của Python AI Worker")
	streamURL := flag.String("stream", "http://127.0.0.1:5006", "URL của Python Stream Server")
	pythonExe := flag.String("python", filepath.Join(".venv", "Scripts", "python.exe"), "Đường dẫn thực thi Python của venv")
	workerScript := flag.String("script", filepath.Join("server", "alpr_worker.py"), "Đường dẫn script worker")
	streamScript := flag.String("stream-script", filepath.Join("video_pipeline", "stream_server.py"), "Đường dẫn stream script")
	staticDir := flag.String("static", "static", "Thư mục chứa giao diện web (HTML/CSS/JS)")
	datasetDir := flag.String("dataset", "vn_test_dataset", "Thư mục chứa ảnh mẫu test")
	flag.Parse()

	if envPort := os.Getenv("PORT"); envPort != "" {
		*port = envPort
	}
	if envWorker := os.Getenv("ALPR_WORKER_URL"); envWorker != "" {
		*workerURL = envWorker
	}
	if envStream := os.Getenv("STREAM_SERVER_URL"); envStream != "" {
		*streamURL = envStream
	}

	fmt.Println("==========================================================================")
	fmt.Println("       KHỞI ĐỘNG HỆ THỐNG VN-ANPR REAL-TIME CCTV & GO BACKEND             ")
	fmt.Println("==========================================================================")

	workerClient := NewWorkerClient(*workerURL)

	// Đảm bảo Python AI Worker & Stream Server đang chạy
	if err := workerClient.EnsureWorkerRunning(*pythonExe, *workerScript); err != nil {
		log.Printf("[CẢNH BÁO] AI Worker chưa sẵn sàng: %v\n", err)
	}
	ensureStreamServerRunning(*streamURL, *pythonExe, *streamScript)

	app := &AppServer{
		Worker:     workerClient,
		StreamURL:  *streamURL,
		StaticDir:  *staticDir,
		DatasetDir: *datasetDir,
	}

	mux := http.NewServeMux()
	app.RegisterRoutes(mux)

	serverAddr := ":" + *port
	httpServer := &http.Server{
		Addr:         serverAddr,
		Handler:      mux,
		ReadTimeout:  0, // Vô hiệu hóa timeout cho luồng streaming vô hạn
		WriteTimeout: 0,
	}

	stopChan := make(chan os.Signal, 1)
	signal.Notify(stopChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		fmt.Printf("\n[Go Gateway] Lắng nghe tại: http://localhost:%s\n", *port)
		fmt.Printf("[Go Gateway] Dashboard Web: http://localhost:%s\n", *port)
		fmt.Printf("[Go Gateway] Live Stream  : http://localhost:%s/api/stream/live?source=traffic_1\n", *port)
		fmt.Printf("[Go Gateway] Event Stream : http://localhost:%s/api/stream/events\n\n", *port)

		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Lỗi máy chủ HTTP: %v", err)
		}
	}()

	<-stopChan
	fmt.Println("\n[Go Backend] Đang dừng máy chủ...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(ctx)
	fmt.Println("[Go Backend] Đã dừng máy chủ thành công.")
}
