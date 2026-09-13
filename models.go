package main

// BoundingBox mô tả tọa độ và kích thước của khung biển số
type BoundingBox struct {
	X1     float64 `json:"x1"`
	Y1     float64 `json:"y1"`
	X2     float64 `json:"x2"`
	Y2     float64 `json:"y2"`
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}

// PlateResult chứa thông tin của một biển số được phát hiện và nhận diện
type PlateResult struct {
	Text             string      `json:"text"`
	DetectConfidence float64     `json:"detect_confidence"`
	OcrConfidence    float64     `json:"ocr_confidence"`
	CharConfidences  []float64   `json:"char_confidences"`
	Region           string      `json:"region"`
	RegionConfidence float64     `json:"region_confidence"`
	PlateType        string      `json:"plate_type,omitempty"`
	CropImage        string      `json:"crop_image,omitempty"`
	BoundingBox      BoundingBox `json:"bounding_box"`
}

// WorkerPredictResponse là phản hồi thô từ Python AI Worker
type WorkerPredictResponse struct {
	Success         bool          `json:"success"`
	Error           string        `json:"error,omitempty"`
	ImageWidth      int           `json:"image_width"`
	ImageHeight     int           `json:"image_height"`
	InferenceTimeMs float64       `json:"inference_time_ms"`
	TotalPlates     int           `json:"total_plates"`
	Plates          []PlateResult `json:"plates"`
	AnnotatedImage  string        `json:"annotated_image"`
}

// RecognizeResponse là phản hồi API trả về cho Frontend UI
type RecognizeResponse struct {
	Success         bool          `json:"success"`
	Error           string        `json:"error,omitempty"`
	ImageWidth      int           `json:"image_width"`
	ImageHeight     int           `json:"image_height"`
	InferenceTimeMs float64       `json:"inference_time_ms"`
	BackendTimeMs   float64       `json:"backend_time_ms"`
	TotalPlates     int           `json:"total_plates"`
	Plates          []PlateResult `json:"plates"`
	AnnotatedImage  string        `json:"annotated_image"`
}

// HealthResponse mô tả trạng thái của Go Backend và AI Worker
type HealthResponse struct {
	Status       string `json:"status"`
	Backend      string `json:"backend"`
	GoVersion    string `json:"go_version"`
	WorkerStatus string `json:"worker_status"`
	WorkerURL    string `json:"worker_url"`
	Device       string `json:"device"`
}

// SampleItem đại diện cho một ảnh mẫu có sẵn để người dùng test nhanh
type SampleItem struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Category    string `json:"category"`
	Description string `json:"description"`
	Filename    string `json:"filename"`
	URL         string `json:"url"`
}

// JsonRecognizeRequest hỗ trợ gửi ảnh qua JSON Base64
type JsonRecognizeRequest struct {
	ImageBase64 string `json:"image_base64"`
}
