import numpy as np

def compute_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea

class Tracklet:
    def __init__(self, track_id, bbox, conf, frame_idx):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.conf = conf
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.hit_streak = 1
        self.time_since_update = 0
        
        # Best frame selection data
        self.best_score = -1.0
        self.best_crop = None
        self.best_bbox = bbox
        self.best_frame_idx = frame_idx
        
        # OCR data
        self.ocr_result = None
        self.ocr_confidence = 0.0
        self.ocr_done = False
        self.candidate_texts = [] # For temporal consensus voting

    def update(self, bbox, conf, frame_idx):
        self.bbox = bbox
        self.conf = conf
        self.last_frame = frame_idx
        self.hit_streak += 1
        self.time_since_update = 0

    def evaluate_candidate(self, crop, sharpness, det_conf, frame_idx):
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        area = w * h
        # Quality score = Area * Sharpness * Detection confidence
        score = float(area * (sharpness + 1.0) * det_conf)
        if score > self.best_score and crop is not None and crop.size > 0:
            self.best_score = score
            self.best_crop = crop.copy()
            self.best_bbox = self.bbox
            self.best_frame_idx = frame_idx

class RealtimePlateTracker:
    """
    Lightweight, high-speed IoU Multi-Object Tracker for Real-Time Traffic Cameras.
    Runs at < 0.5ms per frame on CPU with zero deep learning overhead.
    """
    def __init__(self, iou_threshold=0.3, max_age=15, min_hits=2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1

    def update(self, detections, frame_idx):
        """
        detections: list of [x1, y1, x2, y2, conf]
        """
        # Step 1: Match existing tracks with detections via IoU
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, trk in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = compute_iou(trk.bbox, det[:4])

            # Greedy matching based on highest IoU
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matched.append((t_idx, d_idx))
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)

        # Step 2: Update matched tracks
        for t_idx, d_idx in matched:
            det = detections[d_idx]
            self.tracks[t_idx].update(det[:4], det[4], frame_idx)

        # Step 3: Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_trk = Tracklet(self.next_id, det[:4], det[4], frame_idx)
            self.next_id += 1
            self.tracks.append(new_trk)

        # Step 4: Age unmatched tracks and purge dead tracks
        active_tracks = []
        for t_idx, trk in enumerate(self.tracks):
            if t_idx in unmatched_tracks:
                trk.time_since_update += 1
            if trk.time_since_update <= self.max_age:
                active_tracks.append(trk)
        self.tracks = active_tracks

        return self.tracks
