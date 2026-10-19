import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def calculate_iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_test + area_gt - intersection
    return float(intersection / union) if union > 0 else 0.0

def iou_batch(bb_test, bb_gt):
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.empty((len(bb_test), len(bb_gt)))
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - intersection
    return np.where(union > 0, intersection / union, 0.0)

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array([bbox[0] + w/2.0, bbox[1] + h/2.0, max(1e-4, w*h), max(1e-4, w/float(h if h != 0 else 1e-4))]).reshape((4, 1))

def convert_x_to_bbox(x):
    w = np.sqrt(max(1e-4, x[2, 0] * x[3, 0]))
    h = max(1e-4, x[2, 0] / float(w if w != 0 else 1e-4))
    return np.array([x[0, 0] - w/2.0, x[1, 0] - h/2.0, x[0, 0] + w/2.0, x[1, 0] + h/2.0])

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox, class_id=0, class_name="object", score=1.0):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.R = np.eye(4); self.R[2:, 2:] *= 10.0
        self.Q = np.eye(7); self.Q[4:, 4:] *= 0.01; self.Q[:4, :4] *= 1.0
        self.P = np.eye(7) * 10.0; self.P[4:, 4:] *= 1000.0
        self.x = np.zeros((7, 1))
        self.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 1; self.age = 0
        self.class_id = class_id; self.class_name = class_name
        self.score = score; self.last_detection_score = score
        self.trajectory = [(float((bbox[0]+bbox[2])/2.0), float((bbox[1]+bbox[3])/2.0))]

    def update(self, bbox, score=None, class_name=None, class_id=None):
        self.time_since_update = 0; self.hits += 1
        if score is not None: self.score = 0.7 * self.score + 0.3 * score; self.last_detection_score = score
        if class_name: self.class_name = class_name
        if class_id is not None: self.class_id = class_id
        z = convert_bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(7) - np.dot(K, self.H), self.P)
        self.trajectory.append((float((bbox[0]+bbox[2])/2.0), float((bbox[1]+bbox[3])/2.0)))
        if len(self.trajectory) > 60: self.trajectory.pop(0)

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1; self.time_since_update += 1
        return convert_x_to_bbox(self.x)

    def get_state(self):
        return convert_x_to_bbox(self.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0: return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    if len(detections) == 0: return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))
    iou_matrix = iou_batch(detections, trackers)
    if HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.column_stack((row_ind, col_ind))
    else:
        matched_indices = []
        for r in range(iou_matrix.shape[0]):
            best_c = np.argmax(iou_matrix[r])
            if iou_matrix[r, best_c] >= iou_threshold:
                matched_indices.append((r, best_c))
        matched_indices = np.array(matched_indices) if matched_indices else np.empty((0, 2), dtype=int)
    matches = []
    unmatched_det = list(range(len(detections)))
    unmatched_trk = list(range(len(trackers)))
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] >= iou_threshold:
            matches.append(m)
            if m[0] in unmatched_det: unmatched_det.remove(m[0])
            if m[1] in unmatched_trk: unmatched_trk.remove(m[1])
    return (np.array(matches) if matches else np.empty((0, 2), dtype=int)), np.array(unmatched_det), np.array(unmatched_trk)

class ByteTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age; self.min_hits = min_hits; self.iou_threshold = iou_threshold
        self.trackers = []; self.frame_count = 0; self.total_unique_ids = 0

    def update(self, detections: list):
        self.frame_count += 1
        trks = np.array([trk.predict() for trk in self.trackers]) if self.trackers else np.empty((0, 4))
        det_boxes = np.array([d["box"] for d in detections]) if detections else np.empty((0, 4))
        det_scores = np.array([d.get("score", 1.0) for d in detections]) if detections else np.empty((0,))
        high_mask = det_scores >= 0.45
        matches_a, unmatched_det_a, unmatched_trk_a = associate_detections_to_trackers(det_boxes[high_mask], trks, self.iou_threshold)
        high_indices = np.where(high_mask)[0]
        for m in matches_a:
            orig_d = high_indices[m[0]]
            self.trackers[m[1]].update(detections[orig_d]["box"], score=detections[orig_d]["score"], class_name=detections[orig_d]["class_name"], class_id=detections[orig_d]["class_id"])
        for i in unmatched_det_a:
            orig_d = high_indices[i]
            d = detections[orig_d]
            trk = KalmanBoxTracker(d["box"], class_id=d.get("class_id", 0), class_name=d.get("class_name", "object"), score=d.get("score", 1.0))
            self.trackers.append(trk)
            self.total_unique_ids = max(self.total_unique_ids, trk.id)
        ret = []
        for i in reversed(range(len(self.trackers))):
            trk = self.trackers[i]
            if trk.time_since_update < 1 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append({"track_id": trk.id, "box": [round(float(c), 1) for c in trk.get_state()], "score": round(float(trk.last_detection_score), 2), "class_id": trk.class_id, "class_name": trk.class_name, "trajectory": list(trk.trajectory), "hits": trk.hits, "age": trk.age})
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return ret
