import sys
import numpy as np
from tracker import ByteTracker, calculate_iou

def run_tests():
    print("CodeAlpha AI Internship - Computer Vision Test Suite")
    # Test IoU
    assert calculate_iou([10,10,50,50], [10,10,50,50]) == 1.0
    assert calculate_iou([10,10,50,50], [100,100,150,150]) == 0.0
    print("[PASS] IoU calculation accurate.")

    # Test ID Stability
    tracker = ByteTracker(max_age=15, min_hits=2, iou_threshold=0.3)
    ids_car, ids_person = [], []
    for f in range(25):
        dets = [
            {"box": [100 + f*5, 200, 160 + f*5, 240], "score": 0.88, "class_name": "car", "class_id": 2},
            {"box": [400 - f*4, 280, 450 - f*4, 330], "score": 0.92, "class_name": "person", "class_id": 0}
        ]
        tracked = tracker.update(dets)
        for t in tracked:
            if t["class_name"] == "car": ids_car.append(t["track_id"])
            if t["class_name"] == "person": ids_person.append(t["track_id"])
    assert len(set(ids_car)) == 1, "Car ID switched!"
    assert len(set(ids_person)) == 1, "Person ID switched!"
    print(f"[PASS] Stable IDs verified: Car #{ids_car[0]}, Person #{ids_person[0]}. Zero random ID switches.")
    print("ALL TESTS PASSED SUCCESSFULLY (3/3)")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
