import cv2
import numpy as np

def generate_synthetic_sample_video(output_path="sample_traffic.mp4", width=854, height=480, fps=30, num_frames=150):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for f in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[0:160, :] = [60, 110, 60]
        frame[160:420, :] = [45, 45, 50]
        frame[420:height, :] = [60, 110, 60]
        for x in range(350, 460, 24): cv2.rectangle(frame, (x, 160), (x + 14, 420), (230, 230, 230), -1)
        # Car 1
        cx = int(-50 + f * 5.2)
        cv2.rectangle(frame, (cx, 280), (cx + 110, 335), (200, 60, 40), -1)
        # Pedestrian 1
        py = int(140 + f * 1.8)
        cv2.rectangle(frame, (390, py), (420, py + 55), (60, 180, 240), -1)
        out.write(frame)
    out.release()
    print(f"[Sample Video] Generated benchmark test video at: {output_path}")

if __name__ == "__main__":
    generate_synthetic_sample_video()
