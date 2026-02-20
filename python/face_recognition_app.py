"""
Face Recognition App - Python frontend with C++ backend
========================================================
Uses:
  - face_recognition (Python/dlib) for face detection, encoding & 68-point landmarks
  - MTCNN (facenet-pytorch) for fast face detection (preferred)
  - face_processor_cpp (C++ via pybind11) for fast face matching & texture analysis
  - OpenCV for image display and camera capture
  - Anti-spoofing with challenge-response for maximum liveness accuracy

Dependencies: face_recognition, facenet-pytorch, torch, opencv-python, numpy
Removed: MediaPipe (replaced by dlib landmarks from face_recognition)
"""

import os
import sys
import glob
import time
import threading
from collections import deque
import numpy as np
import cv2
import face_recognition

# MTCNN (facenet-pytorch) for improved face detection
try:
    from facenet_pytorch import MTCNN as MTCNNDetector
    import torch
    MTCNN_AVAILABLE = True
    print("[OK] MTCNN (facenet-pytorch) loaded successfully!")
except ImportError:
    MTCNN_AVAILABLE = False
    print("[INFO] MTCNN not available. Using face_recognition (dlib HOG) detector.")

# Anti-spoofing module
try:
    from anti_spoof import AntiSpoofing
    ANTI_SPOOF_AVAILABLE = True
    print("[OK] Anti-spoofing module loaded successfully!")
except ImportError:
    ANTI_SPOOF_AVAILABLE = False

# C++ backend
try:
    import face_processor_cpp
    print("[OK] C++ face_processor_cpp module loaded successfully!")
    USE_CPP_BACKEND = True
except ImportError:
    print("[WARNING] C++ module not found. Using pure Python fallback.")
    print("          Run build.bat to compile the C++ module.")
    USE_CPP_BACKEND = False


class FaceRecognitionApp:
    """Face recognition application combining Python and C++ backends."""

    def __init__(self, known_faces_dir="data/known_faces"):
        self.known_faces_dir = known_faces_dir

        if USE_CPP_BACKEND:
            self.processor = face_processor_cpp.FaceProcessor()
            print(f"[C++] Backend initialized")
        else:
            self.known_names = []
            self.known_encodings = []
            print(f"[Python] Backend initialized")

        self._load_known_faces()

        # MTCNN face detector (preferred, faster + more accurate)
        self.use_mtcnn = MTCNN_AVAILABLE
        if self.use_mtcnn:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.mtcnn = MTCNNDetector(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=device,
                keep_all=True,
            )
            print(f"[OK] MTCNN initialized on {device}")
        else:
            self.mtcnn = None

        # Anti-spoofing with challenge-response
        if ANTI_SPOOF_AVAILABLE:
            self.anti_spoof = AntiSpoofing()
            print("[OK] Anti-spoofing initialized (Gate1: photo → Gate2: video)")
        else:
            self.anti_spoof = None

    def _detect_faces_mtcnn(self, rgb_frame):
        """Detect faces using MTCNN. Falls back to dlib HOG (fast)."""
        if self.mtcnn is None:
            return face_recognition.face_locations(rgb_frame)

        try:
            boxes, probs = self.mtcnn.detect(rgb_frame)

            if boxes is None:
                return []

            h, w = rgb_frame.shape[:2]
            locations = []
            for box, prob in zip(boxes, probs):
                if prob < 0.9:
                    continue
                x1, y1, x2, y2 = [int(b) for b in box]
                locations.append((
                    max(0, y1), min(w, x2), min(h, y2), max(0, x1)
                ))

            return locations
        except Exception:
            return face_recognition.face_locations(rgb_frame)

    def _load_known_faces(self):
        """Load and encode all known face images from the data directory."""
        if not os.path.exists(self.known_faces_dir):
            print(f"[INFO] Known faces directory not found: {self.known_faces_dir}")
            print(f"       Add face images (jpg/png) named after the person.")
            return

        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.known_faces_dir, ext)))

        if not image_files:
            print(f"[INFO] No face images found in {self.known_faces_dir}")
            return

        print(f"\n[INFO] Loading {len(image_files)} known face(s)...")
        for image_path in image_files:
            name = os.path.splitext(os.path.basename(image_path))[0]
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                encoding = encodings[0]
                if USE_CPP_BACKEND:
                    self.processor.add_known_face(name, encoding.tolist())
                else:
                    self.known_names.append(name)
                    self.known_encodings.append(encoding)
                print(f"  [+] Loaded: {name}")
            else:
                print(f"  [-] No face found in: {image_path}")

        count = self.processor.known_face_count() if USE_CPP_BACKEND else len(self.known_names)
        print(f"[INFO] {count} known face(s) registered\n")

    def recognize_image(self, image_path):
        """Recognize faces in a single image file."""
        print(f"[INFO] Processing: {image_path}")
        image = face_recognition.load_image_file(image_path)
        return self._process_frame(image, display=True, window_name="Face Recognition")

    def recognize_webcam(self):
        """Real-time face recognition from webcam.

        Three-thread architecture for maximum FPS:
          - Capture thread: grabs frames from camera (never blocks display)
          - Worker thread:  detection + recognition + liveness (heavy work)
          - Main thread:    renders display at max speed with smooth interpolation
        """
        print("[INFO] Starting webcam... Press 'q' to quit, 'c' to start challenge.")

        # Try DirectShow on Windows for lower-latency capture
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # --- Shared state ---
        lock = threading.Lock()
        stop_event = threading.Event()
        start_challenge = threading.Event()
        face_present_last_frame = [False]

        captured_frame = [None]            # latest camera frame
        face_results_shared = []           # list of (match_dict, (t,r,b,l))
        spoof_result_shared = [None]       # latest anti-spoof result
        worker_busy = [False]

        # --- Capture thread ---
        def _capture():
            while not stop_event.is_set():
                ret = cap.grab()
                if ret:
                    ret, frame = cap.retrieve()
                    if ret:
                        with lock:
                            captured_frame[0] = frame
                else:
                    time.sleep(0.001)

        # --- Worker thread ---
        def _worker():
            while not stop_event.is_set():
                if start_challenge.is_set() and self.anti_spoof:
                    start_challenge.clear()
                    self.anti_spoof.start_challenge()

                with lock:
                    frame = captured_frame[0]
                    if frame is None or worker_busy[0]:
                        frame = None
                    else:
                        frame = frame.copy()
                        worker_busy[0] = True

                if frame is None:
                    time.sleep(0.003)
                    continue

                try:
                    # Resize or slice first
                    frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    
                    # Convert BGR to RGB AND ensure C-contiguous memory
                    rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    rgb_small = np.ascontiguousarray(rgb_small) 
                    
                    # Detection on 1/4 res
                    locs = self._detect_faces_mtcnn(rgb_small)

                    # 1. Detection on 1/4 res
                    frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    rgb_small = np.ascontiguousarray(rgb_small) # Fixes your previous TypeError

                    locs = self._detect_faces_mtcnn(rgb_small)
                    
                    # 2. FACE TRACKING
                    num_faces = len(locs)
                    has_face_now = num_faces > 0

                    # Reset anti-spoof when face reappears after leaving
                    if has_face_now and not face_present_last_frame[0]:
                        if self.anti_spoof:
                            self.anti_spoof.reset()
                            print("[AUTO] New face detected. Liveness check started.")

                    # 3. MULTI-FACE PAUSE: halt challenge if >1 face
                    if self.anti_spoof and self.anti_spoof.challenge_active():
                        if num_faces > 1 and not self.anti_spoof.challenge_paused():
                            print("[WARN] Multiple faces detected! Pausing challenge...")
                            self.anti_spoof.pause_challenge()
                        elif num_faces == 1 and self.anti_spoof.challenge_paused():
                            print("[INFO] Single face restored. Resuming challenge...")
                            self.anti_spoof.unpause_challenge()

                    face_present_last_frame[0] = has_face_now

                    encs = face_recognition.face_encodings(rgb_small, locs)

                    new_results = []
                    for enc, loc in zip(encs, locs):
                        match = self._match_face(enc)
                        t, r, b, l = [v * 4 for v in loc]
                        new_results.append((match, (t, r, b, l)))

                    # Anti-spoofing with dlib landmarks on full-res
                    new_spoof = None
                    if self.anti_spoof and new_results:
                        _, first_loc = new_results[0]
                        rgb_full = frame[:, :, ::-1]  # BGR->RGB without copy
                        lm_list = face_recognition.face_landmarks(
                            rgb_full, face_locations=[first_loc]
                        )
                        lm_dict = lm_list[0] if lm_list else None
                        new_spoof = self.anti_spoof.evaluate(
                            frame, first_loc, lm_dict
                        )
                    
                finally:
                    with lock:
                        face_results_shared[:] = new_results
                        spoof_result_shared[0] = new_spoof
                        worker_busy[0] = False

        # Start threads
        threading.Thread(target=_capture, daemon=True).start()
        threading.Thread(target=_worker, daemon=True).start()

        # Wait for first frame
        while captured_frame[0] is None:
            time.sleep(0.01)

        # --- Display state (main thread only) ---
        display_boxes = []
        target_boxes = []
        display_matches = []
        display_spoof = None
        display_is_live = True
        smooth_factor = 0.35
        fps_deque = deque(maxlen=120)

        # --- Main display loop ---
        while True:
            with lock:
                frame = captured_frame[0]
                if frame is None:
                    continue
                frame = frame.copy()
                new_results = list(face_results_shared)
                local_spoof = spoof_result_shared[0]

            # FPS
            now = time.time()
            fps_deque.append(now)
            while fps_deque and now - fps_deque[0] > 1.0:
                fps_deque.popleft()
            fps = len(fps_deque)

            # Smooth bounding box interpolation
            if new_results:
                target_boxes = [list(loc) for _, loc in new_results]
                display_matches = [m for m, _ in new_results]
                if len(display_boxes) != len(target_boxes):
                    display_boxes = [list(b) for b in target_boxes]
                else:
                    for i in range(len(display_boxes)):
                        for j in range(4):
                            display_boxes[i][j] += smooth_factor * (
                                target_boxes[i][j] - display_boxes[i][j]
                            )
            elif not new_results and not face_results_shared:
                display_boxes = []
                display_matches = []

            if local_spoof is not None:
                display_spoof = local_spoof
                display_is_live = local_spoof["is_live"]

            # --- Draw face boxes ---
            for idx, box in enumerate(display_boxes):
                if idx >= len(display_matches):
                    break
                match = display_matches[idx]
                top, right, bottom, left = [int(round(v)) for v in box]

                if match["name"] != "Unknown" and display_is_live:
                    color = (0, 255, 0)
                elif match["name"] != "Unknown":
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                label = f"{match['name']} ({match['confidence']:.0%})"
                if self.anti_spoof and display_spoof:
                    if display_is_live:
                        status = "LIVE"
                    elif display_spoof.get('photo_passed'):
                        status = "CHECKING VIDEO"
                    else:
                        status = "CHECKING LIVENESS"
                    label += f" [{status}]"

                cv2.putText(frame, label, (left, max(top - 10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # --- HUD overlay ---
            if self.anti_spoof and display_spoof:
                y = 25
                # Gate 1 — Photo / Liveness
                photo_ok = display_spoof.get('photo_passed', False)
                g1_color = (0, 255, 0) if photo_ok else (0, 0, 255)
                cv2.putText(frame, f"PHOTO  {'PASS' if photo_ok else 'FAIL'} {display_spoof.get('photo_score', 0):.2f}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, g1_color, 1)
                y += 18
                for text in (
                    f"  Blink: {display_spoof['blink_signal']:.2f}",
                    f"  Mouth: {display_spoof.get('mouth_signal', 0):.2f}",
                    f"  Move:  {display_spoof['movement_signal']:.2f}",
                ):
                    cv2.putText(frame, text, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
                    y += 16

                # Gate 2 — Video
                video_ok = display_spoof.get('video_passed', False)
                g2_color = (0, 255, 0) if video_ok else (0, 0, 255)
                cv2.putText(frame, f"VIDEO  {'PASS' if video_ok else 'FAIL'} {display_spoof.get('video_score', 0):.2f}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, g2_color, 1)
                y += 18
                for text in (
                    f"  Chall: {display_spoof.get('challenge_signal', 0):.2f}",
                    f"  Screen:{display_spoof.get('screen_signal', 1.0):.2f}",
                    f"  Depth: {display_spoof.get('consistency_score', 0):.2f}",
                ):
                    cv2.putText(frame, text, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
                    y += 16

                # Combined
                cv2.putText(frame, f"LIVE: {display_spoof['liveness_score']:.2f}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                           (0, 255, 0) if display_is_live else (0, 0, 255), 1)

                cr = display_spoof.get('challenge_result')
                if cr:
                    self._draw_challenge_overlay(frame, cr)

            # FPS + controls
            cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 100, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "'q' quit | 'c' challenge",
                       (frame.shape[1] - 250, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            cv2.imshow("Face Recognition (press 'q' to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and self.anti_spoof:
                start_challenge.set()

        # Cleanup
        stop_event.set()
        time.sleep(0.1)
        cap.release()
        if self.anti_spoof:
            self.anti_spoof.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam stopped.")

    def _draw_challenge_overlay(self, frame, cr):
        """Draw challenge-response status on frame."""
        h, w = frame.shape[:2]

        if cr.get('paused') and cr['is_active']:
            # --- Paused: multiple faces detected ---
            progress = f"{cr['challenges_passed']}/{cr['num_challenges']}"

            roi = frame[h - 80:h, 0:w]
            cv2.addWeighted(roi, 0.3, roi, 0, 0, roi)

            warn_text = "PAUSED - Multiple faces detected"
            text_size = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, warn_text, (text_x, h - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Challenge {progress} | Waiting for single face...",
                       (text_x, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        elif cr['is_active'] and cr['current_challenge']:
            instruction = cr['current_instruction']
            progress = f"{cr['challenges_passed']}/{cr['num_challenges']}"
            timer = f"{cr['time_remaining']:.1f}s"

            # Darken bottom banner (ROI blend, no full-frame copy)
            roi = frame[h - 80:h, 0:w]
            cv2.addWeighted(roi, 0.4, roi, 0, 0, roi)

            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, instruction, (text_x, h - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Challenge {progress} | Time: {timer}",
                       (text_x, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        elif cr['is_complete']:
            if cr['passed']:
                text, color = "LIVENESS VERIFIED", (0, 255, 0)
            else:
                text, color = "CHALLENGE FAILED", (0, 0, 255)

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, text, ((w - text_size[0]) // 2, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _match_face(self, encoding):
        """Match a face encoding against known faces using C++ or Python backend."""
        if USE_CPP_BACKEND:
            result = self.processor.find_best_match(encoding.tolist())
            return {
                "name": result.name,
                "distance": result.distance,
                "confidence": result.confidence
            }
        else:
            if not self.known_encodings:
                return {"name": "Unknown", "distance": 1.0, "confidence": 0.0}

            distances = face_recognition.face_distance(self.known_encodings, encoding)
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]
            tolerance = 0.6

            if best_distance <= tolerance:
                name = self.known_names[best_idx]
            else:
                name = "Unknown"

            return {
                "name": name,
                "distance": float(best_distance),
                "confidence": max(0.0, 1.0 - (best_distance / tolerance))
            }

    def _process_frame(self, image, display=False, window_name="Result"):
        """Process a single RGB image, find and recognize faces."""
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        results = []
        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if display else None

        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            match = self._match_face(encoding)
            results.append(match)

            if display:
                color = (0, 255, 0) if match["name"] != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_image, (left, top), (right, bottom), color, 2)
                label = f"{match['name']} ({match['confidence']:.0%})"
                cv2.rectangle(display_image, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                cv2.putText(display_image, label, (left + 6, bottom - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if display and display_image is not None:
            cv2.imshow(window_name, display_image)
            print("[INFO] Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)

    app = FaceRecognitionApp(known_faces_dir="data/known_faces")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            results = app.recognize_image(image_path)
            for r in results:
                print(f"  -> {r['name']} (confidence: {r['confidence']:.2%})")
        else:
            print(f"[ERROR] File not found: {image_path}")
    else:
        app.recognize_webcam()
