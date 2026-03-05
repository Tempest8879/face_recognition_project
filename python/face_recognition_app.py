"""
Face Recognition App - Python frontend with C++ backend
========================================================
Uses:
  - MTCNN (facenet-pytorch) for face detection
  - ArcFace ONNX model (onnxruntime) for face recognition (512-dim embeddings)
  - face_recognition (dlib) for 68-point landmarks (anti-spoofing)
  - face_processor_cpp (C++ via pybind11) for fast cosine similarity matching
  - OpenCV for image display and camera capture
  - Anti-spoofing with challenge-response for liveness verification

Dependencies: facenet-pytorch, onnxruntime, face_recognition,
              torch, opencv-python, numpy

ArcFace model:
  Place the ArcFace ONNX model (w600k_r50.onnx) in the models/ directory.
  Download from: https://drive.google.com/file/d/15wKZ9Ub03B_24lxmGMQG-wHtaI0Et88W/view?usp=drive_link
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

# Ensure CUDA libraries are discoverable before importing onnxruntime
_cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",  # Windows
    "/usr/local/cuda/lib64",                                           # Linux
    "/usr/local/cuda/bin",                                             # Linux
]
for _cuda_bin in _cuda_paths:
    if os.path.isdir(_cuda_bin):
        if _cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _cuda_bin + os.pathsep + os.environ.get("PATH", "")
        # Windows 10+ / Python 3.8+: required for DLL search
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_cuda_bin)

# MTCNN for face detection (required)
try:
    from facenet_pytorch import MTCNN as MTCNNDetector
    import torch
    MTCNN_AVAILABLE = True
    print("[OK] MTCNN (facenet-pytorch) loaded successfully!")
except ImportError:
    MTCNN_AVAILABLE = False
    print("[ERROR] MTCNN (facenet-pytorch) is required. Install: pip install facenet-pytorch")

# ArcFace via ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("[OK] ONNX Runtime loaded successfully!")
except ImportError:
    ONNX_AVAILABLE = False
    print("[WARNING] ONNX Runtime not available. Face recognition is DISABLED.")
    print("          Install with: pip install onnxruntime  (or onnxruntime-gpu)")

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

        # MTCNN face detector (required)
        if not MTCNN_AVAILABLE:
            raise RuntimeError("MTCNN is required. Install: pip install facenet-pytorch torch")

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

        # ArcFace ONNX recognition model
        self.use_arcface = False
        self.ort_session = None
        self.arcface_input_name = None
        self.arcface_input_size = (112, 112)  # standard ArcFace input
        if ONNX_AVAILABLE:
            model_path = self._find_arcface_model()
            if model_path:
                try:
                    # Try GPU first, fall back to CPU if CUDA libs are missing
                    available = ort.get_available_providers()
                    self.ort_session = None
                    if 'CUDAExecutionProvider' in available:
                        try:
                            self.ort_session = ort.InferenceSession(
                                model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                            )
                            # Verify CUDA actually loaded (not just listed)
                            if 'CUDAExecutionProvider' not in self.ort_session.get_providers():
                                self.ort_session = None
                        except Exception:
                            print("[WARNING] CUDA provider listed but failed to load (missing CUDA/cuDNN libs).")
                            print("          Falling back to CPU.")
                            self.ort_session = None
                    if self.ort_session is None:
                        self.ort_session = ort.InferenceSession(
                            model_path, providers=['CPUExecutionProvider']
                        )
                    self.arcface_input_name = self.ort_session.get_inputs()[0].name
                    input_shape = self.ort_session.get_inputs()[0].shape
                    if len(input_shape) == 4:
                        self.arcface_input_size = (input_shape[2], input_shape[3])
                    active = self.ort_session.get_providers()
                    provider_label = "GPU (CUDA)" if 'CUDAExecutionProvider' in active else "CPU"
                    print(f"[OK] ArcFace ONNX model loaded: {os.path.basename(model_path)}")
                    print(f"     Input: {self.arcface_input_name} {input_shape}, Provider: {provider_label}")
                    self.use_arcface = True
                except Exception as e:
                    print(f"[WARNING] ArcFace ONNX init failed: {e}")
                    print("          Face recognition is DISABLED.")
            else:
                print("[WARNING] No ArcFace ONNX model found in models/ directory.")
                print("          Face recognition is DISABLED (detection + anti-spoofing only).")
                print("          To enable: place a .onnx model in models/")

        self._load_known_faces()

        # Anti-spoofing with challenge-response
        if ANTI_SPOOF_AVAILABLE:
            self.anti_spoof = AntiSpoofing()
            print("[OK] Anti-spoofing initialized (Gate1: photo → Gate2: video)")
        else:
            self.anti_spoof = None

    def _detect_faces_mtcnn(self, rgb_frame):
        """Detect faces using MTCNN.

        Returns:
            locations: list of (top, right, bottom, left) tuples
            landmarks_5point: list of (5, 2) ndarrays (eye, eye, nose, mouth, mouth)
        """
        try:
            boxes, probs, landmarks = self.mtcnn.detect(rgb_frame, landmarks=True)

            if boxes is None:
                return [], []

            h, w = rgb_frame.shape[:2]
            locations = []
            valid_landmarks = []
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < 0.9:
                    continue
                x1, y1, x2, y2 = [int(b) for b in box]
                locations.append((
                    max(0, y1), min(w, x2), min(h, y2), max(0, x1)
                ))
                if landmarks is not None and i < len(landmarks):
                    valid_landmarks.append(landmarks[i])
                else:
                    valid_landmarks.append(None)

            return locations, valid_landmarks
        except Exception:
            return [], []

    @staticmethod
    def _find_arcface_model():
        """Search for an ArcFace .onnx model file in standard locations."""
        search_dirs = ['models', 'data/models', '.']
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith('.onnx'):
                    return os.path.join(d, f)
        return None

    @staticmethod
    def _align_face(bgr_frame, landmark_5point, target_size=(112, 112)):
        """Align a face using 5-point landmarks (standard ArcFace alignment).

        Uses a similarity transform to map detected landmarks to the
        canonical ArcFace reference positions for 112x112 input.
        """
        # ArcFace canonical reference landmarks for 112x112
        ref_pts = np.array([
            [38.2946, 51.6963],   # left eye
            [73.5318, 51.5014],   # right eye
            [56.0252, 71.7366],   # nose tip
            [41.5493, 92.3655],   # left mouth
            [70.7299, 92.2041],   # right mouth
        ], dtype=np.float32)

        src_pts = np.array(landmark_5point, dtype=np.float32)
        if src_pts.shape != (5, 2):
            return None

        # Estimate similarity transform (rotation + scale + translation)
        # Using first 3 points for cv2.getAffineTransform is unstable;
        # instead use a least-squares fit via cv2.estimateAffinePartial2D
        tform, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts)
        if tform is None:
            return None

        aligned = cv2.warpAffine(bgr_frame, tform, target_size, borderValue=0)
        return aligned

    def _get_arcface_embedding(self, bgr_frame, landmark_5point):
        """Get ArcFace 512-dim embedding via ONNX Runtime."""
        try:
            aligned = self._align_face(bgr_frame, landmark_5point, self.arcface_input_size)
            if aligned is None:
                return None

            # Preprocess: BGR to RGB, HWC to CHW, normalize to [-1, 1], add batch dim
            img = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
            img = (img / 127.5) - 1.0
            img = np.transpose(img, (2, 0, 1))  # CHW
            img = np.expand_dims(img, axis=0)    # NCHW

            outputs = self.ort_session.run(None, {self.arcface_input_name: img})
            embedding = outputs[0].flatten()

            # L2-normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 1e-10:
                embedding = embedding / norm

            return embedding
        except Exception:
            return None

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
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"  [-] Could not read: {image_path}")
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Detect face with MTCNN
            locs, landmarks = self._detect_faces_mtcnn(image_rgb)

            if not locs:
                print(f"  [-] No face found in: {image_path}")
                continue

            # Get ArcFace embedding (no dlib fallback — dimensions are incompatible)
            encoding = None
            if self.use_arcface and landmarks[0] is not None:
                encoding = self._get_arcface_embedding(image_bgr, landmarks[0])

            if encoding is None:
                if not self.use_arcface:
                    print(f"  [-] Skipped (ArcFace model not loaded): {image_path}")
                else:
                    print(f"  [-] Could not encode face in: {image_path}")
                continue

            if USE_CPP_BACKEND:
                self.processor.add_known_face(name, encoding.tolist())
            else:
                self.known_names.append(name)
                self.known_encodings.append(encoding)
            print(f"  [+] Loaded: {name}")

        count = self.processor.known_face_count() if USE_CPP_BACKEND else len(self.known_names)
        print(f"[INFO] {count} known face(s) registered\n")

    def recognize_image(self, image_path):
        """Recognize faces in a single image file."""
        print(f"[INFO] Processing: {image_path}")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return []
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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
        blink_confirmed = [False]          # True once a blink is detected

        # --- Capture thread ---
        def _capture():
            while not stop_event.is_set():
                ret = cap.grab()
                if ret:
                    ret, frame = cap.retrieve()
                    if ret:
                        frame = cv2.flip(frame, 1)  # mirror horizontally (selfie-view)
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
                    # Resize to half resolution for detection + encoding
                    # 0.5x on 640x480 = 320x240, enough detail for reliable encodings
                    frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                    # Convert BGR to RGB AND ensure C-contiguous memory
                    rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    rgb_small = np.ascontiguousarray(rgb_small)

                    # Detection on 1/2 res
                    locs, lm5s = self._detect_faces_mtcnn(rgb_small)
                    
                    # 2. FACE TRACKING
                    num_faces = len(locs)
                    has_face_now = num_faces > 0

                    # Reset anti-spoof when face disappears or reappears
                    if has_face_now and not face_present_last_frame[0]:
                        if self.anti_spoof:
                            self.anti_spoof.reset()
                            print("[AUTO] New face detected. Please blink to verify.")
                        blink_confirmed[0] = False
                    elif not has_face_now and face_present_last_frame[0]:
                        if self.anti_spoof:
                            self.anti_spoof.reset()
                            print("[AUTO] Face lost. System reset.")
                        blink_confirmed[0] = False

                    # 3. MULTI-FACE PAUSE: halt challenge if >1 face
                    if self.anti_spoof and self.anti_spoof.challenge_active():
                        if num_faces > 1 and not self.anti_spoof.challenge_paused():
                            print("[WARN] Multiple faces detected! Pausing challenge...")
                            self.anti_spoof.pause_challenge()
                        elif num_faces == 1 and self.anti_spoof.challenge_paused():
                            print("[INFO] Single face restored. Resuming challenge...")
                            self.anti_spoof.unpause_challenge()

                    face_present_last_frame[0] = has_face_now

                    # Scale landmarks to full resolution for ArcFace alignment
                    full_lm5s = [lm * 2 if lm is not None else None for lm in lm5s]

                    # Get ArcFace embeddings (no dlib fallback — dimensions are incompatible)
                    encodings = []
                    for i in range(len(locs)):
                        enc = None
                        if self.use_arcface and i < len(full_lm5s) and full_lm5s[i] is not None:
                            enc = self._get_arcface_embedding(frame, full_lm5s[i])
                        encodings.append(enc)

                    new_results = []
                    for i, loc in enumerate(locs):
                        enc = encodings[i]
                        if enc is not None:
                            match = self._match_face(enc)
                        else:
                            match = {"name": "Unknown", "similarity": 0.0, "confidence": 0.0}
                        t, r, b, l = [v * 2 for v in loc]
                        new_results.append((match, (t, r, b, l)))

                    # Anti-spoofing with dlib landmarks on full-res
                    new_spoof = None
                    if self.anti_spoof and new_results:
                        _, first_loc = new_results[0]
                        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        lm_list = face_recognition.face_landmarks(
                            rgb_full, face_locations=[first_loc]
                        )
                        lm_dict = lm_list[0] if lm_list else None
                        new_spoof = self.anti_spoof.evaluate(
                            frame, first_loc, lm_dict
                        )

                        # Check for first blink to unlock recognition
                        if not blink_confirmed[0] and new_spoof:
                            if len(self.anti_spoof.blink_detector.blink_timestamps) > 0:
                                blink_confirmed[0] = True
                                print("[OK] Blink detected! Identity verified.")
                    elif self.anti_spoof and not has_face_now:
                        # No face → push a zeroed-out result to clear HUD
                        new_spoof = {
                            'is_live': False,
                            'photo_passed': False,
                            'video_passed': False,
                            'gate1_passed': False,
                            'gate2_passed': False, 'gate2_score': 0.0,
                            'photo_score': 0.0,
                            'video_score': 0.0,
                            'liveness_score': 0.0,
                            'blink_signal': 0.0,
                            'challenge_signal': 0.0,
                            'consistency_score': 0.0,
                            'challenge_result': None,
                            'no_face': True,
                        }
                    
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
        display_matches = []
        display_spoof = None
        display_is_live = True
        smooth_factor = 0.35
        fps_deque = deque(maxlen=120)

        # Robust face tracking with spatial matching
        LABEL_CONFIRM_FRAMES = 8   # frames needed to change a label
        MAX_TRACK_AGE = 15         # frames before a lost track is removed

        class FaceTrack:
            """Track a single face across frames by spatial proximity."""
            def __init__(self, box, name, confidence):
                self.box = list(box)            # smoothed display box
                self.target_box = list(box)     # latest raw box
                self.stable_name = name         # currently displayed name
                self.stable_conf = confidence   # currently displayed confidence
                self.cand_name = name           # candidate for next label
                self.cand_count = 1             # consecutive frames of candidate
                self.age = 0                    # frames since last matched

            def center(self):
                t, r, b, l = self.box
                return ((t + b) / 2.0, (l + r) / 2.0)

            def update(self, box, name, confidence):
                self.target_box = list(box)
                self.age = 0
                # Update label candidate
                if name == self.cand_name:
                    self.cand_count += 1
                else:
                    self.cand_name = name
                    self.cand_count = 1
                # Promote candidate to stable if confirmed
                if self.cand_count >= LABEL_CONFIRM_FRAMES:
                    self.stable_name = self.cand_name
                    self.stable_conf = confidence
                elif self.stable_name == name:
                    self.stable_conf = confidence

            def smooth_box(self, factor):
                for j in range(4):
                    self.box[j] += factor * (self.target_box[j] - self.box[j])

        face_tracks = []  # list of FaceTrack

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

            # Spatial face tracking with label smoothing
            if new_results:
                new_boxes = [list(loc) for _, loc in new_results]
                raw_matches = [m for m, _ in new_results]

                # Match new detections to existing tracks by closest center distance
                used_tracks = set()
                used_dets = set()
                assignments = {}  # det_idx -> track_idx

                for di, nb in enumerate(new_boxes):
                    ct = ((nb[0] + nb[2]) / 2.0, (nb[3] + nb[1]) / 2.0)
                    best_ti, best_dist = -1, float('inf')
                    for ti, trk in enumerate(face_tracks):
                        if ti in used_tracks:
                            continue
                        tc = trk.center()
                        d = ((ct[0] - tc[0])**2 + (ct[1] - tc[1])**2)**0.5
                        if d < best_dist:
                            best_dist = d
                            best_ti = ti
                    if best_ti >= 0 and best_dist < 200:  # max pixel distance
                        assignments[di] = best_ti
                        used_tracks.add(best_ti)
                        used_dets.add(di)

                # Update matched tracks
                for di, ti in assignments.items():
                    face_tracks[ti].update(
                        new_boxes[di], raw_matches[di]["name"], raw_matches[di]["confidence"]
                    )

                # Create new tracks for unmatched detections
                for di in range(len(new_boxes)):
                    if di not in used_dets:
                        face_tracks.append(FaceTrack(
                            new_boxes[di], raw_matches[di]["name"], raw_matches[di]["confidence"]
                        ))

                # Age unmatched tracks
                for ti, trk in enumerate(face_tracks):
                    if ti not in used_tracks and ti < len(face_tracks):
                        trk.age += 1

                # Remove old tracks
                face_tracks[:] = [t for t in face_tracks if t.age < MAX_TRACK_AGE]

                # Smooth boxes and build display lists
                for trk in face_tracks:
                    trk.smooth_box(smooth_factor)

                display_boxes = [trk.box for trk in face_tracks]
                display_matches = [{
                    "name": trk.stable_name,
                    "confidence": trk.stable_conf
                } for trk in face_tracks]

            elif not new_results and not face_results_shared:
                # Age all tracks when no results
                for trk in face_tracks:
                    trk.age += 1
                face_tracks[:] = [t for t in face_tracks if t.age < MAX_TRACK_AGE]
                if not face_tracks:
                    display_boxes = []
                    display_matches = []

            if local_spoof is not None:
                display_spoof = local_spoof
                display_is_live = local_spoof["is_live"]
                # If no face, clear HUD and boxes
                if local_spoof.get('no_face'):
                    display_boxes = []
                    display_matches = []

            # --- Draw face boxes ---
            local_blink_ok = blink_confirmed[0]
            for idx, box in enumerate(display_boxes):
                if idx >= len(display_matches):
                    break
                match = display_matches[idx]
                top, right, bottom, left = [int(round(v)) for v in box]

                # Suppress identity until blink verified
                if not local_blink_ok:
                    display_name = "Verifying..."
                    color = (0, 200, 255)  # orange
                elif match["name"] != "Unknown" and display_is_live:
                    display_name = match["name"]
                    color = (0, 255, 0)
                elif match["name"] != "Unknown":
                    display_name = match["name"]
                    color = (0, 165, 255)
                else:
                    display_name = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                label = f"{display_name} ({match['confidence']:.0%})" if local_blink_ok else display_name
                if self.anti_spoof and display_spoof:
                    if display_is_live:
                        status = "REAL"
                    elif display_spoof.get('photo_passed') and display_spoof.get('video_passed'):
                        status = "VERIFYING"
                    elif display_spoof.get('photo_passed'):
                        status = "VIDEO CHECK"
                    else:
                        status = "PHOTO CHECK"
                    label += f" [{status}]"

                cv2.putText(frame, label, (left, max(top - 10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # --- HUD overlay ---
            if self.anti_spoof and display_spoof:
                y = 25

                # No face → show minimal "NO FACE" HUD
                if display_spoof.get('no_face'):
                    cv2.putText(frame, "NO FACE DETECTED", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)
                    y += 22
                    cv2.putText(frame, "System reset - waiting...", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1)
                else:
                    # Gate 1 — Blink verification
                    g1_ok = display_spoof.get('gate1_passed', False)
                    g1_color = (0, 255, 0) if g1_ok else (0, 0, 255)
                    cv2.putText(frame, f"GATE1  {'BLINKED' if g1_ok else 'BLINK!'}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, g1_color, 1)
                    y += 18
                    cv2.putText(frame,
                        f"  EAR: {display_spoof.get('ear', 0):.3f} thr={display_spoof.get('ear_threshold', 0):.3f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
                    y += 16

                    # Gate 2 — Challenge
                    g2_ok = display_spoof.get('gate2_passed', False)
                    g2_color = (0, 255, 0) if g2_ok else (
                        (0, 200, 255) if g1_ok else (100, 100, 100))
                    g2_label = 'PASS' if g2_ok else ('PEND' if g1_ok else 'WAIT')
                    cv2.putText(frame, f"GATE2  {g2_label} {display_spoof.get('gate2_score', 0):.2f}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, g2_color, 1)
                    y += 18
                    cv2.putText(frame,
                        f"  Chall: {display_spoof.get('challenge_signal', 0):.2f}  3D: {display_spoof.get('consistency_score', 0):.2f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
                    y += 16

                    # Combined
                    cv2.putText(frame, f"REAL: {display_spoof['liveness_score']:.2f}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                               (0, 255, 0) if display_is_live else (0, 0, 255), 1)

                cr = display_spoof.get('challenge_result')
                if cr:
                    self._draw_challenge_overlay(frame, cr)

            # --- Blink prompt overlay (before blink confirmed) ---
            if not blink_confirmed[0] and display_boxes:
                h_frame, w_frame = frame.shape[:2]
                prompt = "Please BLINK to verify"
                text_size = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                text_x = (w_frame - text_size[0]) // 2
                text_y = 45
                # Dark background for readability
                cv2.rectangle(frame, (text_x - 10, text_y - 30),
                             (text_x + text_size[0] + 10, text_y + 10),
                             (0, 0, 0), -1)
                cv2.putText(frame, prompt, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

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

        elif cr.get('recentering') and cr['is_active']:
            # --- Recentering: waiting for neutral pose ---
            progress = f"{cr['challenges_passed']}/{cr['num_challenges']}"

            roi = frame[h - 80:h, 0:w]
            cv2.addWeighted(roi, 0.4, roi, 0, 0, roi)

            center_text = "Look straight at the camera"
            text_size = cv2.getTextSize(center_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, center_text, (text_x, h - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            cv2.putText(frame, f"Challenge {progress} | Recalibrating...",
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
        """Match a face encoding against known faces using cosine similarity."""
        if USE_CPP_BACKEND:
            result = self.processor.find_best_match(encoding.tolist())
            return {
                "name": result.name,
                "similarity": result.similarity,
                "confidence": result.confidence
            }
        else:
            if not self.known_encodings:
                return {"name": "Unknown", "similarity": 0.0, "confidence": 0.0}

            # Cosine similarity
            enc_norm = np.linalg.norm(encoding)
            similarities = []
            for known in self.known_encodings:
                known_norm = np.linalg.norm(known)
                if enc_norm < 1e-10 or known_norm < 1e-10:
                    similarities.append(0.0)
                else:
                    similarities.append(float(np.dot(encoding, known) / (enc_norm * known_norm)))

            best_idx = int(np.argmax(similarities))
            best_sim = similarities[best_idx]
            threshold = 0.4

            # Sigmoid confidence centered on threshold
            k = 15.0
            conf = 1.0 / (1.0 + np.exp(-k * (best_sim - threshold)))

            if best_sim >= threshold and conf >= 0.75:
                name = self.known_names[best_idx]
            else:
                name = "Unknown"

            return {
                "name": name,
                "similarity": best_sim,
                "confidence": float(conf)
            }

    def _process_frame(self, image, display=False, window_name="Result"):
        """Process a single RGB image, find and recognize faces."""
        locs, lm5s = self._detect_faces_mtcnn(image)

        results = []
        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if display else None
        image_bgr = None

        for i, (top, right, bottom, left) in enumerate(locs):
            enc = None
            if self.use_arcface and i < len(lm5s) and lm5s[i] is not None:
                if image_bgr is None:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                enc = self._get_arcface_embedding(image_bgr, lm5s[i])

            if enc is not None:
                match = self._match_face(enc)
            else:
                match = {"name": "Unknown", "similarity": 0.0, "confidence": 0.0}
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
