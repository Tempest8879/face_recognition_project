"""
Anti-Spoofing / Liveness Detection Module
==========================================
Multi-layer anti-spoofing system using only MTCNN and CNN for maximum accuracy:
  1. Eye blink detection (dlib 68-point landmarks via face_recognition)
  2. Head micro-movement detection (dlib landmarks)
  3. Mouth micro-movement detection (dlib landmarks)
  4. Head-pose challenge-response with 3D geometric consistency
  5. FFT screen detection (frequency-domain pixel-grid / moiré analysis)

Dependencies: face_recognition (dlib), opencv-python, numpy
"""

import time
import numpy as np
import cv2
import face_recognition


# =============================================================================
# Blink Detector (dlib 68-point EAR)
# =============================================================================

class BlinkDetector:
    """Detect eye blinks using dlib 68-point landmarks EAR (Eye Aspect Ratio).

    Uses left_eye and right_eye points from face_recognition.face_landmarks().
    Each eye has 6 points in standard dlib 68-point ordering.
    """

    def __init__(self, ear_threshold=0.2, consec_frames=2, time_window=5.0):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.time_window = time_window

        self.below_threshold_count = 0
        self.blink_timestamps = []
        self.last_ear = 0.0

    def compute_ear(self, eye_points):
        """Compute Eye Aspect Ratio from 6 dlib eye landmark points.

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        p1=outer corner, p2=upper-outer, p3=upper-inner,
        p4=inner corner, p5=lower-inner, p6=lower-outer
        """
        pts = [np.array(p, dtype=float) for p in eye_points]
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        if C < 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)

    def update(self, left_eye, right_eye):
        """Process one frame's eye landmarks. Returns blink signal [0.0, 1.0].

        Args:
            left_eye: list of 6 (x, y) tuples from face_recognition
            right_eye: list of 6 (x, y) tuples from face_recognition
        """
        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        self.last_ear = avg_ear

        now = time.time()

        if avg_ear < self.ear_threshold:
            self.below_threshold_count += 1
        else:
            if self.below_threshold_count >= self.consec_frames:
                self.blink_timestamps.append(now)
            self.below_threshold_count = 0

        cutoff = now - self.time_window
        self.blink_timestamps = [t for t in self.blink_timestamps if t > cutoff]

        return self.get_signal()

    def get_signal(self):
        """Return blink signal: 0.0 (no blinks) to 1.0 (sufficient blinks)."""
        count = len(self.blink_timestamps)
        if count == 0:
            return 0.0
        elif count == 1:
            return 0.5
        else:
            return 1.0

    def reset(self):
        self.below_threshold_count = 0
        self.blink_timestamps.clear()
        self.last_ear = 0.0


# =============================================================================
# Movement Detector (dlib landmarks)
# =============================================================================

class MovementDetector:
    """Detect head micro-movements by tracking dlib facial landmark displacement.

    Measures non-rigid motion: real faces have independent eye/nose/chin movement,
    while photos held by hand show only rigid (global) motion.
    """

    def __init__(self, movement_threshold=0.002, history_size=15):
        self.movement_threshold = movement_threshold
        self.history_size = history_size
        self.landmark_history = []

    def update(self, landmarks_dict):
        """Process one frame's face_recognition landmarks dict."""
        key_points = self._extract_key_points(landmarks_dict)
        if key_points is None:
            return self.get_signal()

        self.landmark_history.append(key_points)
        if len(self.landmark_history) > self.history_size:
            self.landmark_history.pop(0)

        return self.get_signal()

    def _extract_key_points(self, lm):
        """Extract 5 key tracking points from face_recognition landmarks dict.

        Uses: nose tip, chin center, left eye outer corner,
              right eye outer corner, nose bridge top
        """
        try:
            points = [
                np.array(lm['nose_tip'][2], dtype=float),
                np.array(lm['chin'][8], dtype=float),
                np.array(lm['left_eye'][0], dtype=float),
                np.array(lm['right_eye'][3], dtype=float),
                np.array(lm['nose_bridge'][0], dtype=float),
            ]
            return np.array(points)
        except (KeyError, IndexError):
            return None

    def get_signal(self):
        """Compute movement signal from landmark history.

        Measures independent (non-rigid) landmark motion rather than raw
        displacement. When someone holds a photo/phone, ALL landmarks move
        together (rigid translation). A real face has independent motion:
        eyes move differently from nose from chin. By subtracting the mean
        displacement (centroid motion) we isolate non-rigid deformation.
        """
        if len(self.landmark_history) < 3:
            return 0.0

        non_rigid_scores = []
        for i in range(1, len(self.landmark_history)):
            diff = self.landmark_history[i] - self.landmark_history[i - 1]
            global_motion = np.mean(diff, axis=0)
            local_diff = diff - global_motion
            local_mags = np.linalg.norm(local_diff, axis=1)
            non_rigid_scores.append(float(np.std(local_mags)))

        avg_non_rigid = np.mean(non_rigid_scores)
        signal = min(1.0, avg_non_rigid / self.movement_threshold)
        return float(signal)

    def reset(self):
        self.landmark_history.clear()


# =============================================================================
# Mouth Movement Detector (dlib MAR)
# =============================================================================

class MouthMovementDetector:
    """Detect mouth micro-movements using dlib landmarks MAR (Mouth Aspect Ratio).

    Tracks mouth opening/closing over time. Real faces exhibit involuntary
    mouth micro-movements (breathing, subtle lip adjustments); photos/screens do not.
    Uses outer lip points from face_recognition.face_landmarks().
    """

    def __init__(self, mar_movement_threshold=0.003, history_size=20):
        self.mar_movement_threshold = mar_movement_threshold
        self.history_size = history_size
        self.mar_history = []
        self.last_mar = 0.0

    def compute_mar(self, landmarks_dict):
        """Compute Mouth Aspect Ratio from face_recognition landmarks.

        Uses outer lip points:
          top_lip[3] = dlib 51 (upper center), bottom_lip[3] = dlib 57 (lower center)
          top_lip[0] = dlib 48 (left corner), top_lip[6] = dlib 54 (right corner)

        MAR = mean(vertical_distances) / horizontal_distance
        """
        try:
            top_lip = landmarks_dict['top_lip']
            bottom_lip = landmarks_dict['bottom_lip']

            left_corner = np.array(top_lip[0], dtype=float)
            right_corner = np.array(top_lip[6], dtype=float)

            # Three vertical pairs across the mouth (outer lip points)
            upper_pts = [np.array(top_lip[i], dtype=float) for i in [2, 3, 4]]
            lower_pts = [np.array(bottom_lip[i], dtype=float) for i in [4, 3, 2]]

            vertical_dists = [np.linalg.norm(u - l) for u, l in zip(upper_pts, lower_pts)]
            horizontal_dist = np.linalg.norm(left_corner - right_corner)

            if horizontal_dist < 1e-6:
                return 0.0
            return float(np.mean(vertical_dists) / horizontal_dist)
        except (KeyError, IndexError):
            return 0.0

    def update(self, landmarks_dict):
        """Process one frame's landmarks. Returns mouth movement signal [0.0, 1.0]."""
        mar = self.compute_mar(landmarks_dict)
        self.last_mar = mar
        self.mar_history.append(mar)

        if len(self.mar_history) > self.history_size:
            self.mar_history.pop(0)

        return self.get_signal()

    def get_signal(self):
        """Compute mouth movement signal from MAR variance over time.

        Real faces show continuous micro-fluctuations in MAR;
        photos/screens show near-zero variance.
        """
        if len(self.mar_history) < 3:
            return 0.0

        deltas = [abs(self.mar_history[i] - self.mar_history[i - 1])
                  for i in range(1, len(self.mar_history))]
        avg_delta = np.mean(deltas)
        signal = min(1.0, avg_delta / self.mar_movement_threshold)
        return float(signal)

    def reset(self):
        self.mar_history.clear()
        self.last_mar = 0.0


# =============================================================================
# Screen Detector (FFT frequency-domain analysis)
# =============================================================================

class ScreenDetector:
    """Detect screens / displays via frequency-domain analysis of the face ROI.

    When a camera films a screen the captured image contains artifacts that
    real skin does not produce:

      1. Moiré patterns from camera↔screen pixel-grid interference
      2. Energy concentration along horizontal / vertical frequency axes
         (screen pixels are arranged in a rectilinear grid)
      3. Discrete mid-frequency peaks (pixel-pitch harmonics)
      4. Different Laplacian sharpness profile vs organic skin texture

    Multiple metrics are fused and smoothed over a sliding window so that
    single-frame noise does not cause false positives.

    Uses only OpenCV + NumPy — no extra dependencies.
    """

    ANALYSIS_SIZE = 128          # face ROI is resized to this for FFT

    def __init__(self, history_size=30):
        self.history_size = history_size
        self.score_history = []
        self.last_score = 0.0
        self.last_metrics = {}

    # ----- public API -----

    def update(self, frame_bgr, face_location):
        """Analyse one frame's face ROI for screen artifacts.

        Args:
            frame_bgr:     Full BGR frame from OpenCV capture.
            face_location: (top, right, bottom, left) at frame resolution.

        Returns:
            float  signal [0.0, 1.0] — 1.0 = real face, 0.0 = screen.
        """
        top, right, bottom, left = (int(v) for v in face_location)
        h, w = frame_bgr.shape[:2]
        top, left = max(0, top), max(0, left)
        bottom, right = min(h, bottom), min(w, right)

        roi = frame_bgr[top:bottom, left:right]
        if roi.size == 0 or roi.shape[0] < 32 or roi.shape[1] < 32:
            return self.get_signal()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.ANALYSIS_SIZE, self.ANALYSIS_SIZE))

        score, metrics = self._analyze(gray)

        self.last_score = score
        self.last_metrics = metrics
        self.score_history.append(score)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)

        return self.get_signal()

    def get_signal(self):
        """Median-smoothed screen-detection signal over recent frames."""
        if not self.score_history:
            return 1.0                                   # assume real until proven otherwise
        if len(self.score_history) < 3:
            return float(np.mean(self.score_history))
        return float(np.median(self.score_history))

    def reset(self):
        self.score_history.clear()
        self.last_score = 0.0
        self.last_metrics = {}

    # ----- internal -----

    def _analyze(self, gray):
        """Run frequency + spatial analysis on a 128×128 grayscale face ROI.

        Returns (score, metrics_dict).
            score  : float [0.0, 1.0]  — 1.0 = real skin, 0.0 = screen.
            metrics: dict of intermediate values (for HUD / debugging).
        """
        sz = self.ANALYSIS_SIZE
        gray_f = gray.astype(np.float64)

        # ---- 2D FFT ----
        fshift = np.fft.fftshift(np.fft.fft2(gray_f))
        magnitude = np.abs(fshift)

        cy, cx = sz // 2, sz // 2
        y_idx, x_idx = np.ogrid[:sz, :sz]
        r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        max_r = float(cx)

        total_energy = np.sum(magnitude ** 2) + 1e-10

        # 1. Spectral flatness (Wiener entropy)
        #    Real skin texture → flatter (noise-like) spectrum.
        #    Screen pixel grid  → peaked spectrum.
        log_mag = np.log(magnitude + 1e-10)
        geo_mean = np.exp(np.mean(log_mag))
        arith_mean = np.mean(magnitude) + 1e-10
        spectral_flatness = float(geo_mean / arith_mean)

        # 2. Axis energy ratio
        #    Screen pixels sit on a rectilinear grid → strong H / V axis
        #    energy in the FFT.  Exclude the DC neighbourhood.
        band = 2
        h_axis = np.zeros((sz, sz), dtype=bool)
        h_axis[cy - band:cy + band + 1, :] = True
        v_axis = np.zeros((sz, sz), dtype=bool)
        v_axis[:, cx - band:cx + band + 1] = True
        dc = r < 3
        axis_mask = (h_axis | v_axis) & ~dc

        axis_energy = float(np.sum(magnitude[axis_mask] ** 2) / total_energy)

        # 3. Mid-frequency peak ratio
        #    Screens produce sharp harmonics at pixel-pitch multiples;
        #    organic skin has a smooth spectral roll-off.
        mid_mask = (r > max_r * 0.15) & (r < max_r * 0.75)
        mid_mags = magnitude[mid_mask]
        if mid_mags.size > 0:
            median_mid = np.median(mid_mags)
            peak_ratio = float(np.sum(mid_mags > 4.0 * median_mid) / mid_mags.size) \
                         if median_mid > 0 else 0.0
        else:
            peak_ratio = 0.0

        # 4. Laplacian variance (spatial-domain sharpness)
        #    Screen pixels create fine sharp edges that boost Laplacian.
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # ---- Per-metric scoring (each ∈ [0, 1], 1 = real face) ----

        # Flatness: real skin typically > 0.012, screens < 0.008
        flatness_sig = float(min(1.0, spectral_flatness / 0.015))

        # Axis energy: real < 0.08, screens > 0.12
        axis_sig = float(1.0 - min(1.0, max(0.0, (axis_energy - 0.06)) / 0.10))

        # Peak ratio: real < 0.01, screens > 0.02
        peak_sig = float(1.0 - min(1.0, peak_ratio / 0.03))

        # Laplacian: very high variance → sharp pixel edges (screen)
        lap_sig = float(max(0.3, 1.0 - max(0.0, lap_var - 2000) / 5000)
                        if lap_var > 2000 else 1.0)

        # ---- Fusion ----
        score = (0.30 * flatness_sig +
                 0.30 * axis_sig +
                 0.25 * peak_sig +
                 0.15 * lap_sig)

        metrics = {
            'spectral_flatness': spectral_flatness,
            'axis_energy':       axis_energy,
            'peak_ratio':        peak_ratio,
            'laplacian_var':     lap_var,
            'flatness_sig':      flatness_sig,
            'axis_sig':          axis_sig,
            'peak_sig':          peak_sig,
            'lap_sig':           lap_sig,
        }

        return float(np.clip(score, 0.0, 1.0)), metrics


# =============================================================================
# Challenge-Response Detector (interactive liveness verification)
# =============================================================================

class ChallengeResponseDetector:
    """Head-pose challenge-response with 3D geometric consistency analysis.

    Fixed sequence: TURN LEFT → TURN RIGHT → LOOK UP → LOOK DOWN

    At each completed pose, a normalized landmark feature vector is captured.
    After all poses, euclidean distances and cosine similarities between
    opposing pose pairs (left↔right, up↔down) are computed. A real 3D face
    exhibits perspective-dependent geometry shifts (parallax) that a flat
    video or photo cannot replicate.

    The final signal combines challenge completion with a 3D consistency
    score derived from these geometric comparisons.
    """

    CHALLENGES = [
        ('TURN_LEFT', 'Please turn head LEFT'),
        ('TURN_RIGHT', 'Please turn head RIGHT'),
        ('LOOK_UP', 'Please LOOK UP'),
        ('LOOK_DOWN', 'Please LOOK DOWN'),
    ]

    def __init__(self, num_challenges=4, challenge_timeout=10.0,
                 euc_threshold_lr=0.08, euc_threshold_ud=0.06,
                 cos_dissim_threshold=0.02):
        self.num_challenges = num_challenges
        self.challenge_timeout = challenge_timeout

        # 3D consistency thresholds
        self.euc_threshold_lr = euc_threshold_lr    # left↔right euclidean
        self.euc_threshold_ud = euc_threshold_ud    # up↔down euclidean
        self.cos_dissim_threshold = cos_dissim_threshold  # 1 - cosine_sim

        self.current_challenge = None
        self.current_instruction = ""
        self.challenge_start_time = None
        self.challenges_passed = 0
        self.challenges_failed = 0
        self.is_active = False
        self.is_complete = False
        self.passed = False
        self._baseline = {}
        self._remaining = []

        # Multi-face pause state
        self._paused = False

        # 3D consistency analysis state
        self._pose_snapshots = {}       # {challenge_id: feature_vector}
        self._consistency_score = 0.0
        self._euc_lr = 0.0             # euclidean distance left↔right
        self._euc_ud = 0.0             # euclidean distance up↔down
        self._cos_lr = 1.0             # cosine similarity left↔right
        self._cos_ud = 1.0             # cosine similarity up↔down

    def start(self):
        """Start a new challenge-response sequence (fixed order)."""
        self.challenges_passed = 0
        self.challenges_failed = 0
        self.is_active = True
        self.is_complete = False
        self.passed = False
        self._paused = False
        self._pose_snapshots = {}
        self._consistency_score = 0.0
        self._euc_lr = 0.0
        self._euc_ud = 0.0
        self._cos_lr = 1.0
        self._cos_ud = 1.0
        self._remaining = list(self.CHALLENGES[:self.num_challenges])
        self._next_challenge()

    def _next_challenge(self):
        """Move to the next challenge or finish."""
        if self.challenges_passed >= self.num_challenges:
            self.is_active = False
            self.is_complete = True
            self.passed = True
            self.current_challenge = None
            self.current_instruction = "VERIFIED"
            self._consistency_score = self._compute_3d_consistency()
            return

        if not self._remaining:
            self.is_active = False
            self.is_complete = True
            self.passed = self.challenges_passed >= self.num_challenges
            self.current_challenge = None
            return

        challenge_id, instruction = self._remaining.pop(0)
        self.current_challenge = challenge_id
        self.current_instruction = instruction
        self.challenge_start_time = time.time()
        self._baseline = {}

    def pause(self):
        """Pause the challenge sequence (multiple faces detected).

        Resets the current challenge's baseline and timer so it must be
        re-performed from scratch when unpaused, but preserves progress
        on already-completed challenges and their pose snapshots.
        """
        if not self.is_active or self._paused:
            return
        self._paused = True
        self._baseline = {}

    def unpause(self):
        """Resume after pause (back to single face).

        Restarts the current challenge with a fresh timer and baseline.
        """
        if not self.is_active or not self._paused:
            return
        self._paused = False
        self.challenge_start_time = time.time()
        self._baseline = {}

    def update(self, landmarks_dict):
        """Check if current challenge is completed.

        Args:
            landmarks_dict: face_recognition.face_landmarks() result dict

        Returns:
            dict with challenge status and 3D consistency metrics
        """
        if self._paused:
            return self.get_result()

        if not self.is_active or self.current_challenge is None:
            return self.get_result()

        elapsed = time.time() - self.challenge_start_time
        if elapsed > self.challenge_timeout:
            self.challenges_failed += 1
            if self._remaining:
                self._next_challenge()
            else:
                self.is_active = False
                self.is_complete = True
                self.passed = False
                self.current_challenge = None
                self.current_instruction = "FAILED - Time expired"
            return self.get_result()

        passed = self._check_challenge(landmarks_dict)

        if passed:
            # Capture landmark snapshot for 3D geometric analysis
            features = self._extract_pose_features(landmarks_dict)
            if features is not None:
                self._pose_snapshots[self.current_challenge] = features

            self.challenges_passed += 1
            self._next_challenge()

        return self.get_result()

    # ------------------------------------------------------------------
    # Pose detection
    # ------------------------------------------------------------------

    def _check_challenge(self, landmarks_dict):
        """Evaluate whether the current head-pose challenge was performed."""
        if self.current_challenge == 'TURN_LEFT':
            try:
                nose = np.array(landmarks_dict['nose_tip'][2], dtype=float)
                chin = landmarks_dict['chin']
                left_jaw = np.array(chin[0], dtype=float)
                right_jaw = np.array(chin[16], dtype=float)
                face_width = np.linalg.norm(right_jaw - left_jaw)
                face_center_x = (left_jaw[0] + right_jaw[0]) / 2.0
                if face_width > 1:
                    offset = (nose[0] - face_center_x) / face_width
                    return offset < -0.03
            except (KeyError, IndexError):
                pass

        elif self.current_challenge == 'TURN_RIGHT':
            try:
                nose = np.array(landmarks_dict['nose_tip'][2], dtype=float)
                chin = landmarks_dict['chin']
                left_jaw = np.array(chin[0], dtype=float)
                right_jaw = np.array(chin[16], dtype=float)
                face_width = np.linalg.norm(right_jaw - left_jaw)
                face_center_x = (left_jaw[0] + right_jaw[0]) / 2.0
                if face_width > 1:
                    offset = (nose[0] - face_center_x) / face_width
                    return offset > 0.03
            except (KeyError, IndexError):
                pass

        elif self.current_challenge == 'LOOK_UP':
            try:
                nose_tip = np.array(landmarks_dict['nose_tip'][2], dtype=float)
                bridge_top = np.array(landmarks_dict['nose_bridge'][0], dtype=float)
                chin_bottom = np.array(landmarks_dict['chin'][8], dtype=float)
                upper = abs(nose_tip[1] - bridge_top[1])
                lower = abs(chin_bottom[1] - nose_tip[1])
                ratio = upper / (upper + lower + 1e-6)
                if 'vertical_ratio_baseline' not in self._baseline:
                    self._baseline['vertical_ratio_baseline'] = ratio
                elif ratio < self._baseline['vertical_ratio_baseline'] - 0.04:
                    return True
            except (KeyError, IndexError):
                pass

        elif self.current_challenge == 'LOOK_DOWN':
            try:
                nose_tip = np.array(landmarks_dict['nose_tip'][2], dtype=float)
                bridge_top = np.array(landmarks_dict['nose_bridge'][0], dtype=float)
                chin_bottom = np.array(landmarks_dict['chin'][8], dtype=float)
                upper = abs(nose_tip[1] - bridge_top[1])
                lower = abs(chin_bottom[1] - nose_tip[1])
                ratio = upper / (upper + lower + 1e-6)
                if 'vertical_ratio_baseline' not in self._baseline:
                    self._baseline['vertical_ratio_baseline'] = ratio
                elif ratio > self._baseline['vertical_ratio_baseline'] + 0.04:
                    return True
            except (KeyError, IndexError):
                pass

        return False

    # ------------------------------------------------------------------
    # 3D geometric consistency analysis
    # ------------------------------------------------------------------

    def _extract_pose_features(self, landmarks_dict):
        """Extract a normalized geometric feature vector from landmarks.

        8 features, all normalized by face width or height so they are
        scale-invariant and capture only the 3D perspective geometry:

          0. left_eye → nose_tip  / face_width
          1. right_eye → nose_tip / face_width
          2. nose_tip → left_jaw  / face_width
          3. nose_tip → right_jaw / face_width
          4. nose_tip horizontal offset from face center / face_width
          5. nose_tip vertical offset from face center   / face_height
          6. left_eye → left_jaw  / face_width
          7. right_eye → right_jaw / face_width

        A real 3D face produces perspective-dependent changes in these
        ratios when the head rotates; a flat image does not.
        """
        try:
            nose_tip = np.array(landmarks_dict['nose_tip'][2], dtype=float)
            chin_bottom = np.array(landmarks_dict['chin'][8], dtype=float)
            left_jaw = np.array(landmarks_dict['chin'][0], dtype=float)
            right_jaw = np.array(landmarks_dict['chin'][16], dtype=float)
            left_eye_outer = np.array(landmarks_dict['left_eye'][0], dtype=float)
            right_eye_outer = np.array(landmarks_dict['right_eye'][3], dtype=float)
            bridge_top = np.array(landmarks_dict['nose_bridge'][0], dtype=float)

            face_width = np.linalg.norm(right_jaw - left_jaw)
            face_height = np.linalg.norm(bridge_top - chin_bottom)

            if face_width < 1 or face_height < 1:
                return None

            center_x = (left_jaw[0] + right_jaw[0]) / 2.0
            center_y = (bridge_top[1] + chin_bottom[1]) / 2.0

            features = np.array([
                np.linalg.norm(left_eye_outer - nose_tip) / face_width,
                np.linalg.norm(right_eye_outer - nose_tip) / face_width,
                np.linalg.norm(nose_tip - left_jaw) / face_width,
                np.linalg.norm(nose_tip - right_jaw) / face_width,
                (nose_tip[0] - center_x) / face_width,
                (nose_tip[1] - center_y) / face_height,
                np.linalg.norm(left_eye_outer - left_jaw) / face_width,
                np.linalg.norm(right_eye_outer - right_jaw) / face_width,
            ])
            return features
        except (KeyError, IndexError):
            return None

    def _compute_3d_consistency(self):
        """Compute 3D consistency score from opposing pose snapshots.

        Compares feature vectors of opposing poses using both euclidean
        distance and cosine similarity:

          euclidean distance  — high for real faces (geometry changes)
          cosine dissimilarity — high for real faces (ratio directions shift)

        A flat image (photo / video replay) shows near-identical feature
        vectors across poses because no true parallax occurs.

        Returns:
            float [0.0, 1.0]  — 1.0 = strong 3D parallax (real face)
        """
        scores = []

        # Left ↔ Right comparison
        l_feat = self._pose_snapshots.get('TURN_LEFT')
        r_feat = self._pose_snapshots.get('TURN_RIGHT')
        if l_feat is not None and r_feat is not None:
            self._euc_lr = float(np.linalg.norm(l_feat - r_feat))
            dot = np.dot(l_feat, r_feat)
            norms = np.linalg.norm(l_feat) * np.linalg.norm(r_feat) + 1e-8
            self._cos_lr = float(dot / norms)

            euc_score = min(1.0, self._euc_lr / self.euc_threshold_lr)
            dissim_score = min(1.0, (1.0 - self._cos_lr) / self.cos_dissim_threshold)
            scores.append(0.5 * euc_score + 0.5 * dissim_score)

        # Up ↔ Down comparison
        u_feat = self._pose_snapshots.get('LOOK_UP')
        d_feat = self._pose_snapshots.get('LOOK_DOWN')
        if u_feat is not None and d_feat is not None:
            self._euc_ud = float(np.linalg.norm(u_feat - d_feat))
            dot = np.dot(u_feat, d_feat)
            norms = np.linalg.norm(u_feat) * np.linalg.norm(d_feat) + 1e-8
            self._cos_ud = float(dot / norms)

            euc_score = min(1.0, self._euc_ud / self.euc_threshold_ud)
            dissim_score = min(1.0, (1.0 - self._cos_ud) / self.cos_dissim_threshold)
            scores.append(0.5 * euc_score + 0.5 * dissim_score)

        return float(np.mean(scores)) if scores else 0.0

    # ------------------------------------------------------------------
    # Result / reset
    # ------------------------------------------------------------------

    def get_result(self):
        time_remaining = 0.0
        if self.challenge_start_time and self.is_active:
            time_remaining = max(0, self.challenge_timeout - (time.time() - self.challenge_start_time))

        # Signal factors in 3D consistency: passing alone is not enough;
        # the geometry must also check out.
        if self.passed:
            signal = max(0.4, self._consistency_score)
        else:
            signal = 0.0

        return {
            'is_active': self.is_active,
            'is_complete': self.is_complete,
            'passed': self.passed,
            'paused': self._paused,
            'current_challenge': self.current_challenge,
            'current_instruction': self.current_instruction,
            'challenges_passed': self.challenges_passed,
            'num_challenges': self.num_challenges,
            'signal': signal,
            'time_remaining': time_remaining,
            'consistency_score': self._consistency_score,
            'euc_lr': self._euc_lr,
            'euc_ud': self._euc_ud,
            'cos_lr': self._cos_lr,
            'cos_ud': self._cos_ud,
        }

    def reset(self):
        self.current_challenge = None
        self.current_instruction = ""
        self.challenge_start_time = None
        self.challenges_passed = 0
        self.challenges_failed = 0
        self.is_active = False
        self.is_complete = False
        self.passed = False
        self._paused = False
        self._baseline = {}
        self._remaining = []
        self._pose_snapshots = {}
        self._consistency_score = 0.0
        self._euc_lr = 0.0
        self._euc_ud = 0.0
        self._cos_lr = 1.0
        self._cos_ud = 1.0


# =============================================================================
# Anti-Spoofing Orchestrator
# =============================================================================

class AntiSpoofing:
    """Two-gate anti-spoofing / liveness detection system.

    Gate 1 — Photo / Liveness (passive, continuous):
      Proves the face is a live person, not a static photo.
        • Eye blink detection  (45%)
        • Mouth micro-movement (30%)
        • Head micro-movement  (25%)
      Must reach photo_threshold to pass.

    Gate 2 — Video detection (active challenge + FFT):
      Proves the feed is not a video replay on a screen.
        • Head-pose challenge with 3D consistency (55%)
        • FFT screen / moiré detection             (45%)
      Must reach video_threshold to pass.

    Final decision: BOTH gates must pass for ≥ consec_live_required
    consecutive evaluations before is_live = True.

    Uses only MTCNN and CNN (dlib) — no MediaPipe dependency.
    """

    def __init__(self, photo_threshold=0.40, video_threshold=0.35,
                 num_challenges=4, challenge_timeout=10.0,
                 ear_threshold=0.2, consec_frames=2, blink_time_window=5.0,
                 movement_threshold=0.005, movement_history=15,
                 mar_movement_threshold=0.003, mar_history_size=20):

        # Gate thresholds
        self.photo_threshold = photo_threshold
        self.video_threshold = video_threshold

        # Sub-detectors
        self.blink_detector = BlinkDetector(ear_threshold, consec_frames, blink_time_window)
        self.movement_detector = MovementDetector(movement_threshold, movement_history)
        self.mouth_detector = MouthMovementDetector(mar_movement_threshold, mar_history_size)
        self.challenge_detector = ChallengeResponseDetector(num_challenges, challenge_timeout)
        self.screen_detector = ScreenDetector()

        # State
        self.last_liveness_score = 0.0
        self.last_is_live = False
        self.last_signals = {
            "blink": 0.0, "movement": 0.0,
            "mouth": 0.0, "challenge": 0.0,
            "screen": 1.0,
        }

        # Temporal consistency
        self._consec_live_count = 0
        self._consec_live_required = 5

    def start_challenge(self):
        """Start the interactive challenge-response sequence."""
        self.challenge_detector.start()

    def challenge_active(self):
        """Check if a challenge is currently active."""
        return self.challenge_detector.is_active

    def challenge_paused(self):
        """Check if the challenge is currently paused (multi-face)."""
        return self.challenge_detector._paused

    def pause_challenge(self):
        """Pause the challenge (multiple faces detected).

        Resets the current challenge's baseline/timer but preserves
        progress on already-completed poses.
        """
        self.challenge_detector.pause()

    def unpause_challenge(self):
        """Resume challenge after pause (single face restored)."""
        self.challenge_detector.unpause()

    def evaluate(self, frame_bgr, face_location, landmarks_dict=None):
        """Run all anti-spoofing checks on one frame for one face.

        Two-gate evaluation:
          Gate 1 (photo):  blink + mouth + movement → catches static images
          Gate 2 (video):  challenge + FFT screen   → catches video replay

        Both gates must pass simultaneously for ≥ consec_live_required
        consecutive frames before is_live = True.

        Args:
            frame_bgr: Full BGR frame from OpenCV capture
            face_location: Tuple (top, right, bottom, left) at frame resolution
            landmarks_dict: Result from face_recognition.face_landmarks(), or None

        Returns:
            dict with liveness results, gate scores, and all signal values
        """
        blink_signal = 0.0
        movement_signal = 0.0
        mouth_signal = 0.0
        challenge_result = self.challenge_detector.get_result()

        # --- Gate 2 signals: Screen detection (FFT on face ROI) ---
        screen_signal = self.screen_detector.update(frame_bgr, face_location)

        # --- Gate 1 signals: Landmark-based passive detectors ---
        if landmarks_dict is not None:
            left_eye = landmarks_dict.get('left_eye')
            right_eye = landmarks_dict.get('right_eye')

            if left_eye and right_eye:
                blink_signal = self.blink_detector.update(left_eye, right_eye)

            movement_signal = self.movement_detector.update(landmarks_dict)
            mouth_signal = self.mouth_detector.update(landmarks_dict)

            # --- Gate 2 signals: Head-pose challenge ---
            if self.challenge_detector.is_active:
                challenge_result = self.challenge_detector.update(landmarks_dict)

        challenge_signal = challenge_result['signal']

        # =============================================================
        # Gate 1 — Photo / Liveness (is this a live person, not a photo?)
        # Passive: must detect blinks + mouth/head micro-movement.
        # =============================================================
        photo_score = (0.45 * blink_signal +
                       0.30 * mouth_signal +
                       0.25 * movement_signal)
        photo_passed = photo_score >= self.photo_threshold

        # Auto-start challenge once Gate 1 passes for the first time
        if photo_passed and not self.challenge_detector.is_active \
                and not self.challenge_detector.is_complete:
            self.challenge_detector.start()
            challenge_result = self.challenge_detector.get_result()
            challenge_signal = challenge_result['signal']

        # =============================================================
        # Gate 2 — Video detection (is this a real camera, not a screen?)
        # Only meaningful after Gate 1 passes and challenge starts.
        # =============================================================
        video_score = (0.55 * challenge_signal +
                       0.45 * screen_signal)
        video_passed = video_score >= self.video_threshold

        # =============================================================
        # Final decision — both gates must hold consecutively
        # =============================================================
        liveness_score = 0.50 * photo_score + 0.50 * video_score
        both_passed = photo_passed and video_passed

        if both_passed:
            self._consec_live_count += 1
        else:
            self._consec_live_count = 0

        is_live = self._consec_live_count >= self._consec_live_required

        # Update state
        self.last_liveness_score = liveness_score
        self.last_is_live = is_live
        self.last_signals = {
            "blink": blink_signal, "movement": movement_signal,
            "mouth": mouth_signal, "challenge": challenge_signal,
            "screen": screen_signal,
        }

        return {
            "is_live": is_live,
            "liveness_score": liveness_score,
            "photo_score": photo_score,
            "photo_passed": photo_passed,
            "video_score": video_score,
            "video_passed": video_passed,
            "blink_signal": blink_signal,
            "movement_signal": movement_signal,
            "mouth_signal": mouth_signal,
            "challenge_signal": challenge_signal,
            "screen_signal": screen_signal,
            "challenge_result": challenge_result,
            "ear": self.blink_detector.last_ear,
            "mar": self.mouth_detector.last_mar,
            "consistency_score": challenge_result.get('consistency_score', 0.0),
        }

    def reset(self):
        """Reset all detector state."""
        self.blink_detector.reset()
        self.movement_detector.reset()
        self.mouth_detector.reset()
        self.challenge_detector.reset()
        self.screen_detector.reset()
        self.last_liveness_score = 0.0
        self.last_is_live = False
        self._consec_live_count = 0

    def release(self):
        """Release resources. No-op (no external models to close)."""
        pass
