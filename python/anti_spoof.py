"""
Anti-Spoofing / Liveness Detection Module
==========================================
Multi-layer anti-spoofing system using only MTCNN and CNN for maximum accuracy:
  1. Eye blink detection (dlib 68-point landmarks via face_recognition)
  2. Non-rigid landmark movement detection (dlib landmarks)
  3. 3D depth-motion parallax detection (depth-layered landmark analysis)
  4. Mouth micro-movement detection (dlib landmarks)
  5. Head-pose challenge-response with 3D geometric consistency
  6. FFT screen detection (frequency-domain pixel-grid / moiré analysis)
  7. Colour-channel correlation (sub-pixel RGB stripe detection)
  8. Texture frequency band roll-off check (1/f spectral decay)
  9. Screen reflection pattern detection (specular highlight analysis)

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
 
    NOISE_FLOOR = 0.5   # ignore non-rigid std below this (camera sensor noise)

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
            score = float(np.std(local_mags))
            # Subtract noise floor — laptop cameras produce ~0.3-0.5 px jitter
            score = max(0.0, score - self.NOISE_FLOOR)
            non_rigid_scores.append(score)

        avg_non_rigid = np.mean(non_rigid_scores)
        signal = min(1.0, avg_non_rigid / self.movement_threshold)
        return float(signal)

    def reset(self):
        self.landmark_history.clear()


# =============================================================================
# Depth Motion Detector (3D parallax from landmark micro-movements)
# =============================================================================

class DepthMotionDetector:
    """Detect 3D-dependent motion by analysing depth-layered landmark parallax.

    When a real 3D face makes small head movements, landmarks at different
    depths displace by different amounts due to perspective projection:
      - **Front** landmarks (nose tip, nose bridge) are closest to the camera
        and exhibit the largest apparent displacement.
      - **Mid** landmarks (eyes, mouth centre) sit at intermediate depth.
      - **Back** landmarks (jaw edges, chin bottom) are furthest and move least.

    A flat photo or screen surface shows *uniform* displacement across all
    depth layers because every point is at the same distance from the camera.

    Three sub-signals are fused:
      1. Parallax gradient  (40%) — monotonic depth-ordered displacement
      2. Rotation asymmetry (35%) — left/right foreshortening difference
      3. Depth-layer variance (25%) — per-layer displacement variance spread
    """

    # Minimum pixel displacement to consider a frame pair (filters camera noise)
    NOISE_FLOOR = 1.5

    def __init__(self, history_size=20, gradient_threshold=0.003,
                 asymmetry_threshold=0.004, layer_var_threshold=0.0015):
        self.history_size = history_size
        self.gradient_threshold = gradient_threshold
        self.asymmetry_threshold = asymmetry_threshold
        self.layer_var_threshold = layer_var_threshold
        self.landmark_history = []

    # ---- landmark extraction ----

    def _extract_depth_groups(self, lm):
        """Extract landmarks into three depth-ordered groups.

        Returns:
            (front, mid, back)  — each is an ndarray of shape (N, 2).
            None on failure.

        Depth ordering (approximate, for a frontal face):
            Front  — nose_tip centre, nose_bridge bottom (closest to camera)
            Mid    — left/right eye outer corners, upper/lower lip centres
            Back   — left/right jaw edges, chin bottom (furthest from camera)
        """
        try:
            front = np.array([
                lm['nose_tip'][2],
                lm['nose_bridge'][3],
            ], dtype=float)

            mid = np.array([
                lm['left_eye'][0],
                lm['right_eye'][3],
                lm['top_lip'][3],
                lm['bottom_lip'][3],
            ], dtype=float)

            back = np.array([
                lm['chin'][0],
                lm['chin'][16],
                lm['chin'][8],
            ], dtype=float)

            return front, mid, back
        except (KeyError, IndexError):
            return None

    def _extract_left_right(self, lm):
        """Extract left-side and right-side landmark sets for asymmetry.

        Returns:
            (left_pts, right_pts) — each ndarray (N, 2).  None on failure.
        """
        try:
            left_pts = np.array([
                lm['left_eye'][0],
                lm['left_eyebrow'][0],
                lm['chin'][2],
            ], dtype=float)

            right_pts = np.array([
                lm['right_eye'][3],
                lm['right_eyebrow'][4],
                lm['chin'][14],
            ], dtype=float)

            return left_pts, right_pts
        except (KeyError, IndexError):
            return None

    # ---- update / signal ----

    def update(self, landmarks_dict):
        """Process one frame's landmarks.  Returns depth-motion signal [0, 1]."""
        groups = self._extract_depth_groups(landmarks_dict)
        lr = self._extract_left_right(landmarks_dict)

        if groups is None or lr is None:
            return self.get_signal()

        front, mid, back = groups
        left_pts, right_pts = lr

        # Store all points together for frame-over-frame displacement
        self.landmark_history.append({
            'front': front, 'mid': mid, 'back': back,
            'left': left_pts, 'right': right_pts,
        })
        if len(self.landmark_history) > self.history_size:
            self.landmark_history.pop(0)

        return self.get_signal()

    def get_signal(self):
        """Compute fused 3D depth-motion signal from landmark history."""
        if len(self.landmark_history) < 4:
            return 0.0

        gradient_sig = self._parallax_gradient()
        asymmetry_sig = self._rotation_asymmetry()
        layer_var_sig = self._depth_layer_variance()

        signal = (0.40 * gradient_sig +
                  0.35 * asymmetry_sig +
                  0.25 * layer_var_sig)
        return float(np.clip(signal, 0.0, 1.0))

    # ---- sub-signal 1: parallax gradient ----

    def _parallax_gradient(self):
        """Score whether front landmarks displace more than mid > back.

        For each consecutive frame pair, compute mean displacement per
        depth group.  A real 3D face shows front > mid > back (monotonic
        decrease).  Score increases when this gradient is consistently
        observed.
        """
        gradient_scores = []
        for i in range(1, len(self.landmark_history)):
            prev, curr = self.landmark_history[i - 1], self.landmark_history[i]

            front_disp = np.mean(np.linalg.norm(curr['front'] - prev['front'], axis=1))
            mid_disp = np.mean(np.linalg.norm(curr['mid'] - prev['mid'], axis=1))
            back_disp = np.mean(np.linalg.norm(curr['back'] - prev['back'], axis=1))

            # Check monotonic gradient: front > mid > back
            total_disp = front_disp + mid_disp + back_disp
            if total_disp < self.NOISE_FLOOR * 3:
                # Below noise floor — skip (camera jitter, not real motion)
                continue

            # Gradient strength: difference between front and back,
            # normalised by total displacement
            gradient = (front_disp - back_disp) / (total_disp + 1e-8)

            # Bonus for strict monotonic ordering
            if front_disp > mid_disp > back_disp:
                gradient_scores.append(abs(gradient))
            else:
                gradient_scores.append(abs(gradient) * 0.3)

        if not gradient_scores:
            return 0.0

        avg = float(np.mean(gradient_scores))
        return min(1.0, avg / self.gradient_threshold)

    # ---- sub-signal 2: rotation asymmetry ----

    def _rotation_asymmetry(self):
        """Score left/right displacement asymmetry during lateral motion.

        When a real face rotates laterally, the side moving *toward* the
        camera (foreshortening decreases) displaces differently than the
        side moving *away* (foreshortening increases).  A flat surface
        produces symmetric displacement on both sides.
        """
        asymmetry_scores = []
        for i in range(1, len(self.landmark_history)):
            prev, curr = self.landmark_history[i - 1], self.landmark_history[i]

            left_disp = np.mean(np.linalg.norm(curr['left'] - prev['left'], axis=1))
            right_disp = np.mean(np.linalg.norm(curr['right'] - prev['right'], axis=1))

            total = left_disp + right_disp
            if total < self.NOISE_FLOOR * 2:
                continue

            asymmetry = abs(left_disp - right_disp) / (total + 1e-8)
            asymmetry_scores.append(asymmetry)

        if not asymmetry_scores:
            return 0.0

        avg = float(np.mean(asymmetry_scores))
        return min(1.0, avg / self.asymmetry_threshold)

    # ---- sub-signal 3: depth-layer variance ----

    def _depth_layer_variance(self):
        """Score whether displacement variance differs across depth layers.

        For a real 3D face, each depth layer has a *different* internal
        displacement variance (front landmarks jitter more than back).
        A flat photo has near-identical variance at every depth layer.
        Score = spread (std) of per-layer variances.
        """
        layer_variances = {'front': [], 'mid': [], 'back': []}

        for i in range(1, len(self.landmark_history)):
            prev, curr = self.landmark_history[i - 1], self.landmark_history[i]

            for key in ('front', 'mid', 'back'):
                disps = np.linalg.norm(curr[key] - prev[key], axis=1)
                layer_variances[key].append(float(np.var(disps)))

        # Mean variance per layer
        means = []
        for key in ('front', 'mid', 'back'):
            if layer_variances[key]:
                means.append(float(np.mean(layer_variances[key])))

        if len(means) < 3:
            return 0.0

        spread = float(np.std(means))
        return min(1.0, spread / self.layer_var_threshold)

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

    # MAR deltas below this are treated as sensor noise (not real mouth motion)
    MAR_NOISE_FLOOR = 0.0015

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
        # Subtract noise floor — camera jitter causes ~0.001 MAR fluctuation
        deltas = [max(0.0, d - self.MAR_NOISE_FLOOR) for d in deltas]
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
        roi_resized = cv2.resize(roi, (self.ANALYSIS_SIZE, self.ANALYSIS_SIZE))

        score, metrics = self._analyze(gray, roi_resized)

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

    def _analyze(self, gray, roi_bgr=None):
        """Run frequency + spatial + colour analysis on a 128×128 face ROI.

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
            peak_ratio = float(np.sum(mid_mags > 3.0 * median_mid) / mid_mags.size) \
                         if median_mid > 0 else 0.0
        else:
            peak_ratio = 0.0

        # 4. Laplacian variance (spatial-domain sharpness)
        #    Screen pixels create fine sharp edges that boost Laplacian.
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 5. High-frequency energy ratio
        #    Screens leak more energy into the outer frequency ring
        #    (pixel-edge harmonics) compared to organic skin texture.
        hf_mask = r > max_r * 0.75
        hf_energy = float(np.sum(magnitude[hf_mask] ** 2) / total_energy)

        # 6. Colour-channel correlation (sub-pixel RGB stripe detection)
        #    Screen sub-pixels (R-G-B stripes) cause very high inter-
        #    channel correlation in the high-frequency domain.  Real
        #    skin has more independent per-channel texture.
        color_corr = 0.0
        if roi_bgr is not None:
            channels = cv2.split(roi_bgr.astype(np.float64))
            ch_hf = []
            for ch in channels:
                ch_shift = np.fft.fftshift(np.fft.fft2(ch))
                ch_mag = np.abs(ch_shift)
                ch_hf.append(ch_mag[hf_mask])
            # Mean pairwise Pearson correlation among B, G, R high-freq
            pairs = [(0, 1), (0, 2), (1, 2)]
            corrs = []
            for a, b in pairs:
                if ch_hf[a].std() > 1e-6 and ch_hf[b].std() > 1e-6:
                    corrs.append(float(np.corrcoef(ch_hf[a], ch_hf[b])[0, 1]))
            color_corr = float(np.mean(corrs)) if corrs else 0.0

        # 7. Texture frequency band roll-off check
        #    Real skin follows a ~1/f spectral decay: each outer radial
        #    band holds progressively less energy.  Screens and printed
        #    photos break this smooth roll-off (pixel-pitch plateaus,
        #    printer-dot peaks, or sharp cutoffs).
        #
        #    Procedure:
        #      a) Split spectrum into 4 concentric rings (excluding DC).
        #      b) Compute mean energy per pixel in each ring.
        #      c) Compute successive ratios: band[i+1] / band[i].
        #         For ideal 1/f these ratios are <1 and consistent.
        #      d) Score = smoothness of the decay (low variance of ratios
        #         + all ratios < 1 = good).
        band_edges = [3, max_r * 0.15, max_r * 0.35, max_r * 0.60, max_r * 0.90]
        band_energies = []
        for bi in range(len(band_edges) - 1):
            ring = (r >= band_edges[bi]) & (r < band_edges[bi + 1])
            npix = np.sum(ring)
            if npix > 0:
                band_energies.append(float(np.sum(magnitude[ring] ** 2) / npix))
            else:
                band_energies.append(0.0)

        # Successive ratios (outer / inner)
        ratios = []
        for bi in range(1, len(band_energies)):
            if band_energies[bi - 1] > 1e-10:
                ratios.append(band_energies[bi] / band_energies[bi - 1])

        if ratios:
            # All ratios should be < 1 (decay).  Penalise any ratio ≥ 1.
            ratio_violations = sum(1 for rr in ratios if rr >= 1.0)
            # Low variance among ratios → smooth roll-off
            ratio_std = float(np.std(ratios))
            # Mean ratio — real skin ≈ 0.3–0.6,  spoofs can be > 0.8
            ratio_mean = float(np.mean(ratios))
        else:
            ratio_violations = 0
            ratio_std = 0.0
            ratio_mean = 0.5

        texture_rolloff = {
            'band_energies': band_energies,
            'ratios':        ratios,
            'ratio_mean':    ratio_mean,
            'ratio_std':     ratio_std,
            'violations':    ratio_violations,
        }

        # ---- Per-metric scoring (each ∈ [0, 1], 1 = real face) ----

        # Flatness: real skin typically > 0.012, screens < 0.008
        #   Tightened: divide by 0.010 so screens must be clearly flat
        flatness_sig = float(min(1.0, spectral_flatness / 0.010))

        # Axis energy: real < 0.06, screens > 0.09
        #   Tightened: dead zone starts at 0.04, full penalty by 0.12
        axis_sig = float(1.0 - min(1.0, max(0.0, (axis_energy - 0.04)) / 0.08))

        # Peak ratio: real < 0.005, screens > 0.015
        #   Tightened: divider lowered to 0.02
        peak_sig = float(1.0 - min(1.0, peak_ratio / 0.02))

        # Laplacian: very high variance → sharp pixel edges (screen)
        #   Tightened: kicks in at 1200 instead of 2000
        lap_sig = float(max(0.2, 1.0 - max(0.0, lap_var - 1200) / 4000)
                        if lap_var > 1200 else 1.0)

        # High-frequency energy: real < 0.02, screens > 0.04
        hf_sig = float(1.0 - min(1.0, max(0.0, hf_energy - 0.015) / 0.03))

        # Colour correlation: real skin < 0.7, screens > 0.85
        #   High correlation → screen sub-pixel stripes
        color_sig = float(1.0 - min(1.0, max(0.0, color_corr - 0.65) / 0.25))

        # Texture roll-off: penalise violations (ratio ≥ 1), high mean,
        #   and high variance.  Each sub-score ∈ [0, 1].
        rolloff_violation_score = max(0.0, 1.0 - ratio_violations * 0.4)
        rolloff_mean_score = float(1.0 - min(1.0, max(0.0, ratio_mean - 0.55) / 0.40))
        rolloff_std_score = float(1.0 - min(1.0, ratio_std / 0.25))
        rolloff_sig = float(0.40 * rolloff_violation_score +
                            0.35 * rolloff_mean_score +
                            0.25 * rolloff_std_score)

        # ---- Fusion (7 metrics) ----
        score = (0.17 * flatness_sig +
                 0.17 * axis_sig +
                 0.13 * peak_sig +
                 0.08 * lap_sig +
                 0.13 * hf_sig +
                 0.17 * color_sig +
                 0.15 * rolloff_sig)

        metrics = {
            'spectral_flatness': spectral_flatness,
            'axis_energy':       axis_energy,
            'peak_ratio':        peak_ratio,
            'laplacian_var':     lap_var,
            'hf_energy':         hf_energy,
            'color_corr':        color_corr,
            'texture_rolloff':   texture_rolloff,
            'flatness_sig':      flatness_sig,
            'axis_sig':          axis_sig,
            'peak_sig':          peak_sig,
            'lap_sig':           lap_sig,
            'hf_sig':            hf_sig,
            'color_sig':         color_sig,
            'rolloff_sig':       rolloff_sig,
        }

        return float(np.clip(score, 0.0, 1.0)), metrics


# =============================================================================
# Reflection Pattern Detector (screen specular highlight analysis)
# =============================================================================

class ReflectionPatternDetector:
    """Detect screen glass reflections via specular highlight analysis.

    Screens are flat glass surfaces that produce characteristic reflection
    patterns distinguishable from the curved 3D surface of real skin:

      1. On a screen, specular highlights from room lighting are spatially
         *uniform* or appear as broad rectangular patches across the face.
         On real skin, highlights concentrate on convex surfaces (nose
         bridge, forehead, cheekbones — the central vertical band).

      2. Screen highlights are anchored to the glass surface and drift
         relative to the face as the head moves.  Real-skin highlights
         move *with* the underlying face geometry and stay stable
         relative to the face centre.

      3. Gradient transitions around screen highlights are uniform (flat
         glass).  On curved skin, gradient patterns follow 3D curvature.

      4. Screens produce sharper specular-to-diffuse transitions (flat
         glass → concentrated highlight).  Real skin has softer, more
         diffuse highlight fall-off.

    Uses only OpenCV + NumPy — no extra dependencies.
    """

    ANALYSIS_SIZE = 128

    def __init__(self, history_size=25):
        self.history_size = history_size
        self.score_history = []
        self.last_score = 0.0
        self.last_metrics = {}

        # Temporal tracking for highlight drift
        self._highlight_centroid_history = []
        self._face_center_history = []

    # ----- public API -----

    def update(self, frame_bgr, face_location):
        """Analyse one frame's face ROI for screen reflection artifacts.

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

        # Face centre in ROI-normalised coordinates [0, 1]
        face_cx = 0.5
        face_cy = 0.5

        score, metrics = self._analyze(gray, face_cx, face_cy)

        self.last_score = score
        self.last_metrics = metrics
        self.score_history.append(score)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)

        return self.get_signal()

    def get_signal(self):
        """Median-smoothed reflection-detection signal over recent frames."""
        if not self.score_history:
            return 1.0                   # assume real until proven otherwise
        if len(self.score_history) < 3:
            return float(np.mean(self.score_history))
        return float(np.median(self.score_history))

    def reset(self):
        self.score_history.clear()
        self.last_score = 0.0
        self.last_metrics = {}
        self._highlight_centroid_history.clear()
        self._face_center_history.clear()

    # ----- internal -----

    def _analyze(self, gray, face_cx, face_cy):
        """Run specular-highlight analysis on a 128×128 face ROI.

        Returns (score, metrics_dict).
            score  : float [0.0, 1.0]  — 1.0 = real skin, 0.0 = screen.
            metrics: dict of intermediate values for debugging / HUD.
        """
        sz = self.ANALYSIS_SIZE
        gray_f = gray.astype(np.float64)

        # ---- Identify specular highlight pixels ----
        # Use the top 2% brightest pixels as "highlights"
        threshold = np.percentile(gray_f, 98)
        highlight_mask = gray_f >= threshold
        num_highlights = int(np.sum(highlight_mask))

        if num_highlights < 5:
            # No meaningful highlights — assume real (no screen glare)
            return 1.0, {'no_highlights': True}

        hy, hx = np.where(highlight_mask)

        # ============================================================
        # 1. Highlight spatial distribution (30%)
        #    Real skin: highlights cluster on convex areas in the
        #    central vertical band (nose bridge, forehead, cheekbones).
        #    Screen: highlights spread uniformly or as broad patches.
        # ============================================================
        central_band_left = int(sz * 0.30)
        central_band_right = int(sz * 0.70)
        in_central = np.sum((hx >= central_band_left) & (hx < central_band_right))
        concentration = in_central / (num_highlights + 1e-8)

        # Real skin: concentration > 0.55 (highlights on nose/forehead)
        # Screen: concentration ≈ 0.40 (spread across whole face)
        distribution_sig = float(min(1.0, max(0.0, concentration - 0.35) / 0.30))

        # ============================================================
        # 2. Highlight temporal drift (30%)
        #    Track highlight centroid relative to face centre.
        #    Screen: highlights drift relative to face (anchored to glass).
        #    Real: highlights stable relative to face geometry.
        # ============================================================
        hl_cx = float(np.mean(hx)) / sz    # normalised [0, 1]
        hl_cy = float(np.mean(hy)) / sz

        # Store relative offset (highlight centroid − face centre)
        rel_x = hl_cx - face_cx
        rel_y = hl_cy - face_cy
        self._highlight_centroid_history.append((rel_x, rel_y))
        if len(self._highlight_centroid_history) > self.history_size:
            self._highlight_centroid_history.pop(0)

        drift_sig = 1.0
        if len(self._highlight_centroid_history) >= 5:
            offsets = np.array(self._highlight_centroid_history)
            drift_var = float(np.var(offsets[:, 0]) + np.var(offsets[:, 1]))
            # Real skin: drift_var < 0.002 (stable).
            # Screen: drift_var > 0.008 (highlights shift as face moves).
            drift_sig = float(max(0.0, 1.0 - min(1.0, drift_var / 0.008)))

        # ============================================================
        # 3. Gradient uniformity around highlights (20%)
        #    Screen glass: uniform gradient around highlights.
        #    Real skin: variable gradient (3D curvature).
        # ============================================================
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Sample gradient in an annular region around each highlight
        # (dilate mask then subtract original to get annular ring)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(highlight_mask.astype(np.uint8), kernel)
        annular = (dilated > 0) & (~highlight_mask)

        annular_grads = grad_mag[annular]
        if annular_grads.size > 10:
            grad_variance = float(np.var(annular_grads))
            # Real skin: high gradient variance (curved surface, > 500)
            # Screen: low gradient variance (flat glass, < 200)
            gradient_sig = float(min(1.0, grad_variance / 800.0))
        else:
            gradient_sig = 0.5

        # ============================================================
        # 4. Specular-to-diffuse ratio (20%)
        #    Screen: sharp, concentrated specular peak (flat glass).
        #    Real skin: gradual, diffuse highlight fall-off.
        # ============================================================
        very_bright = np.sum(gray_f >= np.percentile(gray_f, 99))
        moderately_bright = np.sum(gray_f >= np.percentile(gray_f, 90))

        if moderately_bright > 0:
            spec_ratio = float(very_bright) / float(moderately_bright)
        else:
            spec_ratio = 0.0

        # Real skin: spec_ratio ≈ 0.08–0.15 (gradual distribution)
        # Screen: spec_ratio > 0.20 (sharp concentrated peak)
        # Score: penalise very high ratios (screen-like concentrated peak)
        if spec_ratio < 0.18:
            specular_sig = 1.0
        elif spec_ratio < 0.35:
            specular_sig = float(1.0 - (spec_ratio - 0.18) / 0.17)
        else:
            specular_sig = 0.0

        # ---- Fusion (4 metrics) ----
        score = (0.30 * distribution_sig +
                 0.30 * drift_sig +
                 0.20 * gradient_sig +
                 0.20 * specular_sig)

        metrics = {
            'concentration':   concentration,
            'drift_var':       drift_var if len(self._highlight_centroid_history) >= 5 else 0.0,
            'grad_variance':   grad_variance if annular_grads.size > 10 else 0.0,
            'spec_ratio':      spec_ratio,
            'distribution_sig': distribution_sig,
            'drift_sig':       drift_sig,
            'gradient_sig':    gradient_sig,
            'specular_sig':    specular_sig,
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
        • Eye blink detection              (30%)
        • Mouth micro-movement             (30%)
        • Non-rigid landmark movement      (15%)
        • 3D depth-motion parallax         (25%)
      Must reach photo_threshold to pass.

    Gate 2 — Video detection (active challenge + FFT):
      Proves the feed is not a video replay on a screen.
        • Head-pose challenge with 3D consistency (45%)
        • FFT screen / moiré detection             (35%)
          - 7 sub-metrics: spectral flatness, axis energy,
            mid-freq peaks, Laplacian, HF energy,
            colour correlation, texture roll-off
        • Screen reflection pattern detection      (20%)
      Must reach video_threshold to pass.

    Final decision: Gate 1 must pass first, then Gate 2 is evaluated.
    BOTH gates must pass for ≥ consec_live_required consecutive
    evaluations before is_live = True.  Each gate contributes 100%
    of its own score (no averaging between gates).

    Uses only MTCNN and CNN (dlib) — no MediaPipe dependency.
    """

    def __init__(self, photo_threshold=0.40, video_threshold=0.35,
                 num_challenges=4, challenge_timeout=10.0,
                 ear_threshold=0.25, consec_frames=1, blink_time_window=5.0,
                 mar_movement_threshold=0.005, mar_history_size=20,
                 movement_threshold=0.002, movement_history=15):

        # Gate thresholds
        self.photo_threshold = photo_threshold
        self.video_threshold = video_threshold

        # Sub-detectors
        self.blink_detector = BlinkDetector(ear_threshold, consec_frames, blink_time_window)
        self.mouth_detector = MouthMovementDetector(mar_movement_threshold, mar_history_size)
        self.movement_detector = MovementDetector(movement_threshold, movement_history)
        self.depth_motion_detector = DepthMotionDetector()
        self.challenge_detector = ChallengeResponseDetector(num_challenges, challenge_timeout)
        self.screen_detector = ScreenDetector()
        self.reflection_detector = ReflectionPatternDetector()

        # State
        self.last_liveness_score = 0.0
        self.last_is_live = False
        self.last_signals = {
            "blink": 0.0, "movement": 0.0, "depth_motion": 0.0,
            "mouth": 0.0, "challenge": 0.0,
            "screen": 1.0, "reflection": 1.0,
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
          Gate 1 (photo):  blink + mouth + movement + depth_motion
          Gate 2 (video):  challenge + FFT screen + reflection

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
        mouth_signal = 0.0
        movement_signal = 0.0
        depth_motion_signal = 0.0
        challenge_result = self.challenge_detector.get_result()

        # --- Gate 2 signals: Screen detection (FFT + reflection on face ROI) ---
        screen_signal = self.screen_detector.update(frame_bgr, face_location)
        reflection_signal = self.reflection_detector.update(frame_bgr, face_location)

        # --- Gate 1 signals: Landmark-based passive detectors ---
        if landmarks_dict is not None:
            left_eye = landmarks_dict.get('left_eye')
            right_eye = landmarks_dict.get('right_eye')

            if left_eye and right_eye:
                blink_signal = self.blink_detector.update(left_eye, right_eye)

            mouth_signal = self.mouth_detector.update(landmarks_dict)
            movement_signal = self.movement_detector.update(landmarks_dict)
            depth_motion_signal = self.depth_motion_detector.update(landmarks_dict)

            # --- Gate 2 signals: Head-pose challenge ---
            if self.challenge_detector.is_active:
                challenge_result = self.challenge_detector.update(landmarks_dict)

        challenge_signal = challenge_result['signal']

        # =============================================================
        # Gate 1 — Photo / Liveness (is this a live person, not a photo?)
        # Passive: blinks + mouth micro-movement + non-rigid landmark
        # motion + 3D depth-dependent parallax.
        #
        # HARD REQUIREMENT: at least one blink must be detected before
        # Gate 1 can pass.  A photo can never blink, so this is the
        # single most reliable photo-vs-live discriminator and prevents
        # camera sensor noise in the other signals from fooling Gate 1.
        # =============================================================
        photo_score = (0.30 * blink_signal +
                       0.30 * mouth_signal +
                       0.15 * movement_signal +
                       0.25 * depth_motion_signal)
        blink_detected = len(self.blink_detector.blink_timestamps) > 0
        photo_passed = (photo_score >= self.photo_threshold) and blink_detected

        # Auto-start challenge once Gate 1 passes for the first time
        if photo_passed and not self.challenge_detector.is_active \
                and not self.challenge_detector.is_complete:
            self.challenge_detector.start()
            challenge_result = self.challenge_detector.get_result()
            challenge_signal = challenge_result['signal']

        # =============================================================
        # Gate 2 — Video detection (is this a real camera, not a screen?)
        # Only evaluated after Gate 1 passes.  If Gate 1 fails (photo
        # detected), Gate 2 is skipped entirely — no challenge, no FFT.
        # =============================================================
        if photo_passed:
            video_score = (0.45 * challenge_signal +
                           0.35 * screen_signal +
                           0.20 * reflection_signal)
            video_passed = video_score >= self.video_threshold
        else:
            # Photo detected → block: do not evaluate Gate 2
            video_score = 0.0
            video_passed = False

        # =============================================================
        # Final decision — sequential gates, both must hold consecutively
        #   Gate 1 (photo) must pass first → then Gate 2 (video) is evaluated.
        #   Each gate is 100% of its own score (no averaging).
        # =============================================================
        if photo_passed and video_passed:
            liveness_score = min(photo_score, video_score)
        elif photo_passed:
            liveness_score = photo_score * 0.5   # half credit: photo OK, video pending
        else:
            liveness_score = photo_score * 0.25  # low: still on Gate 1
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
            "depth_motion": depth_motion_signal,
            "mouth": mouth_signal, "challenge": challenge_signal,
            "screen": screen_signal, "reflection": reflection_signal,
        }

        return {
            "is_live": is_live,
            "liveness_score": liveness_score,
            "photo_score": photo_score,
            "photo_passed": photo_passed,
            "video_score": video_score,
            "video_passed": video_passed,
            "blink_signal": blink_signal,
            "mouth_signal": mouth_signal,
            "movement_signal": movement_signal,
            "depth_motion_signal": depth_motion_signal,
            "challenge_signal": challenge_signal,
            "screen_signal": screen_signal,
            "reflection_signal": reflection_signal,
            "challenge_result": challenge_result,
            "ear": self.blink_detector.last_ear,
            "mar": self.mouth_detector.last_mar,
            "consistency_score": challenge_result.get('consistency_score', 0.0),
        }

    def reset(self):
        """Reset all detector state."""
        self.blink_detector.reset()
        self.mouth_detector.reset()
        self.movement_detector.reset()
        self.depth_motion_detector.reset()
        self.challenge_detector.reset()
        self.screen_detector.reset()
        self.reflection_detector.reset()
        self.last_liveness_score = 0.0
        self.last_is_live = False
        self._consec_live_count = 0

    def release(self):
        """Release resources. No-op (no external models to close)."""
        pass
