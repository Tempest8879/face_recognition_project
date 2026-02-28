"""
Anti-Spoofing / Liveness Detection Module
==========================================
Two-gate anti-spoofing system using MTCNN and dlib landmarks:
  1. Eye blink detection (dlib 68-point landmarks via face_recognition)
  2. Randomized head-pose challenge-response with 3D geometric consistency

Dependencies: face_recognition (dlib), opencv-python, numpy
"""

import time
import random
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

    **Adaptive calibration**: Instead of a fixed threshold, the detector
    learns the user's personal open-eye EAR during a short warm-up window
    (``calibration_frames`` frames).  The blink threshold is then set to
    ``calibrated_open_ear * close_ratio``.  This makes detection robust
    across different eye shapes, glasses, and camera angles.

    A valid blink requires:
      1. Calibration is complete (enough open-eye samples collected).
      2. EAR was above the adaptive open baseline (eyes confirmed open).
      3. EAR drops below the adaptive close threshold for at least
         ``consec_frames`` consecutive frames.
      4. EAR rises back above the close threshold (reopening).
      5. The closed phase did not exceed ``max_closed_frames``.
    """

    def __init__(self, ear_threshold=0.28, consec_frames=3,
                 time_window=5.0, close_ratio=0.75,
                 max_closed_frames=20,
                 min_blink_interval=0.28,
                 calibration_frames=15):
        # Fixed fallback threshold (used before calibration completes)
        self.ear_threshold_fixed = ear_threshold
        self.consec_frames = consec_frames
        self.time_window = time_window
        self.close_ratio = close_ratio
        self.max_closed_frames = max_closed_frames
        self.min_blink_interval = min_blink_interval
        self.calibration_frames = calibration_frames

        # Adaptive calibration state
        self._calibration_samples = []    # open-eye EAR samples
        self._calibrated_open_ear = None  # median of calibration samples
        self._ear_threshold = ear_threshold  # active threshold (updates after calibration)
        self._open_baseline = ear_threshold * 1.15  # active open baseline

        # Blink state machine
        self.below_threshold_count = 0
        self.was_open = False
        self.blink_timestamps = []
        self.last_ear = 0.0

    @property
    def ear_threshold(self):
        """Current active EAR threshold (may be calibrated or fixed)."""
        return self._ear_threshold

    @property
    def is_calibrated(self):
        return self._calibrated_open_ear is not None

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

    def _calibrate(self, ear):
        """Collect open-eye EAR samples and compute adaptive threshold.

        During calibration, we assume the user's eyes are open (since
        they're looking at the screen / prompt).  We collect samples
        and derive the threshold from the median.
        """
        if self._calibrated_open_ear is not None:
            return  # already calibrated

        self._calibration_samples.append(ear)

        if len(self._calibration_samples) >= self.calibration_frames:
            median_ear = float(np.median(self._calibration_samples))
            # Sanity: only accept calibration if median looks like open eyes
            if median_ear > 0.15:
                self._calibrated_open_ear = median_ear
                self._ear_threshold = median_ear * self.close_ratio
                self._open_baseline = median_ear * 0.90
                self.was_open = True  # eyes are open during calibration

    def update(self, left_eye, right_eye):
        """Process one frame's eye landmarks. Returns blink signal [0.0, 1.0].

        State machine:
          CALIBRATING → collect open-eye samples, no blink detection yet
          OPEN  (EAR ≥ open_baseline)  → arm was_open
          CLOSED (EAR < ear_threshold) → increment counter
          RE-OPEN (EAR ≥ ear_threshold after closed run) → check & register
        """
        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        self.last_ear = avg_ear

        # Calibration phase
        if not self.is_calibrated:
            self._calibrate(avg_ear)
            return self.get_signal()

        now = time.time()

        if avg_ear < self._ear_threshold:
            # --- CLOSED: eyes below threshold ---
            self.below_threshold_count += 1

            # Guard: closure too long → not a blink
            if self.below_threshold_count > self.max_closed_frames:
                self.below_threshold_count = 0
                self.was_open = False
        else:
            # --- NOT CLOSED: eyes at or above threshold ---
            # If we just finished a valid closed run, register blink
            if (self.was_open
                    and self.below_threshold_count >= self.consec_frames
                    and self.below_threshold_count <= self.max_closed_frames):
                self._register_blink(now)

            self.below_threshold_count = 0

            # Arm the detector once EAR is above open baseline
            if avg_ear >= self._open_baseline:
                self.was_open = True

        # Prune old blink timestamps
        cutoff = now - self.time_window
        self.blink_timestamps = [t for t in self.blink_timestamps if t > cutoff]

        return self.get_signal()

    def _register_blink(self, now):
        """Record a blink if enough time has passed since the last one."""
        if (not self.blink_timestamps
                or (now - self.blink_timestamps[-1]) >= self.min_blink_interval):
            self.blink_timestamps.append(now)
        self.was_open = False

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
        self.was_open = False
        self.blink_timestamps.clear()
        self.last_ear = 0.0
        self._calibration_samples.clear()
        self._calibrated_open_ear = None
        self._ear_threshold = self.ear_threshold_fixed
        self._open_baseline = self.ear_threshold_fixed * 1.15


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

    def __init__(self, history_size=30, min_evidence=8):
        self.history_size = history_size
        self.min_evidence = min_evidence
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
        """Median-smoothed screen-detection signal over recent frames.

        Returns 0.0 (suspicious) until at least ``min_evidence`` frames
        have been analysed.  Modern high-PPI OLED screens may have no
        visible pixel grid, so this prevents auto-passing.
        """
        if len(self.score_history) < self.min_evidence:
            return 0.0   # suspicious until proven otherwise
        if len(self.score_history) < 5:
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
# Screen Edge / Bezel Detector
# =============================================================================

class ScreenEdgeDetector:
    """Detect phone / tablet bezels surrounding the face.

    When a face is displayed on a phone / monitor, the camera typically
    captures the device's rectangular edges surrounding the face region.
    These edges form strong, straight lines that are:

      1. Aligned horizontally / vertically (rectilinear).
      2. Located OUTSIDE the face bounding box but INSIDE the camera frame.
      3. Form a roughly rectangular enclosure around the face.

    Real faces viewed directly by a webcam do not have this rectangular
    border pattern — the background is irregular.

    Uses Canny edge detection + Hough line transform.  The signal is
    conservative: a strong rectangular-enclosure detection penalises
    heavily, but absence of edges does NOT auto-pass (returns 0.5).
    """

    def __init__(self, history_size=20, min_evidence=8):
        self.history_size = history_size
        self.min_evidence = min_evidence
        self.score_history = []
        self.last_score = 0.5

    def update(self, frame_bgr, face_location):
        """Analyse frame for rectangular device edges around the face.

        Args:
            frame_bgr:     Full BGR frame.
            face_location: (top, right, bottom, left).

        Returns:
            float signal [0.0, 1.0] — 1.0 = no edges (real),
                                       0.0 = strong rectangular edges.
        """
        top, right, bottom, left = (int(v) for v in face_location)
        h, w = frame_bgr.shape[:2]
        top, left = max(0, top), max(0, left)
        bottom, right = min(h, bottom), min(w, right)

        if h < 64 or w < 64:
            return self.get_signal()

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        score = self._analyze(gray, top, right, bottom, left, h, w)

        self.last_score = score
        self.score_history.append(score)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)

        return self.get_signal()

    def _analyze(self, gray, face_t, face_r, face_b, face_l, h, w):
        """Detect strong rectilinear edges surrounding the face.

        We look for Hough lines in the region OUTSIDE the face bounding
        box.  Strong horizontal and vertical lines that flank the face
        on multiple sides indicate a device bezel.
        """
        # Expand face bbox by 20% to create an exclusion zone
        fh = face_b - face_t
        fw = face_r - face_l
        pad_y = int(fh * 0.20)
        pad_x = int(fw * 0.20)
        excl_t = max(0, face_t - pad_y)
        excl_b = min(h, face_b + pad_y)
        excl_l = max(0, face_l - pad_x)
        excl_r = min(w, face_r + pad_x)

        # Mask out the face region — we only want edges OUTSIDE
        masked = gray.copy()
        masked[excl_t:excl_b, excl_l:excl_r] = 0

        # Canny edge detection
        edges = cv2.Canny(masked, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=60,
                                minLineLength=min(h, w) // 5,
                                maxLineGap=15)

        if lines is None or len(lines) == 0:
            return 0.7  # no strong lines → probably real

        # Classify lines as horizontal or vertical
        h_lines = []  # lines within 10° of horizontal
        v_lines = []  # lines within 10° of vertical
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if angle < 10 or angle > 170:  # horizontal
                h_lines.append((x1, y1, x2, y2, length))
            elif 80 < angle < 100:  # vertical
                v_lines.append((x1, y1, x2, y2, length))

        # Check if lines flank the face on multiple sides
        sides_flanked = 0

        # Top: horizontal line above the face
        for x1, y1, x2, y2, length in h_lines:
            mid_y = (y1 + y2) / 2
            if mid_y < face_t and length > fw * 0.4:
                sides_flanked |= 1
                break

        # Bottom: horizontal line below the face
        for x1, y1, x2, y2, length in h_lines:
            mid_y = (y1 + y2) / 2
            if mid_y > face_b and length > fw * 0.4:
                sides_flanked |= 2
                break

        # Left: vertical line to the left
        for x1, y1, x2, y2, length in v_lines:
            mid_x = (x1 + x2) / 2
            if mid_x < face_l and length > fh * 0.4:
                sides_flanked |= 4
                break

        # Right: vertical line to the right
        for x1, y1, x2, y2, length in v_lines:
            mid_x = (x1 + x2) / 2
            if mid_x > face_r and length > fh * 0.4:
                sides_flanked |= 8
                break

        num_sides = bin(sides_flanked).count('1')

        # Score based on how many sides are flanked
        # 0 sides → 0.8 (probably real)
        # 1 side  → 0.6 (could be a shelf/wall edge)
        # 2 sides → 0.3 (suspicious)
        # 3+ sides → 0.05 (almost certainly a device)
        score_map = {0: 0.8, 1: 0.6, 2: 0.3, 3: 0.05, 4: 0.02}
        score = score_map.get(num_sides, 0.02)

        return float(score)

    def get_signal(self):
        if len(self.score_history) < self.min_evidence:
            return 0.0  # suspicious until proven real
        if len(self.score_history) < 5:
            return float(np.mean(self.score_history))
        return float(np.median(self.score_history))

    def reset(self):
        self.score_history.clear()
        self.last_score = 0.0


# =============================================================================
# Temporal Flicker Detector (screen refresh-rate artifact analysis)
# =============================================================================

class TemporalFlickerDetector:
    """Detect phone / monitor replay via temporal luminance flicker.

    Phone screens refresh at 60–120 Hz.  When captured by a ~30 fps webcam,
    the interaction between the screen's refresh rate and the camera's
    rolling-shutter / global-shutter exposure produces periodic micro-
    fluctuations in average brightness:

      beat_freq  =  |screen_refresh - N × camera_fps|

    For a 60 Hz screen at 30 fps → beat frequency is 0 Hz or 60 Hz
    aliased to observable harmonics.  For 120 Hz at 30 fps → 0 Hz
    (hard), but sub-harmonics at 30/60 Hz leak through.

    Real faces illuminated by natural/DC lighting have NO periodic
    brightness fluctuation — luminance changes are slow and aperiodic
    (movement, clouds, etc.).

    Algorithm:
      1. Collect face-ROI mean brightness per frame into a ring buffer.
      2. Once the buffer has ≥ 32 samples, compute the FFT of the
         temporal brightness signal.
      3. Measure the ratio of mid/high-frequency energy (≥ 5 Hz) to
         total energy.  Screens produce peaks; real faces don't.
      4. Also measure the spectral peak prominence — a single strong
         peak indicates a beat frequency from a screen.

    This is the strongest phone-replay discriminator because it measures
    a *physical property of the screen* that no video content can hide.
    """

    def __init__(self, buffer_size=64, min_evidence=32):
        self.buffer_size = buffer_size
        self.min_evidence = min_evidence
        self._brightness_buffer = []
        self._patch_buffer = []  # small grayscale patches for inter-frame analysis
        self.last_score = 0.0
        self.last_metrics = {}

    def update(self, frame_bgr, face_location):
        """Record face-ROI mean brightness and analyze flicker.

        Args:
            frame_bgr:     Full BGR frame.
            face_location: (top, right, bottom, left).

        Returns:
            float signal [0.0, 1.0] — 1.0 = no flicker (real),
                                       0.0 = periodic flicker (screen).
        """
        top, right, bottom, left = (int(v) for v in face_location)
        h, w = frame_bgr.shape[:2]
        top, left = max(0, top), max(0, left)
        bottom, right = min(h, bottom), min(w, right)

        roi = frame_bgr[top:bottom, left:right]
        if roi.size == 0 or roi.shape[0] < 16 or roi.shape[1] < 16:
            return self.get_signal()

        # Mean brightness of face ROI (luminance channel)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        self._brightness_buffer.append(mean_brightness)
        if len(self._brightness_buffer) > self.buffer_size:
            self._brightness_buffer.pop(0)

        # Store small standardised patch for inter-frame correlation
        patch = cv2.resize(gray, (32, 32)).astype(np.float64)
        self._patch_buffer.append(patch)
        if len(self._patch_buffer) > self.buffer_size:
            self._patch_buffer.pop(0)

        if len(self._brightness_buffer) < self.min_evidence:
            return self.get_signal()

        self.last_score = self._analyze()
        return self.get_signal()

    def _analyze(self):
        """FFT + inter-frame noise analysis of temporal brightness signal."""

        # ============================================================
        # Part A: FFT analysis of temporal mean brightness
        # ============================================================
        signal = np.array(self._brightness_buffer, dtype=np.float64)

        # Remove DC (mean) and linear trend
        signal = signal - np.mean(signal)
        t = np.arange(len(signal), dtype=np.float64)
        if np.std(signal) > 0.01:
            coeffs = np.polyfit(t, signal, 1)
            signal = signal - np.polyval(coeffs, t)

        # Windowing to reduce spectral leakage
        window = np.hanning(len(signal))
        signal = signal * window

        # FFT
        fft_vals = np.fft.rfft(signal)
        power = np.abs(fft_vals) ** 2

        n = len(signal)
        assumed_fps = 30.0
        idx_5hz = max(2, int(n * 5.0 / assumed_fps))

        total_power = float(np.sum(power[1:]))  # exclude DC

        if total_power < 1e-6:
            hf_ratio = 0.0
            peak_prominence = 0.0
            hf_sig = 0.0
            peak_sig = 0.0
        else:
            hf_power = float(np.sum(power[idx_5hz:]))
            hf_ratio = hf_power / total_power

            peak_idx = np.argmax(power[idx_5hz:]) + idx_5hz
            peak_val = float(power[peak_idx])
            mean_power = float(np.mean(power[idx_5hz:]))
            peak_prominence = (peak_val / (mean_power + 1e-10)) if mean_power > 0 else 0.0

            hf_sig = float(max(0.0, 1.0 - min(1.0, (hf_ratio - 0.10) / 0.20)))
            peak_sig = float(max(0.0, 1.0 - min(1.0, (peak_prominence - 2.0) / 6.0)))

        fft_score = 0.55 * hf_sig + 0.45 * peak_sig

        # ============================================================
        # Part B: Inter-frame difference pattern analysis
        # ============================================================
        # Video compression creates I-frame / P-frame structure:
        #   - P-frames are predicted from previous → small differences
        #   - I-frames are independent → larger differences
        # This creates periodic spikes in frame-to-frame difference
        # energy every ~15-30 frames (GOP length).
        #
        # Real camera captures have smooth, Gaussian frame differences
        # with no periodic pattern.
        #
        # Also: video frames have higher inter-frame correlation than
        # real camera captures because compression smooths noise.
        ifd_score = 0.5  # default if not enough patches
        if len(self._patch_buffer) >= self.min_evidence:
            diffs = []
            correlations = []
            for i in range(1, len(self._patch_buffer)):
                prev = self._patch_buffer[i - 1]
                curr = self._patch_buffer[i]
                # Mean absolute difference per pixel
                diff = float(np.mean(np.abs(curr - prev)))
                diffs.append(diff)
                # Pixel-wise Pearson correlation between frames
                std_p = float(np.std(prev))
                std_c = float(np.std(curr))
                if std_p > 0.5 and std_c > 0.5:
                    corr = float(np.corrcoef(prev.flatten(),
                                             curr.flatten())[0, 1])
                    correlations.append(corr)

            if len(diffs) >= 16:
                diffs_arr = np.array(diffs)
                mean_diff = float(np.mean(diffs_arr))

                # Metric B1: regularity of differences (FFT of diff series)
                # Real: smooth/random → flat FFT.  Video: periodic → peaks.
                if mean_diff > 0.1:
                    d_signal = diffs_arr - np.mean(diffs_arr)
                    d_fft = np.abs(np.fft.rfft(d_signal * np.hanning(len(d_signal))))
                    d_power = d_fft ** 2
                    if len(d_power) > 2 and float(np.sum(d_power[1:])) > 1e-10:
                        d_peak = float(np.max(d_power[1:]))
                        d_mean = float(np.mean(d_power[1:]))
                        d_prominence = d_peak / (d_mean + 1e-10)
                        # High prominence → periodic GOP pattern → replay
                        diff_regularity_sig = float(
                            np.clip(1.0 - (d_prominence - 2.0) / 8.0, 0.0, 1.0))
                    else:
                        diff_regularity_sig = 0.5
                else:
                    diff_regularity_sig = 0.5

                # Metric B2: mean inter-frame correlation
                # Re-captured video: very high mean correlation (> 0.990)
                # because compression smooths high-frequency noise.
                # Real camera: lower correlation (< 0.985) due to
                # independent sensor noise per frame.
                if correlations:
                    mean_corr = float(np.mean(correlations))
                    # Map [0.980, 0.998] → [1.0, 0.0]
                    corr_sig = float(
                        np.clip(1.0 - (mean_corr - 0.980) / 0.018, 0.0, 1.0))
                else:
                    corr_sig = 0.5

                ifd_score = 0.50 * diff_regularity_sig + 0.50 * corr_sig
            # end if enough diffs

        self.last_metrics = {
            'hf_ratio': hf_ratio,
            'peak_prominence': peak_prominence,
            'fft_score': fft_score,
            'ifd_score': ifd_score,
        }

        # Combine FFT + inter-frame analysis
        # If FFT has no power (static), rely more on inter-frame analysis.
        if total_power < 1e-6:
            score = ifd_score  # FFT unusable, trust inter-frame only
        else:
            score = 0.45 * fft_score + 0.55 * ifd_score

        return float(np.clip(score, 0.0, 1.0))

    def get_signal(self):
        if len(self._brightness_buffer) < self.min_evidence:
            return 0.0  # suspicious until proven real
        return self.last_score

    def reset(self):
        self._brightness_buffer.clear()
        self._patch_buffer.clear()
        self.last_score = 0.0
        self.last_metrics = {}


# =============================================================================
# Focal Plane Detector (depth-of-field uniformity analysis)
# =============================================================================

class FocalPlaneDetector:
    """Detect video replay via re-capture noise and compression artifacts.

    When a video is played on a phone and re-captured by a webcam, the
    image undergoes double processing that leaves detectable traces:

      1. **Noise kurtosis** — Camera sensor noise is Gaussian (kurtosis
         ≈ 3.0).  Video compression quantises DCT coefficients, creating
         non-Gaussian noise with heavier tails (kurtosis > 3.5).  When
         that compressed video is re-captured, the resulting noise
         distribution is measurably different from single-capture noise.

      2. **Block-boundary autocorrelation** — H.264 / H.265 operate on
         4×4 and 8×8 blocks.  Quantisation creates subtle brightness
         steps at block boundaries that survive display→re-capture.
         These appear as elevated spatial autocorrelation at lags 4 and
         8 in the high-pass residual of the face ROI.

      3. **Noise spatial variance** — Real camera noise has uniform
         spatial variance across the face ROI (sensor noise is position-
         independent).  Re-captured video has NON-uniform noise variance
         because compression allocates different quality to flat vs.
         textured regions (macro-block QP variation).

    These metrics detect properties of the *medium pipeline* (compression
    + display + re-capture) that cannot be removed by higher display
    resolution, unlike DoF or moiré detectors.
    """

    def __init__(self, history_size=20, min_evidence=10):
        self.history_size = history_size
        self.min_evidence = min_evidence
        self.score_history = []
        self.last_score = 0.0
        self.last_metrics = {}

    def update(self, frame_bgr, face_location):
        """Analyze face ROI noise for re-capture artifacts.

        Args:
            frame_bgr:     Full BGR frame.
            face_location: (top, right, bottom, left).

        Returns:
            float signal [0.0, 1.0] — 1.0 = real face (clean noise),
                                       0.0 = replay (compression artifacts).
        """
        top, right, bottom, left = (int(v) for v in face_location)
        h, w = frame_bgr.shape[:2]
        top, left = max(0, top), max(0, left)
        bottom, right = min(h, bottom), min(w, right)

        roi = frame_bgr[top:bottom, left:right]
        if roi.size == 0 or roi.shape[0] < 48 or roi.shape[1] < 48:
            return self.get_signal()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        score = self._analyze(gray)
        self.last_score = score
        self.score_history.append(score)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)

        return self.get_signal()

    def _analyze(self, gray):
        """Compute noise-based re-capture metrics on 128×128 face ROI."""
        gray_f = gray.astype(np.float64)

        # Extract high-pass noise residual (original − median-filtered)
        median_filtered = cv2.medianBlur(gray, 5).astype(np.float64)
        noise = gray_f - median_filtered

        # ----- Metric 1: Noise kurtosis -----
        # Real camera noise: kurtosis ≈ 3.0 (Gaussian)
        # Re-captured video: kurtosis > 3.5 (compression tails)
        noise_flat = noise.flatten()
        noise_std = float(np.std(noise_flat))
        if noise_std > 0.5:
            noise_mean = float(np.mean(noise_flat))
            centered = noise_flat - noise_mean
            kurtosis = float(np.mean(centered ** 4) / (noise_std ** 4))
        else:
            kurtosis = 3.0  # no noise → assume Gaussian

        # Score: low kurtosis = real (≈ 3.0), high = replay (> 4.0)
        # Map [2.5, 5.0] → [1.0, 0.0]
        kurtosis_sig = float(np.clip(1.0 - (kurtosis - 2.5) / 2.5, 0.0, 1.0))

        # ----- Metric 2: Block-boundary autocorrelation -----
        # Video codecs use 4×4 / 8×8 / 16×16 blocks.  Measure spatial
        # autocorrelation of noise at lags 4 and 8 in both axes.
        acorr_4 = self._block_autocorrelation(noise, lag=4)
        acorr_8 = self._block_autocorrelation(noise, lag=8)
        max_acorr = max(acorr_4, acorr_8)

        # Real camera noise: autocorrelation < 0.05 (spatially random)
        # Re-captured video: autocorrelation > 0.10 (block aligned)
        # Map [0.0, 0.15] → [1.0, 0.0]
        block_sig = float(np.clip(1.0 - max_acorr / 0.15, 0.0, 1.0))

        # ----- Metric 3: Noise variance uniformity across blocks -----
        # Divide into 8×8 blocks, compute noise variance per block.
        # Real face: uniform variance (position-independent sensor noise).
        # Replay: non-uniform variance (QP variation across macroblocks).
        block_sz = 16  # 128 / 8 = 16-pixel blocks
        num_blocks = 128 // block_sz
        variances = []
        for by in range(num_blocks):
            for bx in range(num_blocks):
                block = noise[by*block_sz:(by+1)*block_sz,
                              bx*block_sz:(bx+1)*block_sz]
                variances.append(float(np.var(block)))

        variances = np.array(variances)
        mean_var = float(np.mean(variances))
        if mean_var > 0.1:
            cv_noise = float(np.std(variances) / mean_var)
        else:
            cv_noise = 0.0

        # Real face: low CV (< 0.3, uniform sensor noise)
        # Replay: high CV (> 0.5, macro-block QP variation)
        # Map [0.15, 0.60] → [1.0, 0.0]
        uniformity_sig = float(np.clip(1.0 - (cv_noise - 0.15) / 0.45, 0.0, 1.0))

        self.last_metrics = {
            'noise_kurtosis': kurtosis,
            'block_acorr_4': acorr_4,
            'block_acorr_8': acorr_8,
            'noise_cv': cv_noise,
            'kurtosis_sig': kurtosis_sig,
            'block_sig': block_sig,
            'uniformity_sig': uniformity_sig,
        }

        # Fusion — block autocorrelation is the strongest signal
        score = (0.30 * kurtosis_sig +
                 0.40 * block_sig +
                 0.30 * uniformity_sig)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _block_autocorrelation(noise, lag):
        """Compute normalised autocorrelation of noise at given pixel lag.

        Measures how correlated the noise is with a shifted copy of
        itself.  Video compression block boundaries create periodic
        correlation at multiples of the block size.
        """
        h, w = noise.shape
        if lag >= h or lag >= w:
            return 0.0

        # Horizontal autocorrelation at this lag
        a = noise[:, :w-lag]
        b = noise[:, lag:]
        std_a = float(np.std(a))
        std_b = float(np.std(b))
        if std_a < 0.1 or std_b < 0.1:
            h_corr = 0.0
        else:
            h_corr = abs(float(np.mean((a - np.mean(a)) * (b - np.mean(b)))
                               / (std_a * std_b)))

        # Vertical autocorrelation at this lag
        a = noise[:h-lag, :]
        b = noise[lag:, :]
        std_a = float(np.std(a))
        std_b = float(np.std(b))
        if std_a < 0.1 or std_b < 0.1:
            v_corr = 0.0
        else:
            v_corr = abs(float(np.mean((a - np.mean(a)) * (b - np.mean(b)))
                               / (std_a * std_b)))

        return (h_corr + v_corr) / 2.0

    def get_signal(self):
        if len(self.score_history) < self.min_evidence:
            return 0.0  # suspicious until proven real
        if len(self.score_history) < 5:
            return float(np.mean(self.score_history))
        return float(np.median(self.score_history))

    def reset(self):
        self.score_history.clear()
        self.last_score = 0.0
        self.last_metrics = {}


# =============================================================================
# C++ Texture Analyzer Wrapper (LBP micro-texture anti-spoofing)
# =============================================================================

class TextureAnalyzerWrapper:
    """Wrapper around the C++ TextureAnalyzer for anti-spoofing.

    Uses the compiled pybind11 module face_processor_cpp.TextureAnalyzer
    which provides:
      - Laplacian sharpness (sharper = more real)
      - LBP micro-texture entropy (richer patterns = more real)
      - High-frequency energy ratio (more HF detail = more real)

    The C++ implementation is significantly faster than pure Python and
    captures sub-pixel rendering artifacts that differ between screens
    and real skin, even on high-PPI OLED displays.
    """

    def __init__(self, history_size=20, min_evidence=8):
        self.history_size = history_size
        self.min_evidence = min_evidence
        self.score_history = []
        self.last_score = 0.5
        self.last_metrics = {}

        # Try to load the C++ module
        try:
            from face_processor_cpp import TextureAnalyzer as _CppAnalyzer
            self._analyzer = _CppAnalyzer(
                100.0,   # sharpness_threshold
                5.0,     # entropy_threshold
                0.10,    # hf_threshold
            )
            self._available = True
        except ImportError:
            self._analyzer = None
            self._available = False

    def update(self, frame_bgr, face_location):
        """Analyze face ROI texture via C++ LBP / sharpness.

        Args:
            frame_bgr:     Full BGR frame.
            face_location: (top, right, bottom, left).

        Returns:
            float signal [0.0, 1.0] — 1.0 = real skin texture,
                                       0.0 = screen / print texture.
        """
        if not self._available:
            return 0.5  # neutral if C++ module not available

        top, right, bottom, left = (int(v) for v in face_location)
        h, w = frame_bgr.shape[:2]
        top, left = max(0, top), max(0, left)
        bottom, right = min(h, bottom), min(w, right)

        roi = frame_bgr[top:bottom, left:right]
        if roi.size == 0 or roi.shape[0] < 32 or roi.shape[1] < 32:
            return self.get_signal()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # Convert to flat list of uint8 for C++ API
        pixels = gray.flatten().tolist()

        try:
            result = self._analyzer.analyze(pixels, 128, 128)
            score = float(result.texture_score)
            self.last_metrics = {
                'sharpness': result.sharpness,
                'lbp_entropy': result.lbp_entropy,
                'hf_energy': result.hf_energy,
                'unique_patterns': result.num_unique_patterns,
            }
        except Exception:
            score = 0.5

        self.last_score = score
        self.score_history.append(score)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)

        return self.get_signal()

    def get_signal(self):
        if not self._available:
            return 0.5  # cannot analyze — stay neutral
        if len(self.score_history) < self.min_evidence:
            return 0.0  # suspicious until proven real
        if len(self.score_history) < 5:
            return float(np.mean(self.score_history))
        return float(np.median(self.score_history))

    def reset(self):
        self.score_history.clear()
        self.last_score = 0.0
        self.last_metrics = {}


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

    def __init__(self, history_size=25, min_evidence=8):
        self.history_size = history_size
        self.min_evidence = min_evidence
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
        """Median-smoothed reflection-detection signal over recent frames.

        Returns 0.0 (suspicious) until at least ``min_evidence`` frames
        have been analysed.
        """
        if len(self.score_history) < self.min_evidence:
            return 0.0   # suspicious until proven otherwise
        if len(self.score_history) < 5:
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
# Optical Flow Consistency Detector
# =============================================================================

class OpticalFlowDetector:
    """Detect video replay attacks via optical flow analysis.

    When a real 3D face moves in front of a camera, different facial
    regions produce different flow vectors:
      - The nose (closest) displaces more than the ears (further).
      - Eyes, mouth, and jaw move semi-independently (non-rigid).

    A video replayed on a flat screen produces *uniform* optical flow
    across the entire face ROI because every pixel is at the same depth
    (the screen surface).  The flow field is essentially a single rigid
    affine transform.

    Metrics fused:
      1. Flow variance ratio (40%) — high inner variance relative to
         magnitude means non-rigid / depth-varying motion (real face).
      2. Angular dispersion  (35%) — real faces produce diverse flow
         directions; flat replays produce near-parallel vectors.
      3. Residual after affine fit (25%) — how much flow remains after
         subtracting the best rigid-affine approximation.

    Uses dense Farneback optical flow on the face ROI.
    """

    ANALYSIS_SIZE = 96  # face ROI resized to this for flow computation

    def __init__(self, history_size=20,
                 variance_threshold=0.15,
                 angular_threshold=0.30,
                 residual_threshold=0.20):
        self.history_size = history_size
        self.variance_threshold = variance_threshold
        self.angular_threshold = angular_threshold
        self.residual_threshold = residual_threshold

        self._prev_gray = None
        self._prev_full_gray = None  # full-frame for background flow
        self.min_evidence = 8
        self.score_history = []
        self.last_score = 0.0

    def update(self, frame_bgr, face_location):
        """Compute optical flow on the face ROI between consecutive frames.

        Also computes full-frame background flow and compares it with
        face-region flow.  On a phone replay, face and background
        (phone bezel / surrounding area) move as one rigid body, so
        their flow fields are highly correlated.  On a real face, the
        face moves independently of the background.

        Args:
            frame_bgr:     Full BGR frame.
            face_location: (top, right, bottom, left).

        Returns:
            float signal [0.0, 1.0] — 1.0 = real face, 0.0 = replay.
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

        # Down-sample full frame for background flow
        full_sz = self.ANALYSIS_SIZE * 2
        full_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        full_gray = cv2.resize(full_gray, (full_sz, full_sz))

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_full_gray = full_gray
            return self.get_signal()

        # Dense optical flow on face ROI
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Dense optical flow on full frame (for background comparison)
        full_flow = cv2.calcOpticalFlowFarneback(
            self._prev_full_gray, full_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        self._prev_gray = gray
        self._prev_full_gray = full_gray

        # Face-only analysis (non-rigidity metrics)
        face_score = self._analyze_flow(flow)

        # Face-vs-background analysis (independence metric)
        bg_score = self._analyze_face_vs_background(
            full_flow, face_location, h, w, full_sz
        )

        # Combine: face must look non-rigid AND move independently
        # of the background.  bg_score is the strongest phone-replay
        # discriminator, so give it high weight.
        score = 0.45 * face_score + 0.55 * bg_score

        self.last_score = score
        self.score_history.append(score)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)

        return self.get_signal()

    def _analyze_flow(self, flow):
        """Analyze flow field for non-rigidity indicators."""
        sz = self.ANALYSIS_SIZE
        fx = flow[:, :, 0].flatten()
        fy = flow[:, :, 1].flatten()
        mag = np.sqrt(fx**2 + fy**2)

        mean_mag = np.mean(mag)
        if mean_mag < 0.3:
            # Negligible motion — cannot distinguish, assume neutral
            return 0.5

        # 1. Flow variance ratio: std(magnitude) / mean(magnitude)
        #    Real face: high (depth-dependent displacement).
        #    Flat replay: low (uniform displacement).
        variance_ratio = float(np.std(mag) / (mean_mag + 1e-8))
        var_sig = min(1.0, variance_ratio / self.variance_threshold)

        # 2. Angular dispersion: circular std of flow directions.
        #    Real face: varied angles (eyes move differently from jaw).
        #    Flat replay: near-parallel vectors.
        angles = np.arctan2(fy, fx)
        # Circular mean direction
        mean_sin = np.mean(np.sin(angles[mag > 0.5]))
        mean_cos = np.mean(np.cos(angles[mag > 0.5]))
        R = np.sqrt(mean_sin**2 + mean_cos**2)  # resultant length [0,1]
        # Circular variance = 1 - R: high = dispersed, low = clustered
        circ_var = 1.0 - R if np.isfinite(R) else 0.0
        ang_sig = min(1.0, circ_var / self.angular_threshold)

        # 3. Affine residual: fit best affine transform to flow,
        #    measure residual.  Flat surface ≈ 0 residual.
        y_coords, x_coords = np.mgrid[0:sz, 0:sz]
        pts = np.column_stack([x_coords.flatten(), y_coords.flatten(),
                               np.ones(sz * sz)])
        # Solve  A @ [x, y, 1]^T ≈ [fx, fy]^T  via least squares
        try:
            affine_x, _, _, _ = np.linalg.lstsq(pts, fx, rcond=None)
            affine_y, _, _, _ = np.linalg.lstsq(pts, fy, rcond=None)
            pred_fx = pts @ affine_x
            pred_fy = pts @ affine_y
            residual = np.sqrt(np.mean((fx - pred_fx)**2 + (fy - pred_fy)**2))
            residual_norm = residual / (mean_mag + 1e-8)
        except np.linalg.LinAlgError:
            residual_norm = 0.0
        res_sig = min(1.0, residual_norm / self.residual_threshold)

        score = 0.40 * var_sig + 0.35 * ang_sig + 0.25 * res_sig
        return float(np.clip(score, 0.0, 1.0))

    def _analyze_face_vs_background(self, full_flow, face_loc, orig_h, orig_w, full_sz):
        """Compare face-region flow to background-region flow.

        On a phone replay, the face and phone border / surrounding area
        move together as one rigid body.  Mean flow vectors in the face
        region and background region are highly correlated (similar
        direction and magnitude).

        On a real face in front of the camera, the background is mostly
        static while the face moves, OR they move independently.  The
        difference in mean flow is large.

        Returns:
            float [0.0, 1.0]  — 1.0 = independent motion (real face),
                                 0.0 = correlated motion (replay).
        """
        top, right, bottom, left = face_loc
        # Map face bbox to the down-sampled full-flow coordinate space
        if orig_h < 1 or orig_w < 1:
            return 0.5  # neutral

        scale_y = full_sz / orig_h
        scale_x = full_sz / orig_w
        ft = int(max(0, top * scale_y))
        fb = int(min(full_sz, bottom * scale_y))
        fl = int(max(0, left * scale_x))
        fr = int(min(full_sz, right * scale_x))

        if fb - ft < 4 or fr - fl < 4:
            return 0.5

        # Face-region mean flow
        face_flow = full_flow[ft:fb, fl:fr]
        face_mean_fx = float(np.mean(face_flow[:, :, 0]))
        face_mean_fy = float(np.mean(face_flow[:, :, 1]))
        face_mag = np.sqrt(face_mean_fx**2 + face_mean_fy**2)

        # Background-region mean flow (everything outside the face bbox)
        mask = np.ones((full_sz, full_sz), dtype=bool)
        mask[ft:fb, fl:fr] = False
        bg_flow_x = full_flow[:, :, 0][mask]
        bg_flow_y = full_flow[:, :, 1][mask]

        if bg_flow_x.size < 100:
            return 0.5

        bg_mean_fx = float(np.mean(bg_flow_x))
        bg_mean_fy = float(np.mean(bg_flow_y))
        bg_mag = np.sqrt(bg_mean_fx**2 + bg_mean_fy**2)

        # If neither face nor background moves much, neutral
        if face_mag < 0.2 and bg_mag < 0.2:
            return 0.5

        # ---- Metric 1: Flow-difference magnitude ----
        # How different are the mean motion vectors?
        diff_mag = np.sqrt((face_mean_fx - bg_mean_fx)**2 +
                           (face_mean_fy - bg_mean_fy)**2)
        # Real face: diff_mag > 0.5 (face moves, bg doesn't, or differently)
        # Phone replay: diff_mag ≈ 0 (everything translates together)
        diff_sig = float(min(1.0, diff_mag / 0.8))

        # ---- Metric 2: Direction cosine similarity ----
        # Are face and background moving in the same direction?
        dot = face_mean_fx * bg_mean_fx + face_mean_fy * bg_mean_fy
        norms = (face_mag + 1e-8) * (bg_mag + 1e-8)
        cos_sim = dot / norms
        # Real face: cos_sim ≈ 0 or negative (independent directions)
        # Phone replay: cos_sim ≈ 1.0 (same direction)
        # Penalise high similarity (> 0.6 → suspicious)
        if cos_sim > 0.6:
            dir_sig = float(max(0.0, 1.0 - (cos_sim - 0.6) / 0.4))
        else:
            dir_sig = 1.0

        # ---- Metric 3: Magnitude ratio ----
        # On a phone replay, face_mag ≈ bg_mag (rigid body).
        # Real face: face_mag >> bg_mag OR bg_mag ≈ 0.
        if max(face_mag, bg_mag) > 0.2:
            ratio = min(face_mag, bg_mag) / (max(face_mag, bg_mag) + 1e-8)
            # Real: ratio far from 1.0.  Phone: ratio ≈ 1.0
            ratio_sig = float(max(0.0, 1.0 - ratio))
        else:
            ratio_sig = 0.5

        # Fusion
        score = 0.40 * diff_sig + 0.35 * dir_sig + 0.25 * ratio_sig
        return float(np.clip(score, 0.0, 1.0))

    def get_signal(self):
        """Median-smoothed optical flow signal.

        Returns 0.0 (suspicious) until min_evidence frames analysed.
        """
        if len(self.score_history) < self.min_evidence:
            return 0.0  # suspicious until proven otherwise
        if len(self.score_history) < 5:
            return float(np.mean(self.score_history))
        return float(np.median(self.score_history))

    def reset(self):
        self._prev_gray = None
        self._prev_full_gray = None
        self.score_history.clear()
        self.last_score = 0.0


# =============================================================================
# Depth Geometry Detector (landmark ratio changes under rotation)
# =============================================================================

class DepthGeometryDetector:
    """Estimate 3D face depth from how landmark ratios change over time.

    A real 3D face produces perspective-dependent changes in inter-
    landmark distance ratios as the head rotates.  Key observations:

      - When turning left, the left-eye-to-nose distance shortens
        while the right-eye-to-nose distance lengthens (foreshortening).
      - The nose-tip moves non-linearly relative to the jaw outline.
      - Upper-face vs lower-face height ratio changes with pitch.

    A flat photo or video on a screen only produces *affine* ratio
    changes (linear scaling), not the non-linear perspective shifts
    a real 3D face exhibits.

    Metrics:
      1. Lateral asymmetry variance (40%) — how much left/right eye-
         to-nose ratios diverge over time.
      2. Non-linear nose displacement (35%) — nose tip moves non-
         linearly relative to jaw width under yaw rotation.
      3. Vertical ratio variance (25%) — upper/lower face height
         ratio changes under pitch rotation.
    """

    def __init__(self, history_size=25,
                 asymmetry_threshold=0.008,
                 nonlinear_threshold=0.005,
                 vertical_threshold=0.006):
        self.history_size = history_size
        self.asymmetry_threshold = asymmetry_threshold
        self.nonlinear_threshold = nonlinear_threshold
        self.vertical_threshold = vertical_threshold

        self._ratio_history = []  # list of ratio feature dicts
        self.last_score = 0.0

    def update(self, landmarks_dict):
        """Process one frame's landmarks.

        Args:
            landmarks_dict: face_recognition.face_landmarks() dict

        Returns:
            float signal [0.0, 1.0] — 1.0 = real 3D face.
        """
        ratios = self._extract_ratios(landmarks_dict)
        if ratios is None:
            return self.get_signal()

        self._ratio_history.append(ratios)
        if len(self._ratio_history) > self.history_size:
            self._ratio_history.pop(0)

        self.last_score = self._compute_signal()
        return self.last_score

    def _extract_ratios(self, lm):
        """Extract perspective-sensitive ratios from landmarks."""
        try:
            nose_tip = np.array(lm['nose_tip'][2], dtype=float)
            left_eye = np.array(lm['left_eye'][0], dtype=float)
            right_eye = np.array(lm['right_eye'][3], dtype=float)
            left_jaw = np.array(lm['chin'][0], dtype=float)
            right_jaw = np.array(lm['chin'][16], dtype=float)
            chin_bottom = np.array(lm['chin'][8], dtype=float)
            bridge_top = np.array(lm['nose_bridge'][0], dtype=float)

            jaw_width = np.linalg.norm(right_jaw - left_jaw)
            if jaw_width < 5:
                return None

            # Lateral: eye-to-nose ratios (perspective-sensitive)
            left_eye_nose = np.linalg.norm(left_eye - nose_tip) / jaw_width
            right_eye_nose = np.linalg.norm(right_eye - nose_tip) / jaw_width

            # Nose horizontal offset (non-linear under yaw)
            center_x = (left_jaw[0] + right_jaw[0]) / 2.0
            nose_offset = (nose_tip[0] - center_x) / jaw_width

            # Vertical: upper vs lower face ratio (sensitive to pitch)
            face_height = np.linalg.norm(bridge_top - chin_bottom)
            if face_height < 5:
                return None
            upper = np.linalg.norm(bridge_top - nose_tip)
            lower = np.linalg.norm(nose_tip - chin_bottom)
            vertical_ratio = upper / (upper + lower + 1e-8)

            return {
                'left_eye_nose': left_eye_nose,
                'right_eye_nose': right_eye_nose,
                'nose_offset': nose_offset,
                'vertical_ratio': vertical_ratio,
                'jaw_width': jaw_width,
            }
        except (KeyError, IndexError):
            return None

    def _compute_signal(self):
        """Compute depth geometry signal from ratio history."""
        if len(self._ratio_history) < 5:
            return 0.0

        # Extract time series
        left_en = [r['left_eye_nose'] for r in self._ratio_history]
        right_en = [r['right_eye_nose'] for r in self._ratio_history]
        nose_off = [r['nose_offset'] for r in self._ratio_history]
        vert_rat = [r['vertical_ratio'] for r in self._ratio_history]

        # 1. Lateral asymmetry variance
        #    When head turns, left/right eye-nose ratios should diverge.
        #    Compute ratio_diff = |left - right| over time, take variance.
        ratio_diffs = [abs(l - r) for l, r in zip(left_en, right_en)]
        asym_var = float(np.var(ratio_diffs))
        asym_sig = min(1.0, asym_var / self.asymmetry_threshold)

        # 2. Non-linear nose displacement
        #    On a real face, nose_offset vs jaw_width relationship is
        #    non-linear.  Measure residual after linear fit.
        offsets = np.array(nose_off)
        if np.std(offsets) > 0.005:  # enough yaw motion to analyze
            t = np.arange(len(offsets), dtype=float)
            # Fit linear trend
            coeffs = np.polyfit(t, offsets, 1)
            linear_pred = np.polyval(coeffs, t)
            residual = np.std(offsets - linear_pred)
            nonlinear_sig = min(1.0, residual / self.nonlinear_threshold)
        else:
            nonlinear_sig = 0.0

        # 3. Vertical ratio variance (pitch-dependent)
        vert_var = float(np.var(vert_rat))
        vert_sig = min(1.0, vert_var / self.vertical_threshold)

        score = 0.40 * asym_sig + 0.35 * nonlinear_sig + 0.25 * vert_sig
        return float(np.clip(score, 0.0, 1.0))

    def get_signal(self):
        return self.last_score

    def reset(self):
        self._ratio_history.clear()
        self.last_score = 0.0


# =============================================================================
# Challenge-Response Detector (interactive liveness verification)
# =============================================================================

class ChallengeResponseDetector:
    """Head-pose challenge-response with 3D geometric consistency analysis.

    Randomized sequence of: TURN LEFT, TURN RIGHT, LOOK UP, LOOK DOWN

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

        # Recentering state — wait for face to return to neutral
        # before starting the next challenge
        self._recentering = False
        self._recenter_count = 0
        self._pending_challenge = None
        self._neutral_v_ratio = None   # captured from user's first neutral frame

        # Consecutive-frame confirmation counter for pose checks
        self._confirm_count = 0

        # 3D consistency analysis state
        self._pose_snapshots = {}       # {challenge_id: feature_vector}
        self._consistency_score = 0.0
        self._euc_lr = 0.0             # euclidean distance left↔right
        self._euc_ud = 0.0             # euclidean distance up↔down
        self._cos_lr = 1.0             # cosine similarity left↔right
        self._cos_ud = 1.0             # cosine similarity up↔down

    def start(self):
        """Start a new challenge-response sequence (randomized order)."""
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
        random.shuffle(self._remaining)
        self._confirm_count = 0
        self._recentering = False
        self._recenter_count = 0
        self._neutral_v_ratio = None
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
        self._pending_challenge = (challenge_id, instruction)
        # Enter recentering phase — wait for neutral pose before showing
        # the next challenge instruction.  Skip recenter for the very
        # first challenge (nothing to recenter from).
        if self.challenges_passed == 0 and self.challenges_failed == 0:
            # First challenge — start immediately, reset confirm counter
            self.current_challenge = challenge_id
            self.current_instruction = instruction
            self.challenge_start_time = time.time()
            self._baseline = {}
            self._confirm_count = 0
            self._recentering = False
        else:
            self._confirm_count = 0
            self.current_challenge = None
            self.current_instruction = "Return to center"
            self._recentering = True
            self._recenter_count = 0

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

    def _is_centered(self, landmarks_dict):
        """Check if the face is in a roughly neutral / center position.

        Uses the user's personal neutral vertical ratio (captured during
        the first challenge) instead of a hardcoded constant.

        Returns True when nose offset is small and vertical ratio is
        close to the user's own neutral.
        """
        HORZ_NEUTRAL = 0.1
        VERT_NEUTRAL = 0.1

        try:
            nose = np.array(landmarks_dict['nose_tip'][2], dtype=float)
            chin = landmarks_dict['chin']
            left_jaw = np.array(chin[0], dtype=float)
            right_jaw = np.array(chin[16], dtype=float)
            face_width = np.linalg.norm(right_jaw - left_jaw)
            if face_width < 1:
                return False
            face_center_x = (left_jaw[0] + right_jaw[0]) / 2.0
            h_offset = abs((nose[0] - face_center_x) / face_width)

            if h_offset >= HORZ_NEUTRAL:
                return False

            # Vertical check using eye-line pitch ratio (consistent with
            # challenge detection — measures actual head pitch, not frame drift)
            left_eye_inner = np.array(landmarks_dict['left_eye'][3], dtype=float)
            right_eye_inner = np.array(landmarks_dict['right_eye'][0], dtype=float)
            chin_bottom = np.array(chin[8], dtype=float)
            eye_center_y = (left_eye_inner[1] + right_eye_inner[1]) / 2.0
            eye_to_nose = (nose[1] - eye_center_y) / face_width
            nose_to_chin = (chin_bottom[1] - nose[1]) / face_width
            total_height = eye_to_nose + nose_to_chin
            if total_height < 0.01:
                return False
            pitch_ratio = eye_to_nose / (total_height + 1e-6)

            if self._neutral_v_ratio is None:
                # First time — capture this as the user's neutral
                self._neutral_v_ratio = pitch_ratio
                return True

            return abs(pitch_ratio - self._neutral_v_ratio) < VERT_NEUTRAL
        except (KeyError, IndexError):
            return False

    def update(self, landmarks_dict):
        """Check if current challenge is completed.

        Flow: recenter (if needed) → show challenge → detect pose → next.

        Args:
            landmarks_dict: face_recognition.face_landmarks() result dict

        Returns:
            dict with challenge status and 3D consistency metrics
        """
        if self._paused:
            return self.get_result()

        if not self.is_active:
            return self.get_result()

        # --- Recentering phase: wait for neutral pose ---
        if self._recentering:
            RECENTER_CONFIRM = 3
            if self._is_centered(landmarks_dict):
                self._recenter_count += 1
                if self._recenter_count >= RECENTER_CONFIRM:
                    # Face is centered — capture reliable neutral baseline
                    # before starting the next challenge (uses eye-line
                    # pitch ratio consistent with challenge detection)
                    if self._neutral_v_ratio is None:
                        try:
                            nt = np.array(landmarks_dict['nose_tip'][2], dtype=float)
                            lei = np.array(landmarks_dict['left_eye'][3], dtype=float)
                            rei = np.array(landmarks_dict['right_eye'][0], dtype=float)
                            cb = np.array(landmarks_dict['chin'][8], dtype=float)
                            lj = np.array(landmarks_dict['chin'][0], dtype=float)
                            rj = np.array(landmarks_dict['chin'][16], dtype=float)
                            fw = np.linalg.norm(rj - lj)
                            if fw > 1:
                                ec_y = (lei[1] + rei[1]) / 2.0
                                e2n = (nt[1] - ec_y) / fw
                                n2c = (cb[1] - nt[1]) / fw
                                th = e2n + n2c
                                if th > 0.01:
                                    self._neutral_v_ratio = e2n / (th + 1e-6)
                        except (KeyError, IndexError):
                            pass
                    self._recentering = False
                    cid, instr = self._pending_challenge
                    self.current_challenge = cid
                    self.current_instruction = instr
                    self.challenge_start_time = time.time()
                    self._baseline = {}
                    self._confirm_count = 0
            else:
                self._recenter_count = 0
            return self.get_result()

        # --- Normal challenge evaluation ---
        if self.current_challenge is None:
            return self.get_result()

        elapsed = time.time() - self.challenge_start_time
        if elapsed > self.challenge_timeout:
            self.challenges_failed += 1
            # Re-add the timed-out challenge to the back of the queue
            # so it can be retried (instead of silently skipping it)
            timed_out = (self.current_challenge, self.current_instruction)
            self._remaining.append(timed_out)
            if self.challenges_failed >= self.num_challenges:
                # Too many failures overall — give up
                self.is_active = False
                self.is_complete = True
                self.passed = False
                self.current_challenge = None
                self.current_instruction = "FAILED - Too many timeouts"
            else:
                self._next_challenge()
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
        """Evaluate whether the current head-pose challenge was performed.

        Uses baseline-relative detection: the neutral position is captured
        on the first frame of each challenge and the user must move clearly
        away from it in the correct direction.  Requires multiple consecutive
        confirming frames to prevent single-frame jitter from passing.

        Thresholds:
          LEFT / RIGHT  — nose offset must change by ≥ 0.12 * face_width
          UP   / DOWN   — vertical ratio must change by ≥ 0.08
          Confirm frames — 5 consecutive frames above threshold
        """
        CONFIRM_REQUIRED = 5
        LR_THRESHOLD = 0.12
        UD_THRESHOLD = 0.05

        if self.current_challenge in ('TURN_LEFT', 'TURN_RIGHT'):
            try:
                nose = np.array(landmarks_dict['nose_tip'][2], dtype=float)
                chin = landmarks_dict['chin']
                left_jaw = np.array(chin[0], dtype=float)
                right_jaw = np.array(chin[16], dtype=float)
                face_width = np.linalg.norm(right_jaw - left_jaw)
                face_center_x = (left_jaw[0] + right_jaw[0]) / 2.0
                if face_width < 1:
                    self._confirm_count = 0
                    return False
                offset = (nose[0] - face_center_x) / face_width

                # Capture neutral baseline on first frame
                if 'horizontal_baseline' not in self._baseline:
                    self._baseline['horizontal_baseline'] = offset
                    self._confirm_count = 0
                    return False

                delta = offset - self._baseline['horizontal_baseline']

                if self.current_challenge == 'TURN_LEFT':
                    ok = delta < -LR_THRESHOLD
                else:
                    ok = delta > LR_THRESHOLD

                if ok:
                    self._confirm_count += 1
                    return self._confirm_count >= CONFIRM_REQUIRED
                else:
                    self._confirm_count = 0
                    return False
            except (KeyError, IndexError):
                self._confirm_count = 0
                return False

        elif self.current_challenge in ('LOOK_UP', 'LOOK_DOWN'):
            try:
                nose_tip = np.array(landmarks_dict['nose_tip'][2], dtype=float)
                bridge_top = np.array(landmarks_dict['nose_bridge'][0], dtype=float)
                chin_bottom = np.array(landmarks_dict['chin'][8], dtype=float)
                left_eye_inner = np.array(landmarks_dict['left_eye'][3], dtype=float)
                right_eye_inner = np.array(landmarks_dict['right_eye'][0], dtype=float)
                left_jaw = np.array(landmarks_dict['chin'][0], dtype=float)
                right_jaw = np.array(landmarks_dict['chin'][16], dtype=float)

                face_width = np.linalg.norm(right_jaw - left_jaw)
                if face_width < 1:
                    self._confirm_count = 0
                    return False

                # --- Signal 1: Eye-line to nose-tip vs nose-tip to chin ---
                # Under pitch, the nose moves relative to the eye-chin axis.
                # Normalized by face_width to be translation-invariant.
                eye_center_y = (left_eye_inner[1] + right_eye_inner[1]) / 2.0
                eye_to_nose = (nose_tip[1] - eye_center_y) / face_width
                nose_to_chin = (chin_bottom[1] - nose_tip[1]) / face_width

                total_height = eye_to_nose + nose_to_chin
                if total_height < 0.01:
                    self._confirm_count = 0
                    return False
                pitch_ratio = eye_to_nose / (total_height + 1e-6)

                # --- Signal 2: Nose bridge foreshortening ---
                # When looking up, the nose bridge (bridge_top → nose_tip)
                # appears shorter because it's angled toward the camera.
                # Normalize by face_width to be scale-invariant.
                bridge_len = np.linalg.norm(nose_tip - bridge_top) / face_width

                # Capture neutral baselines on first frame
                if 'pitch_ratio_baseline' not in self._baseline:
                    self._baseline['pitch_ratio_baseline'] = pitch_ratio
                    self._baseline['bridge_len_baseline'] = bridge_len
                    self._confirm_count = 0
                    return False

                delta_pitch = pitch_ratio - self._baseline['pitch_ratio_baseline']
                delta_bridge = bridge_len - self._baseline['bridge_len_baseline']

                # Fuse signals: both should agree on direction
                # Look up → pitch_ratio decreases (nose closer to eyes),
                #            bridge_len decreases (foreshortened)
                # Look down → pitch_ratio increases (nose farther from eyes),
                #              bridge_len increases (elongated)
                if self.current_challenge == 'LOOK_UP':
                    ok_pitch = delta_pitch < -UD_THRESHOLD
                    ok_bridge = delta_bridge < -UD_THRESHOLD * 0.5
                    ok = ok_pitch or ok_bridge
                else:
                    ok_pitch = delta_pitch > UD_THRESHOLD
                    ok_bridge = delta_bridge > UD_THRESHOLD * 0.5
                    ok = ok_pitch or ok_bridge

                if ok:
                    self._confirm_count += 1
                    return self._confirm_count >= CONFIRM_REQUIRED
                else:
                    self._confirm_count = 0
                    return False
            except (KeyError, IndexError):
                self._confirm_count = 0
                return False

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
            'recentering': self._recentering,
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
        self._confirm_count = 0
        self._recentering = False
        self._recenter_count = 0
        self._pending_challenge = None
        self._neutral_v_ratio = None
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
    """Two-gate sequential anti-spoofing / liveness detection system.

    Gate 1 — Blink verification (passive, binary):
      User must produce at least one confirmed blink.
      A photo can never blink — definitive anti-photo gate.

    Gate 2 — Challenge-response (active):
      Randomized head-pose challenges with 3D geometric consistency.
      Only starts after Gate 1 passes.  Threshold: challenge_threshold.

    Final decision: Both gates must pass for ≥ consec_live_required
    consecutive evaluations before is_live = True.

    Uses only MTCNN and dlib landmarks — no MediaPipe dependency.
    """

    def __init__(self, challenge_threshold=0.40,
                 num_challenges=4, challenge_timeout=10.0,
                 ear_threshold=0.21, consec_frames=3, blink_time_window=5.0,
                 depth_threshold=0.35):

        # Gate thresholds
        self.challenge_threshold = challenge_threshold
        self.depth_threshold = depth_threshold

        # Sub-detectors
        self.blink_detector = BlinkDetector(
            ear_threshold=ear_threshold,
            consec_frames=consec_frames,
            time_window=blink_time_window,
        )
        self.challenge_detector = ChallengeResponseDetector(num_challenges, challenge_timeout)
        self.depth_geometry_detector = DepthGeometryDetector()

        # State
        self.last_liveness_score = 0.0
        self.last_is_live = False
        self.last_signals = {
            "blink": 0.0, "challenge": 0.0, "depth_geometry": 0.0,
        }

        # Temporal consistency
        self._consec_live_count = 0
        self._consec_live_required = 5

        # Latched gate state — once a gate passes, its detectors stop
        # running and its score is frozen until reset().
        self._gate1_latched = False
        self._gate2_latched = False
        self._gate3_latched = False
        self._gate2_frozen_score = 0.0
        self._gate3_frozen_score = 0.0

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
        """Run anti-spoofing checks on one frame for one face.

        Three-gate sequential evaluation:
          Gate 1 (blink):      at least one blink detected
          Gate 2 (challenge):  randomized head-pose challenge-response
          Gate 3 (depth geometry): 3D landmark ratio changes

        All gates must pass simultaneously for ≥ consec_live_required
        consecutive frames before is_live = True.

        Returns:
            dict with liveness results, gate scores, and signal values
        """
        blink_signal = 0.0
        challenge_result = self.challenge_detector.get_result()
        depth_geometry_signal = 0.0

        # --- Gate 1 detectors (blink) — skip if latched ---
        if not self._gate1_latched and landmarks_dict is not None:
            left_eye = landmarks_dict.get('left_eye')
            right_eye = landmarks_dict.get('right_eye')
            if left_eye and right_eye:
                blink_signal = self.blink_detector.update(left_eye, right_eye)

        # --- Gate 2 detectors (challenge) — skip if latched ---
        if not self._gate2_latched and landmarks_dict is not None:
            if self.challenge_detector.is_active:
                challenge_result = self.challenge_detector.update(landmarks_dict)

        challenge_signal = challenge_result['signal']

        # --- Gate 3 detectors (depth geometry) ---
        if not self._gate3_latched and landmarks_dict is not None:
            depth_geometry_signal = self.depth_geometry_detector.update(landmarks_dict)

        # =============================================================
        # Gate 1 — Blink verification
        # =============================================================
        blink_detected = self._gate1_latched or len(self.blink_detector.blink_timestamps) > 0
        gate1_score = 1.0 if blink_detected else 0.0
        gate1_passed = blink_detected

        # Latch gate 1 on first pass
        if gate1_passed and not self._gate1_latched:
            self._gate1_latched = True

        # Auto-start challenge once Gate 1 passes for the first time
        if gate1_passed and not self.challenge_detector.is_active \
                and not self.challenge_detector.is_complete:
            self.challenge_detector.start()
            challenge_result = self.challenge_detector.get_result()
            challenge_signal = challenge_result['signal']

        # =============================================================
        # Gate 2 — Challenge-response (head-pose, randomized)
        # Only after Gate 1.  Active interaction proves live control.
        # =============================================================
        if self._gate2_latched:
            gate2_score = self._gate2_frozen_score
        else:
            gate2_score = challenge_signal
        gate2_passed = gate1_passed and (gate2_score >= self.challenge_threshold)

        # Latch gate 2 on first pass
        if gate2_passed and not self._gate2_latched:
            self._gate2_latched = True
            self._gate2_frozen_score = gate2_score

        # =============================================================
        # Gate 3 — Depth geometry (3D landmark ratio changes)
        # Only after Gate 1.  Passive, but robust against replay attacks.
        # =============================================================
        if self._gate3_latched:
            gate3_score = self._gate3_frozen_score
        else:
            gate3_score = depth_geometry_signal
        gate3_passed = gate1_passed and (gate3_score >= self.depth_threshold)

        # Latch gate 3 on first pass
        if gate3_passed and not self._gate3_latched:
            self._gate3_latched = True
            self._gate3_frozen_score = gate3_score

        # =============================================================
        # Final decision — all gates must hold consecutively
        # =============================================================
        all_passed = gate1_passed and gate2_passed and gate3_passed

        photo_score = gate1_score
        photo_passed = gate1_passed
        video_score = gate2_score
        video_passed = gate2_passed
        depth_score = gate3_score
        depth_passed = gate3_passed

        if all_passed:
            liveness_score = (gate2_score + gate3_score) / 2.0
        elif gate1_passed:
            liveness_score = 0.20
        else:
            liveness_score = 0.0

        if all_passed:
            self._consec_live_count += 1
        else:
            self._consec_live_count = 0

        is_live = self._consec_live_count >= self._consec_live_required

        # Update state
        self.last_liveness_score = liveness_score
        self.last_is_live = is_live
        self.last_signals = {
            "blink": blink_signal,
            "challenge": challenge_signal,
            "depth_geometry": depth_geometry_signal,
        }

        return {
            "is_live": is_live,
            "liveness_score": liveness_score,
            "photo_score": photo_score,
            "photo_passed": photo_passed,
            "video_score": video_score,
            "video_passed": video_passed,
            "depth_score": depth_score,
            "depth_passed": depth_passed,
            "gate1_passed": gate1_passed,
            "gate2_passed": gate2_passed,
            "gate3_passed": gate3_passed,
            "gate2_score": gate2_score,
            "gate3_score": gate3_score,
            "blink_signal": blink_signal,
            "challenge_signal": challenge_signal,
            "depth_geometry_signal": depth_geometry_signal,
            "challenge_result": challenge_result,
            "ear": self.blink_detector.last_ear,
            "ear_threshold": self.blink_detector.ear_threshold,
            "consistency_score": challenge_result.get('consistency_score', 0.0),
        }

    def reset(self):
        """Reset all detector state."""
        self.blink_detector.reset()
        self.challenge_detector.reset()
        self.depth_geometry_detector.reset()
        self.last_liveness_score = 0.0
        self.last_is_live = False
        self._consec_live_count = 0
        # Clear latched gates
        self._gate1_latched = False
        self._gate2_latched = False
        self._gate3_latched = False
        self._gate2_frozen_score = 0.0
        self._gate3_frozen_score = 0.0

    def release(self):
        """Release resources. No-op (no external models to close)."""
        pass
