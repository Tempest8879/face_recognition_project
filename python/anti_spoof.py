"""
Anti-Spoofing / Liveness Detection Module
==========================================
Multi-layer anti-spoofing system using only MTCNN and CNN for maximum accuracy:
  1. Eye blink detection (dlib 68-point landmarks via face_recognition)
  2. Head micro-movement detection (dlib landmarks)
  3. Mouth micro-movement detection (dlib landmarks)
  4. Challenge-response verification (blink/smile/open mouth/turn head)

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
# Challenge-Response Detector (interactive liveness verification)
# =============================================================================

class ChallengeResponseDetector:
    """Interactive challenge-response liveness verification.

    Issues random challenges (blink, smile, open mouth, turn head) and verifies
    the user performs them within a timeout. Extremely hard to spoof since it
    requires real-time responsive facial actions.
    """

    CHALLENGES = [
        ('BLINK', 'Please BLINK your eyes'),
        ('SMILE', 'Please SMILE'),
        ('OPEN_MOUTH', 'Please OPEN your MOUTH'),
        ('TURN_LEFT', 'Please turn head LEFT'),
        ('TURN_RIGHT', 'Please turn head RIGHT'),
    ]

    def __init__(self, num_challenges=3, challenge_timeout=5.0):
        self.num_challenges = num_challenges
        self.challenge_timeout = challenge_timeout

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

    def start(self):
        """Start a new challenge-response sequence."""
        self.challenges_passed = 0
        self.challenges_failed = 0
        self.is_active = True
        self.is_complete = False
        self.passed = False
        selected = random.sample(
            self.CHALLENGES,
            min(self.num_challenges, len(self.CHALLENGES))
        )
        self._remaining = list(selected)
        self._next_challenge()

    def _next_challenge(self):
        """Move to the next challenge or finish."""
        if self.challenges_passed >= self.num_challenges:
            self.is_active = False
            self.is_complete = True
            self.passed = True
            self.current_challenge = None
            self.current_instruction = "VERIFIED"
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

    def update(self, landmarks_dict, ear, mar):
        """Check if current challenge is completed.

        Args:
            landmarks_dict: face_recognition.face_landmarks() result dict
            ear: Current Eye Aspect Ratio
            mar: Current Mouth Aspect Ratio

        Returns:
            dict with challenge status
        """
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

        passed = self._check_challenge(landmarks_dict, ear, mar)

        if passed:
            self.challenges_passed += 1
            self._next_challenge()

        return self.get_result()

    def _check_challenge(self, landmarks_dict, ear, mar):
        """Evaluate whether the current challenge action was performed."""
        if self.current_challenge == 'BLINK':
            return ear < 0.19

        elif self.current_challenge == 'SMILE':
            try:
                top_lip = landmarks_dict['top_lip']
                chin = landmarks_dict['chin']
                left_corner = np.array(top_lip[0], dtype=float)
                right_corner = np.array(top_lip[6], dtype=float)
                mouth_width = np.linalg.norm(right_corner - left_corner)
                face_width = np.linalg.norm(
                    np.array(chin[16], dtype=float) - np.array(chin[0], dtype=float)
                )
                if face_width > 1:
                    ratio = mouth_width / face_width
                    if 'smile_baseline' not in self._baseline:
                        self._baseline['smile_baseline'] = ratio
                    elif ratio > self._baseline['smile_baseline'] * 1.12:
                        return True
            except (KeyError, IndexError):
                pass

        elif self.current_challenge == 'OPEN_MOUTH':
            return mar > 0.35

        elif self.current_challenge == 'TURN_LEFT':
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

        return False

    def get_result(self):
        time_remaining = 0.0
        if self.challenge_start_time and self.is_active:
            time_remaining = max(0, self.challenge_timeout - (time.time() - self.challenge_start_time))

        return {
            'is_active': self.is_active,
            'is_complete': self.is_complete,
            'passed': self.passed,
            'current_challenge': self.current_challenge,
            'current_instruction': self.current_instruction,
            'challenges_passed': self.challenges_passed,
            'num_challenges': self.num_challenges,
            'signal': 1.0 if self.passed else 0.0,
            'time_remaining': time_remaining,
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
        self._baseline = {}
        self._remaining = []


# =============================================================================
# Anti-Spoofing Orchestrator
# =============================================================================

class AntiSpoofing:
    """Multi-layer anti-spoofing / liveness detection system.

    Combines passive detection + active challenge-response:

    Passive signals (continuous):
      1. Eye blink detection (dlib 68-point EAR) — 25%
      2. Head micro-movement detection (dlib landmarks) — 20%
      3. Mouth micro-movement detection (dlib landmarks) — 20%

    Active signal:
      4. Challenge-response (blink/smile/open mouth/turn head) — 35%

    Uses only MTCNN and CNN (dlib) — no MediaPipe dependency.
    """

    def __init__(self, liveness_threshold=0.55,
                 blink_weight=0.25, movement_weight=0.20,
                 mouth_weight=0.20, challenge_weight=0.35,
                 num_challenges=3, challenge_timeout=5.0,
                 ear_threshold=0.2, consec_frames=2, blink_time_window=5.0,
                 movement_threshold=0.005, movement_history=15,
                 mar_movement_threshold=0.003, mar_history_size=20):

        self.liveness_threshold = liveness_threshold
        self.blink_weight = blink_weight
        self.movement_weight = movement_weight
        self.mouth_weight = mouth_weight
        self.challenge_weight = challenge_weight

        # Sub-detectors
        self.blink_detector = BlinkDetector(ear_threshold, consec_frames, blink_time_window)
        self.movement_detector = MovementDetector(movement_threshold, movement_history)
        self.mouth_detector = MouthMovementDetector(mar_movement_threshold, mar_history_size)
        self.challenge_detector = ChallengeResponseDetector(num_challenges, challenge_timeout)

        # State
        self.last_liveness_score = 0.0
        self.last_is_live = False
        self.last_signals = {
            "blink": 0.0, "movement": 0.0,
            "mouth": 0.0, "challenge": 0.0,
        }

        # Temporal consistency
        self._consec_live_count = 0
        self._consec_live_required = 5

    def start_challenge(self):
        """Start the interactive challenge-response sequence."""
        self.challenge_detector.start()

    def evaluate(self, frame_bgr, face_location, landmarks_dict=None):
        """Run all anti-spoofing checks on one frame for one face.

        Args:
            frame_bgr: Full BGR frame from OpenCV capture
            face_location: Tuple (top, right, bottom, left) at frame resolution
            landmarks_dict: Result from face_recognition.face_landmarks(), or None

        Returns:
            dict with liveness results and all signal values
        """
        blink_signal = 0.0
        movement_signal = 0.0
        mouth_signal = 0.0
        challenge_result = self.challenge_detector.get_result()

        # --- Landmark-based detectors ---
        if landmarks_dict is not None:
            left_eye = landmarks_dict.get('left_eye')
            right_eye = landmarks_dict.get('right_eye')

            if left_eye and right_eye:
                blink_signal = self.blink_detector.update(left_eye, right_eye)

            movement_signal = self.movement_detector.update(landmarks_dict)
            mouth_signal = self.mouth_detector.update(landmarks_dict)

            # Update challenge-response if active
            if self.challenge_detector.is_active:
                challenge_result = self.challenge_detector.update(
                    landmarks_dict,
                    self.blink_detector.last_ear,
                    self.mouth_detector.last_mar,
                )

        # --- Fusion ---
        challenge_signal = challenge_result['signal']

        liveness_score = (
            self.blink_weight * blink_signal +
            self.movement_weight * movement_signal +
            self.mouth_weight * mouth_signal +
            self.challenge_weight * challenge_signal
        )

        # Temporal consistency
        raw_live = liveness_score >= self.liveness_threshold
        if raw_live:
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
        }

        return {
            "is_live": is_live,
            "liveness_score": liveness_score,
            "blink_signal": blink_signal,
            "movement_signal": movement_signal,
            "mouth_signal": mouth_signal,
            "challenge_signal": challenge_signal,
            "challenge_result": challenge_result,
            "ear": self.blink_detector.last_ear,
            "mar": self.mouth_detector.last_mar,
        }

    def reset(self):
        """Reset all detector state."""
        self.blink_detector.reset()
        self.movement_detector.reset()
        self.mouth_detector.reset()
        self.challenge_detector.reset()
        self.last_liveness_score = 0.0
        self.last_is_live = False
        self._consec_live_count = 0

    def release(self):
        """Release resources. No-op (no external models to close)."""
        pass
