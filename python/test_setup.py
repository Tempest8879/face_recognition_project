"""
Test script to verify that the face recognition environment is set up correctly.
"""

import sys
import numpy as np

def test_import(module_name, description):
    """Test if a module can be imported."""
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'N/A')
        print(f"  [PASS] {description:30s} -> {module_name} v{version}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {description:30s} -> {e}")
        return False

def main():
    print("=" * 60)
    print("  Face Recognition Environment Test")
    print("=" * 60)

    results = []

    # Core Python packages
    print("\n1. Python Packages:")
    results.append(test_import("cv2", "OpenCV"))
    results.append(test_import("numpy", "NumPy"))
    results.append(test_import("PIL", "Pillow"))
    results.append(test_import("dlib", "dlib"))
    results.append(test_import("face_recognition", "face_recognition"))
    results.append(test_import("pybind11", "pybind11"))

    # C++ module
    print("\n2. C++ Backend Module:")
    cpp_ok = test_import("face_processor_cpp", "C++ FaceProcessor")
    results.append(cpp_ok)

    if cpp_ok:
        import face_processor_cpp
        print("\n3. C++ Module Functional Test:")
        try:
            proc = face_processor_cpp.FaceProcessor()

            fake_encoding_1 = [0.1] * 128
            fake_encoding_2 = [0.1001] * 128
            fake_encoding_3 = [0.9] * 128

            proc.add_known_face("TestPerson", fake_encoding_1)
            assert proc.known_face_count() == 1
            print(f"  [PASS] add_known_face works (count={proc.known_face_count()})")

            match = proc.find_best_match(fake_encoding_2)
            assert match.name == "TestPerson"
            print(f"  [PASS] find_best_match (similar) -> {match}")

            match = proc.find_best_match(fake_encoding_3)
            assert match.name == "Unknown"
            print(f"  [PASS] find_best_match (different) -> {match}")

            batch = proc.find_matches_batch([fake_encoding_2, fake_encoding_3])
            assert len(batch) == 2
            print(f"  [PASS] find_matches_batch -> {len(batch)} results")

            dist = face_processor_cpp.FaceProcessor.euclidean_distance(
                fake_encoding_1, fake_encoding_3
            )
            print(f"  [PASS] euclidean_distance -> {dist:.4f}")

            proc.clear_known_faces()
            assert proc.known_face_count() == 0
            print(f"  [PASS] clear_known_faces works")

        except Exception as e:
            print(f"  [FAIL] C++ functional test failed: {e}")
            results.append(False)
    else:
        print("\n  [SKIP] C++ functional tests skipped (module not compiled)")
        print("         Run build.bat to compile the C++ module.")

    # TextureAnalyzer C++ test
    print("\n4. Anti-Spoofing C++ (TextureAnalyzer):")
    if cpp_ok:
        try:
            import face_processor_cpp
            ta = face_processor_cpp.TextureAnalyzer()

            import random
            random.seed(42)
            random_pixels = [random.randint(0, 255) for _ in range(32 * 32)]
            result = ta.analyze(random_pixels, 32, 32)
            assert 0.0 <= result.texture_score <= 1.0
            print(f"  [PASS] TextureAnalyzer.analyze (random) -> {result}")

            uniform_pixels = [128] * (32 * 32)
            result_uniform = ta.analyze(uniform_pixels, 32, 32)
            assert result_uniform.texture_score < result.texture_score
            print(f"  [PASS] TextureAnalyzer.analyze (uniform) -> {result_uniform}")
            print(f"  [PASS] Random texture scores higher than uniform (expected)")

            hist = ta.compute_lbp_histogram(random_pixels, 32, 32)
            assert len(hist) == 256
            print(f"  [PASS] compute_lbp_histogram -> {len(hist)} bins")

            results.append(True)
        except Exception as e:
            print(f"  [FAIL] TextureAnalyzer test failed: {e}")
            results.append(False)
    else:
        print("  [SKIP] TextureAnalyzer tests skipped (C++ module not compiled)")

    # Anti-spoofing Python module test
    print("\n5. Anti-Spoofing Module Test:")
    try:
        from anti_spoof import (
            AntiSpoofing, BlinkDetector, MovementDetector,
            MouthMovementDetector, ChallengeResponseDetector
        )
        print(f"  [PASS] anti_spoof module imports successfully")

        # Test BlinkDetector with synthetic dlib-format eye landmarks
        bd = BlinkDetector()
        # Simulate open eyes (high EAR)
        left_eye = [(10, 20), (12, 18), (15, 18), (18, 20), (15, 22), (12, 22)]
        right_eye = [(30, 20), (32, 18), (35, 18), (38, 20), (35, 22), (32, 22)]
        signal = bd.update(left_eye, right_eye)
        assert 0.0 <= signal <= 1.0
        assert bd.last_ear > 0
        print(f"  [PASS] BlinkDetector.update -> signal={signal:.2f}, EAR={bd.last_ear:.2f}")

        # Test MovementDetector with synthetic dlib landmarks dict
        md = MovementDetector()
        fake_lm = {
            'nose_tip': [(100, 150), (102, 150), (104, 150), (106, 150), (108, 150)],
            'chin': [(50, 200)] * 9 + [(150, 200)] * 8,
            'left_eye': [(80, 120)] * 6,
            'right_eye': [(120, 120)] * 4,
            'nose_bridge': [(100, 100)] * 4,
        }
        signal = md.update(fake_lm)
        assert 0.0 <= signal <= 1.0
        print(f"  [PASS] MovementDetector.update -> signal={signal:.2f}")

        # Test MouthMovementDetector
        mmd = MouthMovementDetector()
        fake_lm['top_lip'] = [(80, 160), (85, 158), (90, 156), (100, 155),
                              (110, 156), (115, 158), (120, 160),
                              (115, 162), (110, 163), (100, 164), (90, 163)]
        fake_lm['bottom_lip'] = [(120, 160), (115, 168), (110, 170), (100, 172),
                                 (90, 170), (85, 168), (80, 160),
                                 (85, 163), (90, 164), (100, 165), (110, 164)]
        signal = mmd.update(fake_lm)
        assert 0.0 <= signal <= 1.0
        print(f"  [PASS] MouthMovementDetector.update -> signal={signal:.2f}")

        # Test ChallengeResponseDetector
        crd = ChallengeResponseDetector(num_challenges=2, challenge_timeout=5.0)
        assert not crd.is_active
        crd.start()
        assert crd.is_active
        result = crd.get_result()
        assert result['is_active']
        assert result['current_challenge'] is not None
        print(f"  [PASS] ChallengeResponseDetector.start -> challenge={result['current_challenge']}")

        crd.reset()
        assert not crd.is_active
        print(f"  [PASS] ChallengeResponseDetector.reset works")

        # Test AntiSpoofing construction
        asf = AntiSpoofing()
        print(f"  [PASS] AntiSpoofing() constructor works")
        asf.release()
        print(f"  [PASS] AntiSpoofing.release() works")

        results.append(True)
    except Exception as e:
        print(f"  [FAIL] Anti-spoofing module test failed: {e}")
        results.append(False)

    # MTCNN
    print("\n6. MTCNN:")
    mtcnn_ok = test_import("facenet_pytorch", "facenet-pytorch (MTCNN)")
    if not mtcnn_ok:
        print("  [INFO] MTCNN not installed. Install with: pip install facenet-pytorch")
        print("         The system will fall back to dlib CNN face detection.")

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"  ALL TESTS PASSED ({passed}/{total})")
    else:
        print(f"  {passed}/{total} tests passed")
        if not cpp_ok:
            print("  Note: Run build.bat to compile the C++ backend")
    print("=" * 60)

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
