Face Recognition Project (C++ + Python)
========================================

Project Structure:
  face_recognition_project/
  |-- cpp/
  |   |-- face_processor.h      C++ face matching engine (header-only)
  |   |-- bindings.cpp           pybind11 bindings to expose C++ to Python
  |-- python/
  |   |-- face_recognition_app.py  Main application (webcam + image mode)
  |   |-- test_setup.py            Verify environment is set up correctly
  |-- data/
  |   |-- known_faces/           Put known face images here (name.jpg)
  |   |-- test_images/           Put test images here
  |-- build.bat                  Build script (compiles C++ module)

Quick Start:
  1. Build the C++ module:     build.bat
  2. Test the setup:           python python/test_setup.py
  3. Run with webcam:          python python/face_recognition_app.py
  4. Run on an image:          python python/face_recognition_app.py path/to/image.jpg

Adding Known Faces:
  Put photos in data/known_faces/ named after the person:
    - john.jpg
    - alice.png
  Each image should contain exactly one face.
