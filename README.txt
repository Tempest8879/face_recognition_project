Face Recognition Project (C++ + Python)
========================================

Architecture:
  - Detection:    MTCNN (facenet-pytorch)
  - Recognition:  ArcFace ONNX model (onnxruntime) — 512-dim embeddings
  - Matching:     Cosine similarity (C++ backend via pybind11)
  - Landmarks:    face_recognition (dlib) — 68-point, for anti-spoofing
  - Anti-spoof:   Blink, movement, depth, challenge-response, FFT

Project Structure:
  face_recognition_project/
  |-- cpp/
  |   |-- face_processor.h      C++ cosine similarity matching (header-only)
  |   |-- bindings.cpp           pybind11 bindings to expose C++ to Python
  |-- python/
  |   |-- face_recognition_app.py  Main application (webcam + image mode)
  |   |-- anti_spoof.py            Anti-spoofing / liveness detection
  |   |-- test_setup.py            Verify environment is set up correctly
  |   |-- face_processor_cpp.pyi   Type stubs for C++ module
  |-- models/
  |   |-- (place ArcFace .onnx model here)
  |-- data/
  |   |-- known_faces/           Put known face images here (name.jpg)
  |   |-- test_images/           Put test images here
  |-- build.bat                  Build script (compiles C++ module)

Dependencies:
  pip install facenet-pytorch onnxruntime face_recognition
  pip install torch opencv-python numpy pybind11 Pillow
  (or onnxruntime-gpu for CUDA acceleration)

ArcFace Model:
  Get from my gdrive:
    https://drive.google.com/file/d/15wKZ9Ub03B_24lxmGMQG-wHtaI0Et88W/view?usp=drive_link
  Or manually download an ArcFace ONNX model and place it in models/:
    https://github.com/deepinsight/insightface/tree/master/model_zoo
  Recommended: w600k_r50.onnx (buffalo_l recognition model)
  The app auto-discovers the first .onnx file in models/

Quick Start:
  1. Download the model:       https://drive.google.com/file/d/15wKZ9Ub03B_24lxmGMQG-wHtaI0Et88W/view?usp=drive_link
  2. Build the C++ module:     build.bat
  2. Test the setup:           python python/test_setup.py
  3. Run with webcam:          python python/face_recognition_app.py

Adding Known Faces:
  Put photos in data/known_faces/ named after the person:
    - john.jpg
    - alice.png
  Each image should contain exactly one face.


FOR LINUX INSTALLATION:

# If run with CPU...
docker compose --profile cpu up --build

# Else GPU (only with NVIDIA GPU + nvidia-container-toolkit)
docker compose --profile gpu up --build

To Verify if Working:

# Health check
curl http://localhost:8000/health

# Recognize a face in an image
curl -X POST -F "file=@some_photo.jpg" http://localhost:8000/recognize

# Register a new face
curl -X POST -F "name=John" -F "file=@john_face.jpg" http://localhost:8000/register

