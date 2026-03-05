"""
Face Recognition REST API Server
=================================
FastAPI wrapper around FaceRecognitionApp for server deployment.

Endpoints:
  GET  /health          - Health check
  GET  /faces           - List registered known faces
  POST /recognize       - Upload image, get recognition results
  POST /register        - Register a new face (name + image)
  DELETE /faces/{name}  - Remove a known face

Run:
  uvicorn python.api_server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import io
import shutil
import glob as globmod

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Ensure project root is on the path so imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from face_recognition_app import FaceRecognitionApp, USE_CPP_BACKEND

# --- App initialization ---
app = FastAPI(title="Face Recognition API", version="1.0.0")
recognizer: FaceRecognitionApp = None
KNOWN_FACES_DIR = os.path.join(project_dir, "data", "known_faces")


@app.on_event("startup")
def startup():
    global recognizer
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    recognizer = FaceRecognitionApp(known_faces_dir=KNOWN_FACES_DIR)
    print("[API] Face Recognition API ready.")


def _read_upload_as_bgr(upload: UploadFile) -> np.ndarray:
    """Read an uploaded image file into a BGR numpy array."""
    contents = upload.file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


# --- Endpoints ---

@app.get("/health")
def health():
    count = (
        recognizer.processor.known_face_count()
        if USE_CPP_BACKEND
        else len(recognizer.known_names)
    )
    return {
        "status": "ok",
        "arcface_loaded": recognizer.use_arcface,
        "cpp_backend": USE_CPP_BACKEND,
        "known_faces": count,
    }


@app.get("/faces")
def list_faces():
    """List all registered known faces."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in extensions:
        files.extend(globmod.glob(os.path.join(KNOWN_FACES_DIR, ext)))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return {"faces": sorted(set(names))}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """Upload an image and get face recognition results."""
    try:
        bgr = _read_upload_as_bgr(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Detect faces
    locs, lm5s = recognizer._detect_faces_mtcnn(rgb)

    results = []
    for i, (top, right, bottom, left) in enumerate(locs):
        enc = None
        if recognizer.use_arcface and i < len(lm5s) and lm5s[i] is not None:
            enc = recognizer._get_arcface_embedding(bgr, lm5s[i])

        if enc is not None:
            match = recognizer._match_face(enc)
        else:
            match = {"name": "Unknown", "similarity": 0.0, "confidence": 0.0}

        results.append({
            "name": match["name"],
            "similarity": round(float(match["similarity"]), 4),
            "confidence": round(float(match["confidence"]), 4),
            "box": {"top": int(top), "right": int(right),
                    "bottom": int(bottom), "left": int(left)},
        })

    return {"faces_detected": len(results), "results": results}


@app.post("/register")
async def register(name: str = Form(...), file: UploadFile = File(...)):
    """Register a new known face. Provide a name and an image with one face."""
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Name is required.")

    name = name.strip()

    try:
        bgr = _read_upload_as_bgr(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    locs, lm5s = recognizer._detect_faces_mtcnn(rgb)

    if not locs:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    # Use first detected face
    enc = None
    if recognizer.use_arcface and lm5s[0] is not None:
        enc = recognizer._get_arcface_embedding(bgr, lm5s[0])

    if enc is None:
        raise HTTPException(status_code=500, detail="Could not encode face.")

    # Save image to known_faces dir
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    save_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    cv2.imwrite(save_path, bgr)

    # Register in-memory
    if USE_CPP_BACKEND:
        recognizer.processor.add_known_face(name, enc.tolist())
    else:
        recognizer.known_names.append(name)
        recognizer.known_encodings.append(enc)

    count = (
        recognizer.processor.known_face_count()
        if USE_CPP_BACKEND
        else len(recognizer.known_names)
    )

    return {"registered": name, "total_known_faces": count}


@app.delete("/faces/{name}")
def delete_face(name: str):
    """Remove a known face by name. Requires app restart to take effect in-memory."""
    removed = False
    for ext in ["jpg", "jpeg", "png", "bmp"]:
        path = os.path.join(KNOWN_FACES_DIR, f"{name}.{ext}")
        if os.path.exists(path):
            os.remove(path)
            removed = True

    if not removed:
        raise HTTPException(status_code=404, detail=f"Face '{name}' not found.")

    return {
        "deleted": name,
        "note": "In-memory encodings will be refreshed on next restart.",
    }
