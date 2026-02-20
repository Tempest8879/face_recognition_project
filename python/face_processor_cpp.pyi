"""Type stubs for face_processor_cpp C++ extension module."""

from typing import List

class FaceLocation:
    top: int
    right: int
    bottom: int
    left: int
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class FaceMatch:
    name: str
    distance: float
    confidence: float
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class FaceProcessor:
    def __init__(self) -> None: ...
    def add_known_face(self, name: str, encoding: List[float]) -> None:
        """Register a known face with a name and 128-dim encoding vector."""
        ...
    def find_best_match(self, unknown_encoding: List[float], tolerance: float = 0.6) -> FaceMatch:
        """Find the closest matching known face for an unknown encoding."""
        ...
    def find_matches_batch(self, unknown_encodings: List[List[float]], tolerance: float = 0.6) -> List[FaceMatch]:
        """Match multiple unknown faces in batch (faster)."""
        ...
    def known_face_count(self) -> int:
        """Get the number of registered known faces."""
        ...
    def clear_known_faces(self) -> None:
        """Remove all known faces."""
        ...
    @staticmethod
    def euclidean_distance(a: List[float], b: List[float]) -> float:
        """Compute Euclidean distance between two 128-dim encodings."""
        ...

class TextureResult:
    texture_score: float
    histogram_variance: float
    num_unique_patterns: int
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class TextureAnalyzer:
    def __init__(self, variance_threshold: float = 50.0, min_unique_patterns: int = 100) -> None:
        """Create a texture analyzer for LBP-based anti-spoofing."""
        ...
    def analyze(self, pixels: List[int], width: int, height: int) -> TextureResult:
        """Analyze a grayscale face ROI for texture authenticity."""
        ...
    def compute_lbp_histogram(self, pixels: List[int], width: int, height: int) -> List[int]:
        """Compute raw 256-bin LBP histogram for a grayscale image."""
        ...
