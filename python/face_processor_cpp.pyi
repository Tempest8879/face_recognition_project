"""Type stubs for face_processor_cpp C++ extension module."""

from typing import List

class FaceMatch:
    name: str
    similarity: float
    confidence: float
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class FaceProcessor:
    def __init__(self) -> None: ...
    def add_known_face(self, name: str, encoding: List[float]) -> None:
        """Register a known face with a name and encoding vector."""
        ...
    def find_best_match(self, unknown_encoding: List[float], threshold: float = 0.4) -> FaceMatch:
        """Find the best matching known face using cosine similarity."""
        ...
    def find_matches_batch(self, unknown_encodings: List[List[float]], threshold: float = 0.4) -> List[FaceMatch]:
        """Match multiple unknown faces in batch using cosine similarity."""
        ...
    def known_face_count(self) -> int:
        """Get the number of registered known faces."""
        ...
    def clear_known_faces(self) -> None:
        """Remove all known faces."""
        ...
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two encoding vectors."""
        ...

class TextureResult:
    texture_score: float
    sharpness: float
    lbp_entropy: float
    hf_energy: float
    num_unique_patterns: int
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class TextureAnalyzer:
    def __init__(self, sharpness_threshold: float = 100.0, entropy_threshold: float = 5.0, hf_threshold: float = 0.10) -> None:
        """Create a texture analyzer for multi-method anti-spoofing."""
        ...
    def analyze(self, pixels: List[int], width: int, height: int) -> TextureResult:
        """Analyze a grayscale face ROI for texture authenticity."""
        ...
    def compute_lbp_histogram(self, pixels: List[int], width: int, height: int) -> List[int]:
        """Compute raw 256-bin LBP histogram for a grayscale image."""
        ...
    def compute_laplacian_variance(self, pixels: List[int], width: int, height: int) -> float:
        """Compute Laplacian variance (sharpness measure)."""
        ...
    def compute_hf_energy(self, pixels: List[int], width: int, height: int) -> float:
        """Compute high-frequency energy ratio."""
        ...
