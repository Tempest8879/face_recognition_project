#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "face_processor.h"

namespace py = pybind11;

// ============================================================================
// Pybind11 module - exposes C++ FaceProcessor to Python
// ============================================================================

PYBIND11_MODULE(face_processor_cpp, m) {
    m.doc() = "C++ Face Processor - high-performance face recognition backend";

    // Expose FaceLocation struct
    py::class_<FaceLocation>(m, "FaceLocation")
        .def(py::init<>())
        .def_readwrite("top", &FaceLocation::top)
        .def_readwrite("right", &FaceLocation::right)
        .def_readwrite("bottom", &FaceLocation::bottom)
        .def_readwrite("left", &FaceLocation::left)
        .def("__repr__", [](const FaceLocation& fl) {
            return "FaceLocation(top=" + std::to_string(fl.top) +
                   ", right=" + std::to_string(fl.right) +
                   ", bottom=" + std::to_string(fl.bottom) +
                   ", left=" + std::to_string(fl.left) + ")";
        });

    // Expose FaceMatch struct
    py::class_<FaceMatch>(m, "FaceMatch")
        .def(py::init<>())
        .def_readwrite("name", &FaceMatch::name)
        .def_readwrite("distance", &FaceMatch::distance)
        .def_readwrite("confidence", &FaceMatch::confidence)
        .def("__repr__", [](const FaceMatch& fm) {
            return "FaceMatch(name='" + fm.name +
                   "', distance=" + std::to_string(fm.distance) +
                   ", confidence=" + std::to_string(fm.confidence) + ")";
        });

    // Expose FaceProcessor class
    py::class_<FaceProcessor>(m, "FaceProcessor")
        .def(py::init<>())
        .def("add_known_face", &FaceProcessor::add_known_face,
             py::arg("name"), py::arg("encoding"),
             "Register a known face with a name and 128-dim encoding vector")
        .def("find_best_match", &FaceProcessor::find_best_match,
             py::arg("unknown_encoding"), py::arg("tolerance") = 0.6,
             "Find the closest matching known face for an unknown encoding")
        .def("find_matches_batch", &FaceProcessor::find_matches_batch,
             py::arg("unknown_encodings"), py::arg("tolerance") = 0.6,
             "Match multiple unknown faces in batch (faster)")
        .def("known_face_count", &FaceProcessor::known_face_count,
             "Get the number of registered known faces")
        .def("clear_known_faces", &FaceProcessor::clear_known_faces,
             "Remove all known faces")
        .def_static("euclidean_distance", &FaceProcessor::euclidean_distance,
                     py::arg("a"), py::arg("b"),
                     "Compute Euclidean distance between two 128-dim encodings");

    // ========================================================================
    // Anti-Spoofing: Texture Analyzer (LBP-based)
    // ========================================================================

    // Expose TextureResult struct
    py::class_<TextureResult>(m, "TextureResult")
        .def(py::init<>())
        .def_readwrite("texture_score", &TextureResult::texture_score)
        .def_readwrite("sharpness", &TextureResult::sharpness)
        .def_readwrite("lbp_entropy", &TextureResult::lbp_entropy)
        .def_readwrite("hf_energy", &TextureResult::hf_energy)
        .def_readwrite("num_unique_patterns", &TextureResult::num_unique_patterns)
        .def("__repr__", [](const TextureResult& tr) {
            return "TextureResult(score=" + std::to_string(tr.texture_score) +
                   ", sharpness=" + std::to_string(tr.sharpness) +
                   ", entropy=" + std::to_string(tr.lbp_entropy) +
                   ", hf=" + std::to_string(tr.hf_energy) +
                   ", unique=" + std::to_string(tr.num_unique_patterns) + ")";
        });

    // Expose TextureAnalyzer class
    py::class_<TextureAnalyzer>(m, "TextureAnalyzer")
        .def(py::init<double, double, double>(),
             py::arg("sharpness_threshold") = 100.0,
             py::arg("entropy_threshold") = 5.0,
             py::arg("hf_threshold") = 0.10,
             "Create a texture analyzer for multi-method anti-spoofing")
        .def("analyze", &TextureAnalyzer::analyze,
             py::arg("pixels"), py::arg("width"), py::arg("height"),
             "Analyze a grayscale face ROI for texture authenticity")
        .def("compute_lbp_histogram", &TextureAnalyzer::compute_lbp_histogram,
             py::arg("pixels"), py::arg("width"), py::arg("height"),
             "Compute raw 256-bin LBP histogram for a grayscale image")
        .def("compute_laplacian_variance", &TextureAnalyzer::compute_laplacian_variance,
             py::arg("pixels"), py::arg("width"), py::arg("height"),
             "Compute Laplacian variance (sharpness measure)")
        .def("compute_hf_energy", &TextureAnalyzer::compute_hf_energy,
             py::arg("pixels"), py::arg("width"), py::arg("height"),
             "Compute high-frequency energy ratio");
}
