#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

// ============================================================================
// Face Processor - C++ backend for fast face recognition operations
// Uses cosine similarity for ArcFace embedding matching
// ============================================================================

struct FaceMatch {
    std::string name;
    double similarity;
    double confidence;
};
 
class FaceProcessor {
public:
    // Store a known face encoding with a name
    void add_known_face(const std::string& name, const std::vector<double>& encoding) {
        if (encoding.empty()) {
            throw std::invalid_argument("Face encoding must not be empty");
        }
        if (!known_encodings_.empty() && encoding.size() != known_encodings_[0].size()) {
            throw std::invalid_argument("All face encodings must have the same dimensions");
        }
        known_names_.push_back(name);
        known_encodings_.push_back(encoding);
    }

    // Compare a face encoding against all known faces using cosine similarity
    // Returns the best match with similarity and confidence
    FaceMatch find_best_match(const std::vector<double>& unknown_encoding, double threshold = 0.4) const {
        if (known_encodings_.empty()) {
            return {"Unknown", 0.0, 0.0};
        }

        double best_similarity = -1.0;
        int best_index = -1;

        for (size_t i = 0; i < known_encodings_.size(); ++i) {
            double sim = cosine_similarity(known_encodings_[i], unknown_encoding);
            if (sim > best_similarity) {
                best_similarity = sim;
                best_index = static_cast<int>(i);
            }
        }

        FaceMatch match;
        match.similarity = best_similarity;
        // Sigmoid confidence centered on threshold
        // k=15 gives a steep transition around the threshold
        double k = 15.0;
        match.confidence = 1.0 / (1.0 + std::exp(-k * (best_similarity - threshold)));
        // Above threshold and confident enough → use matched name
        if (best_similarity >= threshold && best_index >= 0 && match.confidence >= 0.75) {
            match.name = known_names_[best_index];
        } else {
            match.name = "Unknown";
        }
        return match;
    }

    // Compare all faces in a batch (faster for multiple unknowns)
    std::vector<FaceMatch> find_matches_batch(
        const std::vector<std::vector<double>>& unknown_encodings,
        double threshold = 0.4
    ) const {
        std::vector<FaceMatch> results;
        results.reserve(unknown_encodings.size());
        for (const auto& enc : unknown_encodings) {
            results.push_back(find_best_match(enc, threshold));
        }
        return results;
    }

    // Get the number of known faces
    size_t known_face_count() const {
        return known_names_.size();
    }

    // Clear all known faces
    void clear_known_faces() {
        known_names_.clear();
        known_encodings_.clear();
    }

    // Calculate cosine similarity between two face encodings
    // Returns value in [-1, 1] where 1 means identical
    static double cosine_similarity(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Encoding dimensions must match");
        }
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom < 1e-10) return 0.0;
        return dot / denom;
    }

private:
    std::vector<std::string> known_names_;
    std::vector<std::vector<double>> known_encodings_;
};

// ============================================================================
// Texture Analyzer - Multi-method anti-spoofing for face liveness detection
//
// Combines:
//   1. Laplacian sharpness (real faces are sharper than re-photographed ones)
//   2. LBP micro-texture entropy (real skin has richer local patterns)
//   3. High-frequency energy ratio (screens/prints lose fine detail)
// ============================================================================

struct TextureResult {
    double texture_score;       // 0.0 (likely spoof) to 1.0 (likely real)
    double sharpness;           // Laplacian variance (higher = sharper/more real)
    double lbp_entropy;         // LBP histogram entropy (higher = richer texture)
    double hf_energy;           // High-frequency energy ratio
    int num_unique_patterns;    // Count of distinct LBP codes present
};

class TextureAnalyzer {
public:
    // Thresholds calibrated for 128x128 face ROI from webcam
    // sharpness_threshold: Laplacian variance above which face is likely real
    //   Real webcam faces typically: 200-800+, phone screen photos: 30-150
    // entropy_threshold: LBP entropy above which texture is rich enough
    //   Real faces: 5.5-7.5, flat/blurred photos: 3.0-5.0
    // hf_threshold: High-frequency energy ratio
    //   Real faces: 0.15-0.40, photos through screen: 0.05-0.12
    TextureAnalyzer(double sharpness_threshold = 100.0,
                    double entropy_threshold = 5.0,
                    double hf_threshold = 0.10)
        : sharpness_threshold_(sharpness_threshold)
        , entropy_threshold_(entropy_threshold)
        , hf_threshold_(hf_threshold) {}

    // Analyze a grayscale face ROI for texture authenticity
    TextureResult analyze(const std::vector<uint8_t>& pixels, int width, int height) const {
        if (static_cast<int>(pixels.size()) != width * height) {
            throw std::invalid_argument("Pixel count must equal width * height");
        }
        if (width < 5 || height < 5) {
            throw std::invalid_argument("Image must be at least 5x5");
        }

        // Method 1: Laplacian sharpness (variance of 2nd derivative)
        double sharpness = compute_laplacian_variance(pixels, width, height);

        // Method 2: LBP entropy
        auto histogram = compute_lbp_histogram(pixels, width, height);
        double entropy = compute_entropy(histogram);
        int unique_count = 0;
        for (int bin : histogram) {
            if (bin > 0) unique_count++;
        }

        // Method 3: High-frequency energy ratio
        double hf = compute_hf_energy(pixels, width, height);

        // Score each metric: 0.0 (below threshold) to 1.0 (well above)
        double sharp_score = std::min(1.0, sharpness / sharpness_threshold_);
        double entropy_score = std::min(1.0, entropy / entropy_threshold_);
        double hf_score = std::min(1.0, hf / hf_threshold_);

        // Weighted fusion: sharpness is the strongest spoof discriminator
        double texture_score = 0.50 * sharp_score + 0.30 * entropy_score + 0.20 * hf_score;

        return {texture_score, sharpness, entropy, hf, unique_count};
    }

    // Compute raw 256-bin LBP histogram
    std::vector<int> compute_lbp_histogram(const std::vector<uint8_t>& pixels, int width, int height) const {
        std::vector<int> histogram(256, 0);

        static constexpr int dx[8] = { 0,  1, 1, 1, 0, -1, -1, -1};
        static constexpr int dy[8] = {-1, -1, 0, 1, 1,  1,  0, -1};

        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                uint8_t center = pixels[y * width + x];
                uint8_t code = 0;
                for (int i = 0; i < 8; ++i) {
                    int nx = x + dx[i];
                    int ny = y + dy[i];
                    if (pixels[ny * width + nx] >= center) {
                        code |= (1 << (7 - i));
                    }
                }
                histogram[code]++;
            }
        }
        return histogram;
    }

    // Laplacian variance: measures image sharpness / focus
    // Real faces captured directly have higher sharpness than photos of photos
    double compute_laplacian_variance(const std::vector<uint8_t>& pixels, int width, int height) const {
        // Apply 3x3 Laplacian kernel: [0,1,0; 1,-4,1; 0,1,0]
        double sum = 0.0, sq_sum = 0.0;
        int count = 0;
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                double lap = -4.0 * pixels[y * width + x]
                    + pixels[(y-1) * width + x]
                    + pixels[(y+1) * width + x]
                    + pixels[y * width + (x-1)]
                    + pixels[y * width + (x+1)];
                sum += lap;
                sq_sum += lap * lap;
                count++;
            }
        }
        if (count == 0) return 0.0;
        double mean = sum / count;
        return (sq_sum / count) - (mean * mean);  // variance
    }

    // High-frequency energy: ratio of high-freq to total energy
    // Uses simple horizontal + vertical gradient magnitude
    double compute_hf_energy(const std::vector<uint8_t>& pixels, int width, int height) const {
        double hf_energy = 0.0;
        double total_energy = 0.0;
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                double gx = static_cast<double>(pixels[y * width + (x+1)]) - pixels[y * width + (x-1)];
                double gy = static_cast<double>(pixels[(y+1) * width + x]) - pixels[(y-1) * width + x];
                double grad = gx * gx + gy * gy;
                hf_energy += grad;
                total_energy += static_cast<double>(pixels[y * width + x]) * pixels[y * width + x];
            }
        }
        if (total_energy < 1e-6) return 0.0;
        return hf_energy / total_energy;
    }

private:
    double sharpness_threshold_;
    double entropy_threshold_;
    double hf_threshold_;

    // Shannon entropy of a histogram (measures texture complexity)
    static double compute_entropy(const std::vector<int>& histogram) {
        double total = 0.0;
        for (int v : histogram) total += v;
        if (total < 1.0) return 0.0;

        double entropy = 0.0;
        for (int v : histogram) {
            if (v > 0) {
                double p = static_cast<double>(v) / total;
                entropy -= p * std::log2(p);
            }
        }
        return entropy;  // Max possible: log2(256) = 8.0
    }
};
