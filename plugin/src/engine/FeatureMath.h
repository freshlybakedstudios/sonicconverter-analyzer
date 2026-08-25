#pragma once
#include <vector>
#include <string>
#include <map>

// Mirror of audio_analyzer.py::_extract_core for the features that are
// meaningful on a rolling window. Every formula here must match the Python
// implementation numerically — tolerances are enforced by tests/test_parity.py.
// The engine always runs at 44.1 kHz (ANALYSIS_SR); the plugin resamples its
// tap when the session rate differs.

namespace sonic {

constexpr double ANALYSIS_SR = 44100.0;
constexpr int    WINDOW_SAMPLES = 176400; // 4.0 s — the GEMS/mac_worker sample basis

struct WindowFeatures {
    // Stereo
    double stereo_width = 0.0;
    double mid_side_ratio = 1.0;
    double stereo_correlation = 1.0;
    // Loudness
    double lufs_integrated = -30.0;
    double loudness_range = 0.0;
    // 7-band spectrum
    double sub_ratio = 0.0, bass_ratio = 0.0, low_mid_ratio = 0.0, mid_ratio = 0.0;
    double high_mid_ratio = 0.0, presence_ratio = 0.0, air_ratio = 0.0;
    double harmonic_distortion = 0.0;
    // Energy & dynamics
    double energy = 0.0;
    double true_peak_dbfs = -200.0;
    double dynamic_range = 10.0;
    double crest_factor = 1.0;
    double compression_amount = 0.0;
    // Spectral (STFT 2048/512)
    double brightness = 0.0;
    double brightness_variance = 0.0;
    double spectral_rolloff = 0.0;
    double spectral_complexity = 0.0;
    double zcr = 0.0;
    double spectral_flux = 0.0;
    double dissonance = 0.0;

    std::map<std::string, double> asMap() const;
};

// mono is (L+R)/2; left/right may be empty (mono source) — stereo trio then
// keeps its Python defaults (0, 1, 1).
WindowFeatures computeWindowFeatures(const std::vector<double>& mono,
                                     const std::vector<double>& left,
                                     const std::vector<double>& right);

// np.percentile with linear interpolation on an already-sorted vector.
double percentileSorted(const std::vector<double>& sorted, double p);

// Thread-local BandAnalyzer for length n, shared between computeWindowFeatures
// and callers needing extra true-peak passes (the plans are ~30 MB each, so
// one per analysis thread, not per use).
class BandAnalyzer;
BandAnalyzer& sharedBandAnalyzer(int n);

} // namespace sonic
