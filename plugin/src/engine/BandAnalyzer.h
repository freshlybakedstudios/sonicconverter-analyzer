#pragma once
#include <vector>

// Whole-window rectangular FFT features, mirroring audio_analyzer.py lines
// 726-773: 7 band-energy ratios, harmonic-distortion estimate, and the
// scipy.signal.resample-style 4x-oversampled true peak. One forward real FFT
// of the raw (unwindowed) buffer feeds all three, exactly like np.fft.rfft.

namespace sonic {

struct BandResult {
    double ratios[7] = { 0.10, 0.20, 0.15, 0.25, 0.15, 0.10, 0.05 }; // Python zero-energy defaults
    double harmonic_distortion = 0.0;
    double true_peak_dbfs = -200.0;
};

class BandAnalyzer {
public:
    // n = window length in samples (176,400 live; parity_cli may differ).
    // Both n and 4n must be even (kiss_fftr requirement) — n even suffices.
    explicit BandAnalyzer(int n);
    ~BandAnalyzer();
    BandAnalyzer(const BandAnalyzer&) = delete;
    BandAnalyzer& operator=(const BandAnalyzer&) = delete;

    BandResult analyze(const std::vector<double>& x);

    // 4x-oversampled peak of an arbitrary same-length buffer (per-channel
    // safety true peak). Reuses the FFT plans.
    double truePeakDb(const std::vector<double>& x);

    struct Impl; // public so the file-local free function can use it

private:
    Impl* impl;
};

} // namespace sonic
