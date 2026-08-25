#pragma once
#include <vector>

// BS.1770 loudness matching pyloudnorm's implementation exactly (the Python
// side uses pyloudnorm, NOT the spec's 48k coefficients — this build's Meter
// designs a 1500 Hz / +4 dB high shelf and a 38 Hz Q=0.5 high-pass at 44.1k).
// Coefficients below were printed from the installed pyloudnorm at fs=44100.

namespace sonic {

// K-weight a signal in place (both filter stages, zero initial state, like
// scipy.signal.lfilter).
void kWeight(std::vector<double>& x);

// pyloudnorm Meter.integrated_loudness on a mono, already-44.1k signal.
// Returns -HUGE_VAL when everything is gated out (silence), like the Python
// -inf. NaN never escapes.
double integratedLoudnessMono(const std::vector<double>& x);

// audio_analyzer.py loudness_range: p95 - p10 of integrated_loudness over
// 400 ms blocks hopped 200 ms across the window. 0.0 if fewer than 2 blocks.
double loudnessRange(const std::vector<double>& mono);

// Streaming "since reset" integrated loudness over an arbitrary channel count
// (we use stereo). Feed K-weighted-internally; call loudness() any time.
class IntegratedLoudnessStream {
public:
    explicit IntegratedLoudnessStream(int numChannels);
    void reset();
    // samples per channel, non-interleaved
    void push(const std::vector<const double*>& chans, int n);
    double loudness() const;

private:
    struct BiquadState { double z1 = 0.0, z2 = 0.0; };
    int numCh;
    std::vector<BiquadState> shelfState, hpState;
    std::vector<std::vector<double>> pending; // K-weighted tail awaiting block completion
    std::vector<double> blockPowerSum;        // per completed 400ms block: sum over ch of z
    static constexpr int BLOCK = 17640;       // 400 ms @ 44.1k
    static constexpr int HOP = 4410;          // 75% overlap
};

} // namespace sonic
